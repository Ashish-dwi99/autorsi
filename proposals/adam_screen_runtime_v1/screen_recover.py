"""Once-only bounded saved-artifact recovery; never launches a model/container."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import signal
import time
from screen_host import ROOT,read_json,write_once,verify_sdk
from screen_contract import IMPORT_RUN,GPU_RUN,source_manifest,MAX_ARTIFACT_BYTES,MAX_VIDEO_BYTES


def recover_once(run_id,volume,*,root=ROOT,clock=time.monotonic):
    if run_id not in (IMPORT_RUN,GPU_RUN):raise ValueError('fixed current run required')
    root=Path(root);runs=root/'runs'
    report,report_hash=read_json(runs/(run_id+'.remote-report.json'))
    call,_=read_json(runs/(run_id+'.call.json'))
    if (report.get('run')!=run_id or report.get('function_call_id')!=call.get('function_call_id')
        or report.get('source_manifest')!=source_manifest(root)
        or report.get('status')!='completed' or report.get('container_termination_required') is not False):
        raise ValueError('exact completed saved call/source/report required for artifact recovery')
    inventory=(report.get('child',{}).get('artifacts') or {}).get('files',[])
    if not inventory or len(inventory)>400:raise ValueError('bounded actual artifact inventory required')
    names=set();total=videos=0
    for item in inventory:
        name=item.get('relative_path');size=item.get('bytes');digest=item.get('sha256')
        if (not isinstance(name,str) or name in names or Path(name).is_absolute() or '..' in Path(name).parts
            or Path(name).suffix not in('.webm','.png','.json') or type(size) is not int or not 0<size<=64*1024**2
            or not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest)):
            raise ValueError('invalid artifact declaration')
        names.add(name);total+=size
        if name.endswith('.webm'):videos+=size
    if total>MAX_ARTIFACT_BYTES or videos>MAX_VIDEO_BYTES:raise ValueError('aggregate artifact limit')
    destination=runs/(run_id+'.artifacts')
    if shutil.disk_usage(runs).free<200*1024**2+total:raise RuntimeError('fresh recovery reserve missing')
    destination.mkdir(exist_ok=False)
    receipt={'schema':1,'run':run_id,'report_sha256':report_hash,'declared_bytes':total,
        'files':[],'status':'entered','source_read_attempts_per_file':1,
        'SDK_internal_transfer_retries':'Not bounded by source loop; installed SDK semantics remain separate',
        'video_decode_or_presentation':'NOT_PERFORMED; validate actual media before publishing'}
    write_once(destination/'recovery-started.json',receipt)
    started=clock();remaining=total
    try:
        for item in inventory:
            if clock()-started>=300:raise TimeoutError('original recovery300s deadline')
            if shutil.disk_usage(destination).free<200*1024**2+remaining:raise RuntimeError('fresh per-file recovery reserve missing')
            path=destination/item['relative_path'];path.parent.mkdir(parents=True,exist_ok=True)
            h=hashlib.sha256();count=0
            with path.open('xb') as stream:
                for block in volume.read_file(run_id+'/artifacts/'+item['relative_path']):
                    if clock()-started>=300:raise TimeoutError('original recovery300s deadline')
                    count+=len(block)
                    if count>item['bytes']:raise ValueError('actual file exceeds declared bound')
                    stream.write(block);h.update(block)
            if count!=item['bytes'] or h.hexdigest()!=item['sha256']:raise ValueError('saved artifact length/hash mismatch')
            remaining-=count;receipt['files'].append(dict(item))
        receipt['status']='completed';receipt['elapsed_s']=clock()-started
        write_once(destination/'recovery-completed.json',receipt)
        return receipt
    except BaseException as exc:
        receipt.update(status='failed',error={'type':type(exc).__name__,'message':str(exc)[:1000]},
            partial_files_preserved=True,retry_authorized=False)
        write_once(destination/'recovery-failed.json',receipt);raise


def main():
    parser=argparse.ArgumentParser();parser.add_argument('run',choices=(IMPORT_RUN,GPU_RUN));args=parser.parse_args()
    import modal
    verify_sdk(modal)
    previous=signal.getsignal(signal.SIGALRM)
    def deadline(*args):raise TimeoutError('recovery300s hard host deadline')
    signal.signal(signal.SIGALRM,deadline);signal.setitimer(signal.ITIMER_REAL,300)
    try:print(json.dumps(recover_once(args.run,modal.Volume.from_name('bu-weight-space-results-v3'))))
    finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)

if __name__=='__main__':main()
