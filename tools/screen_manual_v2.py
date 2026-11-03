"""One real workspace-Astra demonstration using the frozen screen-v2 protocol.

No actor starts at import. Run requires an actual matching Adam case/video,
complete screen development report, root candidate-freeze and saved runtime
attribution. `next` exposes the unchanged PNG plus screenshot-only observations;
`submit` accepts normalized0..999 actions and the frozen harness converts once.
No DOM/source inspection, replay, provider API or inferred model identity.
"""
import argparse
import asyncio
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
SCREEN=ROOT/'proposals/adam_screen_v2'
SOURCE_MANIFEST_SHA256='b7e28dca20e38eb05679a409cad9c1021fb4c14b6218c6f50beea2e7430f5153'
SOURCE_PINS={
 'screen_cases.py':'de5540ecbfabeef02d456d92ed6c7ca34fc07563dbc6160a31d0350747b46364',
 'screen_protocol.py':'255f311a20640ba7bab77252316ede41c381a745c3b89933eece7c42a021895d',
 'screen_harness.py':'ec64ed1ecc232049eadbfe9c920e0a2ca0060f79bde1f7d2073fd7ac732654c9',
 'screen_manual.py':'b6c4ad440108a38f3c2213b9538acac9fc5d7df60b646d6cc6da33ab2e7b807e'}


def read_json(path,cap=16*1024**2):
    path=Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size>cap:
        raise ValueError('ordinary bounded saved JSON required')
    raw=path.read_bytes()
    return json.loads(raw),hashlib.sha256(raw).hexdigest()


def source_modules():
    for name,expected in SOURCE_PINS.items():
        path=SCREEN/name
        if path.is_symlink() or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise ValueError('frozen screenshot source changed: '+name)
        cached=sys.modules.get(path.stem)
        if cached is not None and Path(cached.__file__).resolve()!=path:
            raise ValueError('isolated interpreter required; wrong screenshot module already loaded')
    sys.path.insert(0,str(SCREEN))
    return tuple(importlib.import_module(name) for name in
        ('screen_cases','screen_protocol','screen_harness','screen_manual'))


def validate_freeze(freeze,adam_report,report_hash,adam_case,runtime_identity):
    manifest=adam_report.get('source_manifest',{})
    source_hash=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    child=adam_report.get('child',{})
    if (freeze.get('schema')!=1 or freeze.get('decision')!='frozen_for_demonstration_only'
            or freeze.get('verification_claim') is not False
            or freeze.get('development_report_sha256')!=report_hash
            or freeze.get('source_manifest_sha256')!=SOURCE_MANIFEST_SHA256
            or source_hash!=SOURCE_MANIFEST_SHA256
            or type(freeze.get('selected_attention_scale')) not in (int,float)
            or freeze['selected_attention_scale'] not in (1.0,.5,.25)
            or type(freeze.get('selected_visual_gain')) not in (int,float)
            or freeze['selected_visual_gain'] not in (.98,1.0,1.02)):
        raise ValueError('exact root-frozen demonstration candidate required')
    if (adam_report.get('run')!='adam-screen-visual-dev-20261104-v2'
            or adam_report.get('status')!='completed' or adam_report.get('child_reaped') is not True
            or adam_report.get('child_exitcode')!=0 or adam_report.get('container_termination_required') is not False
            or child.get('source_manifest')!=manifest or child.get('status')!='completed_screen_development_only'
            or child.get('complete_coverage') is not True or child.get('browser_close_returned') is not True
            or child.get('attention_scale')!=freeze['selected_attention_scale']
            or child.get('visual_selection',{}).get('selected_gain')!=freeze['selected_visual_gain']
            or 'partial_case' in child or 'native_disposition' in child):
        raise ValueError('actual completed selected Adam screen report required')
    variants=child.get('variants',{})
    if set(variants)!={'0.98','1.0','1.02'} or any(len(rows)!=8 for rows in variants.values()):
        raise ValueError('all24 actual screen-development rows required')
    selected=variants[str(float(freeze['selected_visual_gain']))]
    matches=[row for row in selected if row.get('case_id')==adam_case.get('case_id')]
    if len(matches)!=1 or matches[0]!=adam_case or adam_case.get('case_id') not in ('screen-development-01','screen-development-04'):
        raise ValueError('exact actual contact/dialog recording from selected candidate required')
    identity=adam_case.get('actor_identity_declared',{})
    if (identity.get('attention_scale')!=freeze['selected_attention_scale']
            or identity.get('visual_gain')!=freeze['selected_visual_gain']
            or adam_case.get('context_close_returned') is not True
            or adam_case.get('policy_pending') is not False or adam_case.get('abort_session') is not False
            or adam_case.get('artifact_error') or not adam_case.get('video')):
        raise ValueError('actual closed Adam recording and declared recipe must join')
    if (runtime_identity.get('schema')!=1 or runtime_identity.get('model_agent')!='Astra'
            or runtime_identity.get('attribution')!='external_workspace_runtime'
            or not isinstance(runtime_identity.get('runtime_identity'),dict)
            or not runtime_identity['runtime_identity']):
        raise ValueError('root-saved workspace-Astra runtime attribution required; CLI cannot verify it')


def prepare(args):
    case_module,protocol,harness,manual=source_modules()
    freeze,freeze_hash=read_json(args.candidate_freeze,65536)
    report,report_hash=read_json(args.adam_report)
    row,row_hash=read_json(args.adam_case_receipt,2*1024**2)
    identity,identity_hash=read_json(args.runtime_identity,65536)
    validate_freeze(freeze,report,report_hash,row,identity)
    case=next(case for case in case_module.demo_cases() if case.case_id==row['case_id'])
    if any(row.get(key)!=value for key,value in case.manifest().items()):
        raise ValueError('exact current fixture/goal/oracle/viewport differs from Adam recording')
    relative=Path(row['video']['path']);folder=Path(args.adam_case_receipt).resolve().parent
    if relative.is_absolute() or '..' in relative.parts:raise ValueError('invalid Adam video path')
    video=folder
    for part in relative.parts:
        video=video/part
        if video.is_symlink():raise ValueError('Adam video symlink refused')
    if (not video.is_file() or not 0<video.stat().st_size<=64*1024**2
            or video.stat().st_size!=row['video']['bytes']):raise ValueError('actual bounded Adam video required')
    digest=hashlib.sha256()
    with video.open('rb') as stream:
        while data:=stream.read(1024*1024):digest.update(data)
    if digest.hexdigest()!=row['video']['sha256']:raise ValueError('actual Adam video hash differs')
    admission={'schema':1,'candidate_freeze_sha256':freeze_hash,'adam_report_sha256':report_hash,
        'adam_case_receipt_sha256':row_hash,'adam_video_sha256':digest.hexdigest(),
        'screen_source_manifest_sha256':SOURCE_MANIFEST_SHA256,'source_pins':SOURCE_PINS,
        'runtime_identity_receipt_sha256':identity_hash,'runtime_attribution':identity,
        'model_identity_verification':'External workspace attribution only; not independently authenticated by this CLI',
        'case_manifest':case.manifest(),'coordinate_convention':'normalized0..999; floor(n*dimension/1000); converted once by frozen harness',
        'limits':{'cases':1,'policy_seconds':30,'case_seconds':90,'actions':12,'batch':4,'tokens':128},
        'composition_positions':{'left':'Astra','right':'Adam-1'},'verification_claim':False}
    return case,admission,(protocol,harness,manual)


async def run(args):
    case,admission,(protocol,harness,manual)=prepare(args)
    from playwright.async_api import async_playwright
    import importlib.metadata
    folder=Path(args.run_dir).resolve()
    if shutil.disk_usage(folder.parent).free<400*1024**2:raise ValueError('fresh400MiB local floor required')
    folder.mkdir(exist_ok=False)
    policy=manual.ManualPolicy(folder/'bridge',args.session_id,timeout_s=30)
    policy.case_id=case.case_id
    identity={'model_agent':'Astra','external_runtime_attribution':admission['runtime_attribution'],
              'runtime_identity_receipt_sha256':admission['runtime_identity_receipt_sha256'],
              'verification':'External attribution, not authenticated by CLI; model tokens/cost remain unknown',
              'interface':'Actual screenshot observation and manually submitted typed action; tool delay included'}
    session=harness.ScreenSession(policy,actor_label='Astra (workspace attribution)',
        actor_identity=identity,limits=protocol.Limits())
    report={'schema':1,'status':'running','session_id':args.session_id,'admission':admission,
            'started_unix':time.time(),'rows':[],'browser_close_returned':False,
            'actual_actor_model_requests':None,'actual_actor_tokens':None,'actual_actor_cost':None,
            'verification_cases_used':0,'demonstration_only':True}
    harness.write_json(folder/'demo.json',report)
    try:
        async with async_playwright() as pw:
            launch={'headless':True}
            if args.browser_executable:
                binary=Path(args.browser_executable)
                if binary.is_symlink() or not binary.is_file():raise ValueError('ordinary explicit browser required')
                launch['executable_path']=str(binary.resolve())
            browser=await asyncio.wait_for(pw.chromium.launch(**launch),30)
            report['browser_version']=browser.version
            report['playwright_version']=importlib.metadata.version('playwright')
            report['browser_executable']=launch.get('executable_path',pw.chromium.executable_path)
            try:
                row=await session.run_case(browser,case,folder/case.case_id,demonstration=True)
                report['rows'].append(row)
                report['status']='completed_demonstration_only'
            finally:
                await asyncio.wait_for(browser.close(),10)
                report['browser_close_returned']=True
                harness.write_json(folder/'demo.json',report)
    except BaseException as exc:
        report.update(status='failed',error={'type':type(exc).__name__,'message':str(exc)[:1000]})
        harness.write_json(folder/'demo.json',report)
        raise
    print(json.dumps({'report':str(folder/'demo.json'),'actual_gameplay_video':row.get('video'),'demonstration_only':True}))


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    start=sub.add_parser('run')
    for name in ('run-dir','session-id','candidate-freeze','adam-report','adam-case-receipt','runtime-identity'):
        start.add_argument('--'+name,required=True)
    start.add_argument('--browser-executable')
    for name in ('next','submit'):
        command=sub.add_parser(name);command.add_argument('--run-dir',required=True)
        if name=='submit':command.add_argument('--response',required=True)
    args=parser.parse_args()
    if args.command=='run':asyncio.run(run(args))
    else:
        _,protocol,_,manual=source_modules();folder=Path(args.run_dir)/'bridge'
        value=({'system_prompt':protocol.SYSTEM_PROMPT,'observation':manual.next_observation(folder),
                'prompt_source_sha256':SOURCE_PINS['screen_protocol.py']}
               if args.command=='next' else manual.submit(folder,args.response))
        print(json.dumps(value,indent=2))


if __name__=='__main__':main()
