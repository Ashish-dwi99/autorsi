"""Frozen screenshot development scope; no verification/provider action at import."""
import hashlib
import json
import math
from pathlib import Path
from screen_cases import cases
from vision_edit import GAINS

APP_NAME="adam-1-screen-visual-dev-20261104-v2"
IMPORT_RUN="adam-screen-imports-20261104-v2"
GPU_RUN="adam-screen-visual-dev-20261104-v2"
NOT_AFTER_UNIX=1793880000
CHILD_WALL_SECONDS=1200
NEW_WORK_SECONDS=1140
GENERATION_HARD_SECONDS=35
MAX_REQUESTS=300
MAX_PROGRESS_BYTES=16*1024**2
MAX_VIDEO_BYTES=128*1024**2
MAX_ARTIFACT_BYTES=160*1024**2
DEV_SOURCE_HASH="b44dbbf430face18a694a4a8a3f4f6d8e10f1abf841806a2b9fba5ef401fde85"
FILES=("screen_contract.py","screen_cpu.py","screen_worker.py","screen_entry.py",
       "modal_screen.py","screen_host.py","screen_recover.py","SCOPE.json")
SCREEN_FILES=("screen_cases.py","screen_protocol.py","screen_harness.py","vision_edit.py","screen_schema.py")
ATTENTION_FILES=("dev_contract.py","dev_patch.py","action_schema.py")


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def spec():
    return {"phase":"screen_development","case_manifests":[c.manifest() for c in cases('development')],
        "gains":list(GAINS),"max_model_requests_including_three_warmups":MAX_REQUESTS,
        "max_tokens":128,"max_actions_each":12,"case_seconds":90,"policy_seconds":30,
        "protocol":"screenshot-only normalized0..999 structural-json-v2",
        "coordinate_formula":"floor(n*viewportDimension/1000)","structural_decoding":__import__("screen_schema").schema_receipt(),"verification_executable":False,
        "selection":"all24complete; successes descending; action errors ascending; completion tokens ascending; prefer1then.98then1.02"}


def validate_spec(admission):
    if admission.get('experiment_spec')!=spec():
        raise ValueError('exact screenshot development scope required; verification cannot run')


def validate_attention_selection(selection, report, report_sha256, *, quality_review, quality_sha256):
    """Join actual complete fresh development and separately frozen root selection."""
    from dev_contract import select_development, experiment_spec
    from action_schema import schema_receipt as attention_schema, validate_cpu_grammar as validate_attention_grammar
    child=report.get('child',{})
    if (report.get('status')!='completed' or report.get('run')!='adam-schema-miniwob-20261104-v2'
        or not report.get('child_reaped') or report.get('child_exitcode')!=0
        or digest(report.get('source_manifest'))!=DEV_SOURCE_HASH
        or child.get('status')!='completed_development_only' or child.get('complete_coverage') is not True
        or child.get('browser_close_returned') is not True
        or child.get('experiment_spec')!=experiment_spec()
        or child.get('structural_decoding')!=attention_schema()
        or child.get('verification_performed') is not False):
        raise ValueError('complete exact actual MiniWoB development receipt required')
    validate_attention_grammar(child.get('grammar_preflight',{}))
    actual=select_development(child.get('variants',{}))
    if actual!=child.get('development_selection'):
        raise ValueError('actual development selection does not recompute')
    if (selection.get('schema')!=1 or type(selection.get('selected_attention_scale')) not in (int,float)
        or selection.get('development_report_sha256')!=report_sha256
        or selection.get('source_manifest_sha256')!=DEV_SOURCE_HASH
        or selection.get('selected_attention_scale')!=actual['selected_scale']
        or selection.get('scope')!='development-only attention choice for separate visual calibration'
        or selection.get('development_selection')!=actual
        or selection.get('verification_performed') is not False
        or selection.get('protocol_quality_reviewed') is not True
        or selection.get('quality_review_sha256')!=quality_sha256
        or not isinstance(selection.get('quality_review_path'),str)
        or not selection['quality_review_path'].startswith('evidence/')):
        raise ValueError('hashed root selection must freeze the exact development-selected attention recipe')
    # This is a separately saved root decision, not authority inferred from the
    # mathematical winner. Host reads and hashes its exact named evidence file.
    if (quality_review.get('schema')!=1
        or quality_review.get('development_report_sha256')!=report_sha256
        or quality_review.get('source_manifest_sha256')!=DEV_SOURCE_HASH
        or quality_review.get('complete_episodes')!=36
        or any(type(quality_review.get(key)) is not int or quality_review[key]!=0
               for key in ('malformed_outputs','missing_envelope_outputs','unknown_operations'))
        or quality_review.get('semantic_invalid_actions_reviewed') is not True
        or quality_review.get('carry_forward_to_screen_development') is not True
        or quality_review.get('promotion')!='NONE'):
        raise ValueError('explicit clean-protocol root quality review required')
    return actual['selected_scale']


def gain_order(index):
    if type(index) is not int or not 0<=index<8:raise ValueError('development case index required')
    offset=index%3
    return GAINS[offset:]+GAINS[:offset]


def select_gain(variants):
    expected=[case.case_id for case in cases('development')]
    if set(variants)!={str(g) for g in GAINS}:raise ValueError('three gains required')
    scores=[]
    for gain in GAINS:
        rows=variants[str(gain)]
        if len(rows)!=8 or [row.get('case_id') for row in rows]!=expected:
            raise ValueError('all24paired cases required before visual-gain selection')
        if any(type(r.get('success')) is not bool or type(r.get('action_errors')) is not int
            or r['action_errors']<0 or type(r.get('completion_tokens')) is not int
            or r['completion_tokens']<0 or r.get('token_usage_complete') is not True
            or r.get('abort_session') is not False or r.get('context_close_returned') is not True
            or not r.get('video') or r.get('artifact_error') for r in rows):
            raise ValueError('complete native task/token/video/close receipts required')
        scores.append({'gain':gain,'successes':sum(r['success'] for r in rows),
            'action_errors':sum(r['action_errors'] for r in rows),'completion_tokens':sum(r['completion_tokens'] for r in rows)})
    tie={1.0:0,.98:1,1.02:2}
    selected=sorted(scores,key=lambda x:(-x['successes'],x['action_errors'],x['completion_tokens'],tie[x['gain']]))[0]['gain']
    return {'scores':scores,'selected_gain':selected,'verification_required':True,
        'verification_performed':False,'promotion':'NONE; development selection only',
        'astra_actor_or_comparison':'NONE','prior_official_MiniWoB_score':'21/24 unchanged'}


def source_manifest(root, proposal_root=None):
    root=Path(root);proposal=Path(proposal_root) if proposal_root else root/'proposals/adam_screen_v2'
    packaged=str(root)=='/opt/weight-space'
    screen=root/'screen' if packaged else root/'proposals/adam_screen_v2'
    attention=root/'attention' if packaged else root/'proposals/adam_schema_v2'
    paths={"proposals/adam_screen_v2/"+n:proposal/n for n in FILES}
    paths.update({"proposals/adam_screen_v2/"+n:screen/n for n in SCREEN_FILES})
    paths.update({"proposals/adam_schema_v2/"+n:attention/n for n in ATTENTION_FILES})
    pkg=root/'src/weight_space_lab'
    for path in sorted(pkg.rglob('*')):
        rel=path.relative_to(pkg)
        if '__pycache__' in rel.parts or path.suffix=='.pyc':continue
        if path.is_symlink():raise ValueError('source symlink refused')
        if path.is_file():paths['src/weight_space_lab/'+rel.as_posix()]=path
    output={};total=0
    for name,path in paths.items():
        if path.is_symlink() or not path.is_file():raise ValueError('ordinary source file required: '+name)
        raw=path.read_bytes();total+=len(raw)
        if total>2*1024**2 or len(paths)>80:raise ValueError('bounded source closure exceeded')
        output[name]=hashlib.sha256(raw).hexdigest()
    return output


def executing_join(manifest, modules):
    modules={"proposals/adam_screen_v2/screen_contract.py":__file__,**modules}
    output={}
    for key,path in modules.items():
        actual=hashlib.sha256(Path(path).read_bytes()).hexdigest()
        if actual!=manifest[key]:raise ValueError('executing module differs: '+key)
        output[key]={'path':str(path),'sha256':actual}
    return output


def artifact_inventory(directory):
    """Bound accepted actual saved artifacts; does not bound live encoder allocation."""
    directory=Path(directory);rows=[];total=videos=0
    for path in sorted(directory.rglob('*')):
        if path.is_symlink():raise ValueError('artifact symlink refused')
        if not path.is_file():continue
        if path.suffix not in ('.png','.webm','.json'):raise ValueError('unexpected artifact type')
        size=path.stat().st_size;total+=size
        if path.suffix=='.webm':videos+=size
        if total>MAX_ARTIFACT_BYTES or videos>MAX_VIDEO_BYTES or len(rows)>=400:
            raise ValueError('aggregate artifact bound exceeded')
        h=hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
        rows.append({'relative_path':path.relative_to(directory).as_posix(),'bytes':size,'sha256':h.hexdigest()})
    return {'files':rows,'bytes':total,'video_bytes':videos,
        'bounds_scope':'Returned accepted files; native encoder/OS allocation and failures may exceed before observed stop'}
