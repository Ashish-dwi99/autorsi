"""Saved-only summary for the once Adam development run, not new evaluation.

This is a root reconciliation, not a substitute for distinct result review.
Development selection cannot establish a verification/benchmark/Astra win.
"""
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "proposals/adam_dev_v1")]
from dev_contract import cases, SCALES, select_development, experiment_spec, variant_order, GPU_RUN
from weight_space_lab.benchmarks.miniwob import binary_success
from dev_benchmark import validate_preflight


def count(value, label, maximum=None):
    if type(value) is not int or value < 0 or (maximum is not None and value > maximum):
        raise ValueError("invalid count: " + label)
    return value


def seconds(value, label):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("invalid elapsed time: " + label)
    return value


def reconcile_receipts(outer):
    """Reconcile actual returned metadata; never read/download image payloads.

    This validates receipt identity/accounting, not the contents of a screenshot
    or an independent replay of native browser rewards.
    """
    child=outer['child'];generations=child['generation_records'];warmups=child['warmups']
    if (child.get('planned_cases') != 12 or child.get('planned_episodes') != 36
            or child.get('complete_coverage') is not True
            or child.get('environments_constructed') != 36 or child.get('environment_close_returns') != 36
            or 'partial_episode' in child or 'native_inference_disposition' in child):
        raise ValueError('complete36-episode lifecycle required')
    if len(warmups)!=3 or [x['scale'] for x in warmups]!=list(SCALES):
        raise ValueError('three fixed warmup receipts required')
    if len(generations)!=count(child['model_calls'],'all calls',288):
        raise ValueError('generation count differs')
    if [x['call'] for x in generations]!=list(range(1,len(generations)+1)):
        raise ValueError('generation numbering differs')
    for index,request in enumerate(generations):
        count(request['call'],'generation ordinal')
        cap=64 if index<3 else 96
        if type(request['scale']) not in (int,float) or request['scale'] not in SCALES or request['max_tokens']!=cap:
            raise ValueError('generation recipe/token cap differs')
        count(request['prompt_tokens'],'request input tokens')
        count(request['output_tokens'],'request output tokens',cap)
        seconds(request['elapsed_s'],'request')
    for index,(scale,warmup) in enumerate(zip(SCALES,warmups,strict=True)):
        request=generations[index]
        if (request['scope']!='unscored_neutral_warmup' or request['case_id'] is not None
                or request['scale']!=scale or request['prompt_tokens']!=warmup['prompt_tokens']
                or request['output_tokens']!=warmup['completion_tokens']):
            raise ValueError('warmup generation/receipt join differs')
    artifacts=child['artifacts'];artifact_index=0;request_index=3;last_capture=-1.0
    names=set();total_bytes=0;episode_calls=0;episode_steps=0
    for index,case in enumerate(cases()):
        for scale in variant_order(index):
            row=child['variants'][str(scale)][index]
            if row['case_id']!=case.case_id or row['env_id']!=case.env_id or row['seed']!=case.seed:
                raise ValueError('episode identity differs')
            seconds(row['elapsed_s'],'episode')
            calls=count(row['model_calls'],'episode calls',8)
            events=row['events'];dispatches=count(row['policy_dispatches'],'dispatches',8)
            steps=count(row['environment_steps'],'environment steps',8)
            invalid=count(row['invalid_actions'],'invalid actions',1)
            if not 1<=calls<=dispatches or dispatches!=len(events):
                raise ValueError('episode dispatch/call coverage differs')
            if [event['dispatch'] for event in events]!=list(range(1,len(events)+1)):
                raise ValueError('episode dispatch sequence differs')
            if (sum(event['kind']=='inference' for event in events)!=calls
                    or any(event['kind'] not in {'inference','queued_dispatch'} for event in events)
                    or sum('native_action' in event for event in events)!=steps
                    or sum('error' in event for event in events)!=invalid
                    or steps+invalid!=dispatches):
                raise ValueError('action/event accounting differs')
            if row['success'] is not binary_success(row['reward'],row['terminated'],row['truncated']):
                raise ValueError('native reward/success join differs')
            if row['success'] and row['stop_reason']!='native_success':
                raise ValueError('successful episode stop reason differs')
            requests=generations[request_index:request_index+calls];request_index+=calls
            if len(requests)!=calls or any(r['scope']!='scored_miniwob_episode' or r['case_id']!=case.case_id or r['scale']!=scale for r in requests):
                raise ValueError('per-case generation order/identity differs')
            if (sum(r['output_tokens'] for r in requests)!=count(row['completion_tokens'],'episode output tokens')
                    or sum(r['prompt_tokens'] for r in requests)!=count(row['prompt_tokens'],'episode input tokens')
                    or row['token_usage_complete'] is not True):
                raise ValueError('per-case token usage differs')
            episode_calls+=calls;episode_steps+=steps
            for event in events:
                if artifact_index>=len(artifacts):raise ValueError('missing observation receipt')
                artifact=artifacts[artifact_index];artifact_index+=1
                dispatch=event['dispatch'];name=f'case-{index:02d}-scale-{scale}-step-{dispatch}.png'
                if (event['observation_artifact']!=artifact or artifact['relative_path']!=name or name in names
                        or artifact['case_id']!=case.case_id or artifact['scale']!=scale or artifact['dispatch']!=dispatch
                        or artifact['width']!=160 or artifact['height']!=210 or artifact['kind']!='actual_pre_action_observation'):
                    raise ValueError('observation receipt identity/order differs')
                names.add(name);size=count(artifact['bytes'],'image bytes',128*1024)
                digest=artifact['sha256']
                if size==0 or not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):
                    raise ValueError('invalid image size/hash receipt')
                captured=seconds(artifact['elapsed_s'],'observation')
                if captured<last_capture or captured>seconds(child['elapsed_s'],'child'):
                    raise ValueError('observation chronology differs')
                last_capture=captured;total_bytes+=size
    if request_index!=len(generations) or artifact_index!=len(artifacts) or artifact_index>288 or total_bytes>288*128*1024:
        raise ValueError('unaccounted generation/artifact receipt')
    preserved=outer.get('artifact_preservation',{})
    if (preserved.get('files')!=artifact_index or preserved.get('bytes')!=total_bytes
            or preserved.get('kind')!='actual pre-action screenshots with elapsed timestamps; not continuous video'):
        raise ValueError('parent artifact preservation does not reconcile')
    return {'episodes':36,'scored_model_requests':episode_calls,'neutral_warmup_requests':3,
            'environment_steps':episode_steps,'observation_receipts':artifact_index,
            'parent_preserved_image_bytes':total_bytes,
            'payload_verification':'Saved receipt joins only; image payloads not independently recovered or decoded here'}


def summarize(path):
    path = Path(path)
    raw = path.read_bytes()
    if len(raw) > 16 * 1024**2:
        raise ValueError("oversized saved receipt")
    outer = json.loads(raw)
    child = outer["child"]
    source = json.loads((ROOT / "proposals/adam_dev_v1/SOURCE_RECORD_V2.json").read_bytes())
    if (outer["status"] != "completed" or outer.get("run") != GPU_RUN or child.get("run") != GPU_RUN
            or outer.get("container_termination_required") is not False or outer.get("child_reaped") is not True
            or outer.get("child_exitcode") != 0 or child["status"] != "completed_development_only"
            or child["browser_close_returned"] is not True
            or child["experiment_spec"] != experiment_spec()
            or outer["source_manifest"] != source["source_manifest"]
            or child["source_manifest"] != source["source_manifest"]):
        raise ValueError("complete exact saved development result required")
    chosen = select_development(child["variants"])
    if chosen != child["development_selection"]:
        raise ValueError("selection does not reconcile")
    receipt_reconciliation=reconcile_receipts(outer)
    initials=validate_preflight(child['native_preflight'])
    for episodes in child['variants'].values():
        for row in episodes:
            if row['initial']!=initials[row['case_id']]:
                raise ValueError('scored initial screenshot/DOM differs from CPU paired preflight')
    expected_edits=list(SCALES)+[scale for index in range(12) for scale in variant_order(index)]
    events=child['patch_events']
    if [event['scale'] for event in events]!=expected_edits:
        raise ValueError('fixed edit and cache-reset sequence differs')
    for event in events:
        cache=event['cache_reset']
        if any(cache.get(name) is not True for name in ('prefix_reset_returned_true','mm_reset_returned','encoder_reset_returned')):
            raise ValueError('cold cache reset receipt missing')
        digest=event['audit_sha256']
        if not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('invalid retained edit audit hash')
    receipt_reconciliation['cold_cache_reset_receipts']=len(events)
    receipt_reconciliation['initial_observations_join_cpu_preflight']=True
    generations = child["generation_records"]
    warmups = child["warmups"]
    if [x["scale"] for x in warmups] != list(SCALES):
        raise ValueError("missing fixed warmups")
    if not len(generations) == child["model_calls"] <= 288:
        raise ValueError("generation count differs")
    if [x["call"] for x in generations] != list(range(1, len(generations) + 1)):
        raise ValueError("generation numbering differs")
    expected = [case.case_id for case in cases()]
    rows = {}
    for scale in SCALES:
        episodes = child["variants"][str(scale)]
        if [x["case_id"] for x in episodes] != expected:
            raise ValueError("fresh paired task coverage differs")
        requests = [x for x in generations if x["scale"] == scale and x["scope"] == "scored_miniwob_episode"]
        if (len(requests) != sum(x["model_calls"] for x in episodes)
                or sum(x["output_tokens"] for x in requests) != sum(x["completion_tokens"] for x in episodes)
                or sum(x["prompt_tokens"] for x in requests) != sum(x["prompt_tokens"] for x in episodes)):
            raise ValueError("per-variant usage does not reconcile")
        for item in requests:
            if not math.isfinite(item["elapsed_s"]) or item["elapsed_s"] < 0:
                raise ValueError("invalid latency")
        rows[str(scale)] = {"successes": sum(x["success"] for x in episodes), "planned": 12,
            "invalid_actions": sum(x["invalid_actions"] for x in episodes),
            "requests": len(requests), "output_tokens": sum(x["output_tokens"] for x in requests),
            "prompt_tokens": sum(x["prompt_tokens"] for x in requests),
            "sum_episode_seconds": sum(x["elapsed_s"] for x in episodes),
            "median_full_request_seconds": statistics.median(x["elapsed_s"] for x in requests) if requests else None,
            "failed_case_ids": [x["case_id"] for x in episodes if not x["success"]]}
    if sum(x["requests"] for x in rows.values()) + 3 != len(generations):
        raise ValueError("unaccounted request scope")
    inspection = child["weight_value_inspection"]
    if len(inspection["entries"]) != 192 or inspection["parameter_mutations"] != 0:
        raise ValueError("actual value inspection incomplete")
    for scale in SCALES:
        audit = child["patch_audits"][str(scale)]
        if audit["pair_count"] != 192 or audit["scale"] != scale or audit["totals"]["elements"] != 905969664:
            raise ValueError("actual edit scope mismatch")
        if scale == 1.0 and audit["totals"]["changed_elements"] != 0:
            raise ValueError("baseline was changed")
        if scale != 1.0 and audit["totals"]["changed_elements"] <= 0:
            raise ValueError("edit did not change actual weights")
    return {"schema": 1, "development_report_sha256": hashlib.sha256(raw).hexdigest(),
        "source_manifest_sha256": source["source_manifest_sha256"],
        "selected_attention_scale": chosen["selected_scale"], "development_selection": chosen,
        "metrics": rows, "actual_model_requests_including_warmups": len(generations),
        "weight_value_inspection_groups": inspection["groups"],
        "actual_edit_totals": {str(s): child["patch_audits"][str(s)]["totals"] for s in SCALES},
        "receipt_reconciliation":receipt_reconciliation,
        "verification_performed": False, "prior_official_score": "21/24 unchanged",
        "scope": "development-only attention choice for separate visual calibration",
        "not_established": ["perfect general computer use", "Astra superiority", "tenfold performance", "checkpoint export"],
        "independent_result_review_required": True}


if __name__ == "__main__":
    print(json.dumps(summarize(sys.argv[1]), indent=2, allow_nan=False))
