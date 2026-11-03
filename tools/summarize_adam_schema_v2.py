"""Saved-only summary for the once Adam structural-schema development run, not new evaluation.

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
PROPOSAL = ROOT / "proposals/adam_schema_v2"
SOURCE_RECORD_SHA256 = "6c271b9ef5b685502b5060c18d0d6a8274ab980472b7530f8abc44b4400cf2da"
# The older standalone auditor uses the same private module names. Never use
# its cached contracts silently in a new-schema audit; CLI invocations isolate.
for _name in ("dev_contract", "dev_benchmark", "dev_native", "action_schema"):
    _existing = sys.modules.get(_name)
    if _existing is not None and Path(_existing.__file__).resolve().parent != PROPOSAL:
        raise RuntimeError("Run this saved-only auditor in an isolated Python process")


def frozen_source():
    record_path=PROPOSAL/'SOURCE_RECORD.json'
    if record_path.is_symlink() or not record_path.is_file() or record_path.stat().st_size>2*1024**2:
        raise ValueError('ordinary bounded frozen source record required')
    raw=record_path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=SOURCE_RECORD_SHA256:
        raise ValueError('frozen source record changed')
    record=json.loads(raw)
    digest=hashlib.sha256(json.dumps(record['source_manifest'],sort_keys=True).encode()).hexdigest()
    if digest!=record['source_manifest_sha256']:
        raise ValueError('source manifest default-space serialization does not join')
    for name,expected in record['source_manifest'].items():
        path=ROOT/name
        if (path.is_symlink() or not path.is_file() or path.stat().st_size!=record['source_files'][name]['bytes']
                or hashlib.sha256(path.read_bytes()).hexdigest()!=expected):
            raise ValueError('reviewed local source bytes changed: '+name)
    return record


# Verify exact local review-bound source before importing any helper module.
_FROZEN_SOURCE = frozen_source()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "proposals/adam_schema_v2")]
from dev_contract import cases, SCALES, select_development, experiment_spec, variant_order, GPU_RUN
from weight_space_lab.benchmarks.miniwob import binary_success
from dev_benchmark import validate_preflight
from action_schema import validate_cpu_grammar, schema_receipt, engine_config, action_schema
from dev_contract import IMPORT_RUN
from dev_native import expected_timer_ms


def reconcile_schema_receipts(outer,cpu,source):
    child=outer['child']; cpu_child=cpu.get('child',{})
    if (cpu.get('status')!='completed' or cpu.get('run')!=IMPORT_RUN
            or cpu.get('child_reaped') is not True or cpu.get('child_exitcode')!=0
            or cpu.get('container_termination_required') is not False
            or cpu_child.get('status')!='imports_passed' or cpu_child.get('run')!=IMPORT_RUN
            or cpu.get('source_manifest')!=source['source_manifest']
            or cpu_child.get('source_manifest')!=source['source_manifest']
            or cpu_child.get('experiment_spec')!=experiment_spec()
            or cpu_child.get('model_calls')!=0 or cpu_child.get('tensor_payloads_loaded')!=0
            or cpu_child.get('cuda_initialized') is not False):
        raise ValueError('exact source-specific saved CPU admission required')
    validate_cpu_grammar(cpu_child['grammar_preflight'])
    if (child.get('grammar_preflight')!=cpu_child['grammar_preflight']
            or child.get('native_preflight')!=cpu_child.get('miniwob_preflight')
            or child.get('runtime_manifest')!=cpu_child.get('runtime_manifest')
            or child.get('structural_decoding')!=schema_receipt()
            or child.get('actual_structured_output_config')!=engine_config()):
        raise ValueError('CPU/GPU grammar, browser or actual engine config join differs')
    if any(row.get('schema_sha256')!=schema_receipt()['sha256'] for row in child['generation_records']):
        raise ValueError('a request used a different schema')
    for case in cases():
        for scale in SCALES:
            row=child['variants'][str(scale)][[c.case_id for c in cases()].index(case.case_id)]
            timer=row.get('reset',{})
            if (row.get('native_timer_ms')!=expected_timer_ms(case)
                    or timer.get('case_id')!=case.case_id or timer.get('env_id')!=case.env_id
                    or timer.get('seed')!=case.seed or timer.get('status')!='accepted'
                    or timer.get('observed_timer_type')!='int'
                    or timer.get('observed_timer_encoding')!='json_scalar'
                    or type(timer.get('observed_timer_value')) is not int
                    or timer.get('observed_timer_value')!=expected_timer_ms(case)
                    or timer.get('expected_native_timer_ms')!=expected_timer_ms(case)):
                raise ValueError('scored reset native timer receipt differs')
    return {'CPU_zero_model_and_CUDA':True,'same_source_spec_and_fresh_seed':True,
            'grammar_conformance_cases':len(cpu_child['grammar_preflight']['cases']),
            'same_schema_all_requests':True,'actual_engine_config':engine_config(),
            'CPU_browser_fingerprint_and_native_timer_receipts_joined':True,
            'limitation':'Saved metadata reconciliation, not independent runtime replay, response repair or physical-stop proof'}


def _error_category(event,row,structural_valid):
    if 'error' not in event:return 'none'
    text=event['error']
    if row['stop_reason']=='stale_queued_action':return 'stale_queued_action'
    if not structural_valid:return 'incomplete_or_structurally_invalid_json'
    if 'ASCII' in text or 'bounded' in text:return 'parser_string_constraint'
    if 'remaining' in text:return 'remaining_action_quota'
    if 'done must' in text:return 'done_order'
    if 'visible' in text or 'text input' in text:return 'target_identity_or_type'
    if 'nonempty' in text:return 'clear_semantics'
    return 'other_strict_parser_error'


def audit_raw_outputs(child):
    """Classify original saved strings, never repair or replace model actions."""
    import jsonschema
    if (child.get('complete_coverage') is not True or 'partial_episode' in child
            or 'native_inference_disposition' in child):
        raise ValueError('raw quality counts require completed lifecycle without unknown operation')
    validator=jsonschema.Draft202012Validator(action_schema())
    def unique(pairs):
        value={}
        for key,item in pairs:
            if key in value:raise ValueError('duplicate key')
            value[key]=item
        return value
    def nonfinite(_):raise ValueError('nonfinite JSON')
    output=[];errors={};stop_reasons={};structural_failures=0;missing_envelopes=0;semantic_errors=0
    generations=child['generation_records'];offset=3
    for index,case in enumerate(cases()):
        for scale in variant_order(index):
            row=child['variants'][str(scale)][index]
            stop_reasons[row['stop_reason']]=stop_reasons.get(row['stop_reason'],0)+1
            requests=generations[offset:offset+row['model_calls']];offset+=row['model_calls']
            request_index=0;events=[]
            for event in row['events']:
                if event['kind']=='queued_dispatch':
                    if 'native_response' in event:raise ValueError('queued action claims a new model response')
                    if 'error' in event:
                        category='stale_queued_action';errors[category]=errors.get(category,0)+1
                        semantic_errors+=1
                        events.append({'dispatch':event['dispatch'],'category':category,'error':event['error']})
                    continue
                raw=event.get('native_response');truncated=event.get('native_response_truncated')
                if not isinstance(raw,str) or len(raw)>4096 or type(truncated) is not bool:
                    raise ValueError('exact bounded returned model response/truncation receipt required')
                request=requests[request_index];request_index+=1
                if request.get('finish_reason') not in ('stop','length','abort'):
                    raise ValueError('unknown actual generation finish reason')
                parse_error=None;parsed=None
                try:
                    parsed=json.loads(raw,object_pairs_hook=unique,parse_constant=nonfinite)
                    structural_valid=validator.is_valid(parsed) and not truncated
                except (ValueError,TypeError) as exc:
                    structural_valid=False;parse_error=str(exc)[:256]
                if not structural_valid:structural_failures+=1
                missing_envelope=parse_error is None and (not isinstance(parsed,dict) or 'action' not in parsed)
                if missing_envelope:missing_envelopes+=1
                category=_error_category(event,row,structural_valid)
                if category!='none':errors[category]=errors.get(category,0)+1
                if category!='none' and structural_valid:semantic_errors+=1
                events.append({'dispatch':event['dispatch'],'call':request['call'],'raw_response':raw,
                    'raw_sha256':hashlib.sha256(raw.encode()).hexdigest(),'saved_response_truncated':truncated,
                    'static_schema_valid':structural_valid,'static_parse_error':parse_error,'missing_envelope':missing_envelope,
                    'parser_error':event.get('error'),'category':category,
                    'finish_reason':request['finish_reason'],'stop_reason':request.get('stop_reason'),
                    'prompt_tokens':request['prompt_tokens'],'output_tokens':request['output_tokens'],
                    'native_action':event.get('native_action')})
            if request_index!=len(requests):raise ValueError('raw response/request count differs')
            output.append({'case_id':case.case_id,'scale':scale,'reward':row['reward'],
                'terminated':row['terminated'],'truncated':row['truncated'],'success':row['success'],
                'stop_reason':row['stop_reason'],'invalid_actions':row['invalid_actions'],'events':events})
    if offset!=len(generations) or len(output)!=36:raise ValueError('all36 raw-output rows required')
    if sum(errors.values())!=sum(row['invalid_actions'] for rows in child['variants'].values() for row in rows):
        raise ValueError('invalid-action categories do not join saved counters')
    scored_structural_failures=structural_failures;scored_missing=missing_envelopes;warmups=[]
    for request,warmup in zip(generations[:3],child['warmups'],strict=True):
        raw=warmup.get('native_response')
        if not isinstance(raw,str) or len(raw)>4096:raise ValueError('saved warmup response missing')
        parsed=None;warmup_parse_error=None
        try:
            parsed=json.loads(raw,object_pairs_hook=unique,parse_constant=nonfinite)
            valid=validator.is_valid(parsed) and len(raw)<4096
        except (ValueError,TypeError) as exc:
            valid=False;warmup_parse_error=str(exc)[:256]
        missing=warmup_parse_error is None and (not isinstance(parsed,dict) or 'action' not in parsed)
        structural_failures+=not valid;missing_envelopes+=missing
        warmups.append({'scale':warmup['scale'],'call':request['call'],'raw_response':raw,
            'raw_sha256':hashlib.sha256(raw.encode()).hexdigest(),'finish_reason':request['finish_reason'],
            'static_schema_valid':valid,'static_parse_error':warmup_parse_error,'missing_envelope':missing,
            'limitation':'Warmup writer caps4096characters without separate truncation flag; not scored.'})
    return {'episodes':output,'neutral_warmups':warmups,'invalid_categories':errors,
            'stop_reasons':stop_reasons,'structurally_invalid_scored_responses':scored_structural_failures,
            'quality_metrics':{'malformed_outputs':structural_failures,
                'missing_envelope_outputs':missing_envelopes,'semantic_invalid_actions':semantic_errors,
                'malformed_scored_outputs':scored_structural_failures,'missing_envelope_scored_outputs':scored_missing,
                'unknown_operations':0,'aborted_generations':sum(row['finish_reason']=='abort' for row in generations),
                'definitions':'Malformed counts all scored+warmup incomplete/truncated/static-schema-invalid outputs; missing-envelope is an overlapping subset of parseable outputs. A4096char warmup is conservatively flagged because its writer lacks a truncation flag. Semantic counts include strict-parser failures after schema-valid output and stale queued actions. Unknown0 is valid only after complete lifecycle reconciliation, never for partial results.'},
            'rescored':False,'response_repair':False,
            'dynamic_target_replay':'Not performed: per-step visible DOM is not fully retained; native receipts remain the scoring evidence',
            'candidate_promotion':'NONE; complete arithmetic selection alone does not establish improvement or general skill'}


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


def summarize(path, cpu_path=None):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size>16*1024**2:
        raise ValueError('ordinary saved GPU report within16MiB required')
    raw = path.read_bytes()
    if len(raw) > 16 * 1024**2:
        raise ValueError("oversized saved receipt")
    outer = json.loads(raw)
    child = outer["child"]
    source = frozen_source()
    if (outer["status"] != "completed" or outer.get("run") != GPU_RUN or child.get("run") != GPU_RUN
            or outer.get("container_termination_required") is not False or outer.get("child_reaped") is not True
            or outer.get("child_exitcode") != 0 or child["status"] != "completed_development_only"
            or child["browser_close_returned"] is not True
            or child["experiment_spec"] != experiment_spec()
            or outer["source_manifest"] != source["source_manifest"]
            or child["source_manifest"] != source["source_manifest"]):
        raise ValueError("complete exact saved development result required")
    cpu_path = Path(cpu_path) if cpu_path is not None else ROOT / "runs" / (IMPORT_RUN + ".remote-report.json")
    if cpu_path.is_symlink() or not cpu_path.is_file() or cpu_path.stat().st_size>16*1024**2:
        raise ValueError('ordinary saved CPU report within16MiB required')
    cpu_raw = cpu_path.read_bytes()
    if len(cpu_raw) > 16 * 1024**2:
        raise ValueError("oversized saved CPU receipt")
    cpu = json.loads(cpu_raw)
    schema_join = reconcile_schema_receipts(outer, cpu, source)
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
        expected_digest=hashlib.sha256(json.dumps(child['patch_audits'][str(event['scale'])], sort_keys=True, allow_nan=False).encode()).hexdigest()
        if digest!=expected_digest:
            raise ValueError("edit event does not join actual retained audit")
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
        "scope": "development-only structural-schema comparison; not capability promotion",
        "schema_receipt_reconciliation": schema_join,
        "cpu_report_sha256": hashlib.sha256(cpu_raw).hexdigest(),
        "raw_output_audit": audit_raw_outputs(child),
        "promotion": "NONE; arithmetic development winner alone is not an improvement claim",
        "not_established": ["perfect general computer use", "Astra superiority", "tenfold performance", "checkpoint export"],
        "independent_result_review_required": True, "root_candidate_quality_assessment_required": True}


if __name__ == "__main__":
    print(json.dumps(summarize(sys.argv[1], sys.argv[2] if len(sys.argv)>2 else None), indent=2, allow_nan=False))
