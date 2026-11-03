"""One loaded worker: fixed attention recipe, three reversible visual-merger gains.
No model/provided action occurs at module import. Native uncertainty is fatal.
"""
from pathlib import Path
import hashlib
import json
import time
from weight_space_lab.diagnostic import atomic_json
from screen_contract import (GAINS,GPU_RUN,MAX_REQUESTS,NEW_WORK_SECONDS,validate_spec,
    validate_attention_selection,gain_order,select_gain,executing_join,artifact_inventory,
    MAX_ARTIFACT_BYTES,MAX_VIDEO_BYTES)


def inspect_visual_values(editor, torch):
    """Actual two retained BF16 tensors; hashes/norms do not imply capability."""
    rows=[]
    with torch.no_grad():
        for name,original in editor.originals.items():
            raw=original.contiguous().view(torch.uint8).numpy().tobytes()
            if len(raw)>20*1024**2:raise ValueError("visual tensor byte bound")
            value=original.to(dtype=torch.float32)
            if not bool(torch.isfinite(value).all().item()):raise ValueError("nonfinite visual values")
            squared=float(torch.sum(value.square(),dtype=torch.float64).item())
            maximum=float(value.abs().max().item())
            import math
            if not math.isfinite(squared) or not math.isfinite(maximum):raise ValueError("nonfinite visual norm")
            rows.append({"name":name,"shape":list(original.shape),"bytes":len(raw),
                "sha256":hashlib.sha256(raw).hexdigest(),"l2":math.sqrt(squared),"max_abs":maximum})
    if len(rows)!=2:raise ValueError("exact main-merger weight and bias required")
    return {"scope":"actual retained main visual merger output weight and bias only",
        "targets":rows,"parameter_mutations":0,"capability_inference":"NONE"}


def require_idle(engine):
    method=getattr(engine.llm_engine,"has_unfinished_requests",None)
    if not callable(method) or method() is not False:raise RuntimeError("idle engine required")


def run_screen_development(admission, progress_path):
    import asyncio,base64,io,tempfile,math
    from PIL import Image
    import torch
    from safetensors.torch import load_file
    from vllm import LLM
    from screen_schema import engine_config,sampling_parameters,schema_receipt,validate_cpu_grammar,runtime_backend_identity,verify_engine_config
    from playwright.async_api import async_playwright
    from weight_space_lab.vllm018_compat import normalize_config,install_single_process_loader
    from weight_space_lab.vllm_attention import AttentionView
    from dev_patch import MergedWeightEditor
    from weight_space_lab.patch import file_sha256
    from weight_space_lab.policy_io import processor_manifest
    from weight_space_lab.diagnostic import reconcile_fixture_admission
    import screen_worker,screen_cpu,screen_cases,screen_harness,screen_protocol,vision_edit,screen_schema,dev_patch,dev_contract,action_schema
    from screen_cases import cases
    from screen_harness import ScreenSession
    from screen_protocol import Reply,Limits,SessionAborted,parse_actions
    from vision_edit import VisualMergerEditor,invalidate_after_edit
    from screen_cpu import validate_preflight,browser_runtime_manifest
    started=time.monotonic();native_pending=False
    artifact_dir=Path(progress_path).parent/'artifacts';artifact_dir.mkdir(exist_ok=False)
    report={"schema":1,"run":GPU_RUN,"status":"running","stage":"source_admission",
        "source_manifest":admission["source_manifest"],"experiment_spec":admission["experiment_spec"],
        "model_calls":0,"model_call_counter_scope":"Admitted chat request attempts before to_thread, including3warmups; internal startup/profiling forwards excluded",
        "generation_records":[],"warmups":[],"variants":{str(g):[] for g in GAINS},
        "visual_edit_events":[],"planned_cases":8,"planned_episodes":24,
        "browser_close_returned":False,"verification_performed":False,"astra_actor":"NONE",
        "limits":{"child_wall_seconds":1200,"new_work_before_seconds":1140,"model_requests":300,
            "generation_watchdog_seconds":35,"case_seconds":90,"call_seconds":30,"close_seconds":10},
        "structural_decoding":schema_receipt(),"checkpoint_exported":False,"official_prior_MiniWoB_score":"21/24 unchanged"}
    def save():
        report['elapsed_s']=time.monotonic()-started;atomic_json(progress_path,report)
    def stage(name):
        report['stage']=name;report['stage_started_monotonic']=time.monotonic();save()
    def admit():
        if time.monotonic()-started>=NEW_WORK_SECONDS:raise RuntimeError("original1140s new-work gate closed")
    def check_artifact_sizes():
        total=videos=0
        for path in artifact_dir.rglob('*'):
            if path.is_file():
                size=path.stat().st_size;total+=size
                if path.suffix=='.webm':videos+=size
        if total>MAX_ARTIFACT_BYTES or videos>MAX_VIDEO_BYTES:raise RuntimeError("aggregate artifact bound")
    try:
        stage('source_admission');validate_spec(admission)
        scale=validate_attention_selection(admission['attention_selection'],admission['attention_development_report'],
                                            admission['attention_development_report_sha256'], quality_review=admission["attention_quality_review"], quality_sha256=admission["attention_quality_review_sha256"])
        report['attention_selection']=admission['attention_selection'];report['attention_scale']=scale
        report['attention_quality_review']=admission['attention_quality_review']
        report['attention_quality_review_sha256']=admission['attention_quality_review_sha256']
        expected_initials=validate_preflight(admission['screen_preflight'])
        validate_cpu_grammar(admission['grammar_preflight'])
        if runtime_backend_identity()!=admission['grammar_preflight']['runtime']:
            raise ValueError('structured decoder runtime differs from fresh CPU receipt')
        report['grammar_preflight']=admission['grammar_preflight']
        report['executing_source_join']=executing_join(admission['source_manifest'],{
            'proposals/adam_screen_v2/'+module.__name__+'.py':module.__file__
            for module in (screen_worker,screen_cpu) } | {
            'proposals/adam_screen_v2/'+module.__name__+'.py':module.__file__
            for module in (screen_cases,screen_harness,screen_protocol,vision_edit,screen_schema)} | {
            'proposals/adam_schema_v2/'+module.__name__+'.py':module.__file__ for module in (dev_patch,dev_contract,action_schema)})
        merged, adapter = Path("/merged/bu-30b-ft"), Path("/adapter/adapter")
        base = Path("/base/models--browser-use--bu-30b-a3b-preview/snapshots") / admission["revision"]
        if reconcile_fixture_admission(admission["fixtures"]["fixture_manifest"]) != admission["runner_reconciliation"]:
            raise ValueError("existing source reconciliation changed")
        if processor_manifest(base) != admission["fixtures"]["processor_manifest"]:
            raise ValueError("base processor artifacts changed")
        for name, expected in admission["merged_metadata_pins"].items():
            if file_sha256(merged / name)["sha256"] != expected:
                raise ValueError("merged metadata changed")
        for name, expected in admission["adapter_pins"].items():
            if file_sha256(adapter / name)["sha256"] != expected:
                raise ValueError("adapter artifact changed")
        report["source"] = admission["prior_source"]
        overlay = Path(tempfile.mkdtemp(prefix="adam-screen-bu-metadata-"))
        normalized = json.dumps(normalize_config((merged / "config.json").read_bytes()), sort_keys=True).encode()
        (overlay / "config.json").write_bytes(normalized)
        report["normalized_config_sha256"] = hashlib.sha256(normalized).hexdigest()
        index_bytes = (merged / "model.safetensors.index.json").read_bytes()
        for name in {"model.safetensors.index.json", *json.loads(index_bytes)["weight_map"].values()}:
            if Path(name).name != name:
                raise ValueError("invalid checkpoint filename")
            (overlay / name).symlink_to(merged / name)
        for name in list(admission["fixtures"]["processor_manifest"]) + ["processor_config.json"]:
            if name != "config.json" and (base / name).is_file():
                (overlay / name).symlink_to(base / name)
        report["loader_hook"] = install_single_process_loader(index_bytes)
        torch.manual_seed(20261103)
        torch.set_num_threads(4)
        stage("model_load")
        load_start = time.monotonic()
        engine = LLM(model=str(overlay), tokenizer=str(base), trust_remote_code=False,
                     dtype="bfloat16", quantization=None, tensor_parallel_size=1,
                     pipeline_parallel_size=1, distributed_executor_backend="uni",
                     gpu_memory_utilization=0.92, max_model_len=8192, max_num_seqs=1,
                     max_num_batched_tokens=8192, enforce_eager=True, enable_prefix_caching=False,
                     enable_chunked_prefill=False, limit_mm_per_prompt={"image": 1},
                     mm_processor_kwargs={"max_pixels": 1100 * 850}, generation_config="vllm",
                     seed=20261103, disable_log_stats=True,structured_outputs_config=engine_config())
        report['actual_structured_output_config']=verify_engine_config(engine)
        report["model_load_s"] = time.monotonic() - load_start
        report["gpu"] = torch.cuda.get_device_name(0)
        report["load_receipts"] = engine.apply_model(lambda model: model._bu_merged_compat_receipt)
        if len(report["load_receipts"]) != 1 or not report["load_receipts"][0].get("complete"):
            raise ValueError("complete single-worker load receipt required")
        def prepare_editors(model):
            model.eval()
            model._adam_attention=MergedWeightEditor(AttentionView(model),
                load_file(str(adapter/'adapter_model.safetensors'),device='cpu'),
                json.loads((adapter/'adapter_config.json').read_text()))
            attention_inspection=model._adam_attention.inspect_values()
            attention_edit=model._adam_attention.apply(scale)
            model._adam_visual=VisualMergerEditor(model)
            visual_inspection=inspect_visual_values(model._adam_visual,torch)
            return {'attention_inspection':attention_inspection,'attention_edit':attention_edit,
                    'visual_inspection':visual_inspection}
        require_idle(engine);stage('retain_inspect_and_fixed_attention_edit')
        prepared=engine.apply_model(prepare_editors)
        if len(prepared)!=1:raise ValueError('single local worker required')
        report['prepared_editors']=prepared[0]
        report['attention_cache_reset']=invalidate_after_edit(engine);require_idle(engine);save()
        visual_norms={row['name']:row['l2'] for row in prepared[0]['visual_inspection']['targets']}
        def apply_gain(gain):
            admit()
            if native_pending:raise RuntimeError('no visual edit while request disposition unknown')
            require_idle(engine);stage('visual_weight_edit')
            audits=engine.apply_model(lambda model:model._adam_visual.apply(gain))
            if len(audits)!=1:raise ValueError('single worker edit receipt required')
            cache=invalidate_after_edit(engine);require_idle(engine)
            for row in audits[0]['targets']:
                norm=visual_norms[row['name']]
                row['relative_l2_delta']=row['l2_delta']/norm if norm>0 else None
                if row['relative_l2_delta'] is not None and not math.isfinite(row['relative_l2_delta']):
                    raise ValueError('nonfinite actual relative visual change')
            report['current_gain']=gain
            report['visual_edit_events'].append({'gain':gain,'audit':audits[0],'cache_reset':cache})
            save()
        async def infer(messages,cap):
            nonlocal native_pending
            admit()
            if native_pending or report['model_calls']>=MAX_REQUESTS or cap not in(64,128):
                raise RuntimeError('request admission refused')
            require_idle(engine);report['model_calls']+=1;native_pending=True;stage('generation')
            call_started=time.monotonic()
            try:
                returned=await asyncio.to_thread(engine.chat,messages,
                    sampling_parameters(cap),use_tqdm=False)
            except BaseException:
                report['native_disposition']='UNKNOWN; original generation watchdog remains active';save();raise
            native_pending=False
            if len(returned)!=1 or not returned[0].finished or len(returned[0].outputs)!=1:
                raise RuntimeError('complete single generation receipt required')
            output=returned[0].outputs[0]
            report['generation_records'].append({'request':report['model_calls'],
                'gain':report['current_gain'],'case_id':report.get('current_case'),'scope':report['request_scope'],
                'elapsed_s':time.monotonic()-call_started,'prompt_tokens':len(returned[0].prompt_token_ids),
                'completion_tokens':len(output.token_ids),'max_tokens':cap,
                'schema_sha256':schema_receipt()['sha256'],
                'finish_reason':output.finish_reason,'stop_reason':output.stop_reason})
            stage('browser_work');return Reply(output.text,len(returned[0].prompt_token_ids),len(output.token_ids))
        async def evaluate():
            blank=io.BytesIO();Image.new('RGB',(390,844),'white').save(blank,format='PNG')
            warm=[{'role':'user','content':[{'type':'text','text':'Unscored blank-screen warmup. Return {"actions":[{"done":{}}]} only.'},
                {'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(blank.getvalue()).decode()}}]}]
            for gain in GAINS:
                apply_gain(gain);report['request_scope']='neutral_warmup';report['current_case']=None
                reply=await asyncio.wait_for(infer(warm,64),30)
                report['warmups'].append({'gain':gain,'text':reply.text[:4096],
                    'prompt_tokens':reply.prompt_tokens,'completion_tokens':reply.completion_tokens});save()
                if parse_actions(reply.text,390,844)!=[{'done':{}}]:
                    raise ValueError('neutral structured warmup did not return the frozen done response')
            async with async_playwright() as pw:
                actual=browser_runtime_manifest(pw.chromium.executable_path)
                if actual!=admission['screen_preflight']['runtime_manifest']:
                    raise ValueError('browser executable/runtime differs after CPU')
                browser=await asyncio.wait_for(pw.chromium.launch(headless=True,executable_path=pw.chromium.executable_path),30)
                report['browser_version']=browser.version
                if browser.version!=admission['screen_preflight']['browser_version']:
                    raise ValueError('actual Chromium version differs after CPU')
                session=ScreenSession(infer,actor_label='Adam-1 experimental visual gain',limits=Limits())
                try:
                    for index,case in enumerate(cases('development')):
                        for gain in gain_order(index):
                            admit();apply_gain(gain);check_artifact_sizes()
                            report['current_case']=case.case_id;report['request_scope']='screen_development'
                            session.actor_identity={'target_checkpoint':admission['prior_source'],
                                'attention_scale':scale,'visual_gain':gain,'provider':'private Modal/vLLM'}
                            def publish(row):
                                frames=row.get('frames',[])
                                if frames and frames[0]['sha256']!=expected_initials[case.case_id]:
                                    raise ValueError('actual first screen differs from fresh pairedCPU render')
                                report['partial_case']=row
                                # Crucially do not replace generation stage/time while native request is pending.
                                save()
                            result=await session.run_case(browser,case,artifact_dir/f'case-{index:02d}-gain-{gain}',
                                demonstration=case.family in('contact','dialog'),publish=publish)
                            if not result.get('context_close_returned') or result.get('artifact_error') or not result.get('video'):
                                raise RuntimeError('complete actual video/close receipt required')
                            report['variants'][str(gain)].append(result);report.pop('partial_case',None)
                            check_artifact_sizes();save()
                finally:
                    if not native_pending:stage('finalizing_browser')
                    await asyncio.wait_for(browser.close(),10)
                    report['browser_close_returned']=True;save()
        async def guarded():
            try:await evaluate()
            except BaseException as exc:
                report['status']='incomplete';report['error']={'type':type(exc).__name__,'message':str(exc)[:2000]}
                if isinstance(exc,SessionAborted):report['partial_case']=exc.result
                if native_pending:save()
                else:stage('failed')
                raise
        asyncio.run(guarded())
        report['visual_selection']=select_gain(report['variants'])
        require_idle(engine);apply_gain(1.0)
        report['restored_visual_gain_one']=True
        report['artifacts']=artifact_inventory(artifact_dir)
        report['status']='completed_screen_development_only';report['complete_coverage']=True;stage('completed')
    except BaseException as exc:
        report['status']='incomplete';report['error']={'type':type(exc).__name__,'message':str(exc)[:2000]}
        if native_pending:save()
        else:stage('failed')
