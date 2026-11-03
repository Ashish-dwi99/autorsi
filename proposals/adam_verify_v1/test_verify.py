"""Pure fake tests; no model, browser, provider, game or native tensor runtime."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'tests'))
from test_merged_patch import FakeTorch, inputs, bytes_of
from verify_patch import MergedWeightEditor, validate_scale
from verify_contract import (cases,scales,variant_order,compare_verification,experiment_spec,
    validate_spec,validate_candidate,VERIFICATION_SEED,source_manifest,executing_source_join)
from verify_contract import reset_variant_caches
from verify_native import parse_batch,visible_observation
from verify_host import dispatch_once,alarm_handlers,IMPORT_RUN,APP_NAME
from test_verify_native import observation


def fake_candidate_admission():
    import action_schema,development_contract
    from verify_contract import DEVELOPMENT_SOURCE_SHA256
    grammar={'status':'passed','schema':action_schema.schema_receipt(),'vocab_size':151936,
        'model_calls':0,'model_instantiations':0,'cuda_initialized':False,
        'runtime':{'versions':{'vllm':'0.18.0','xgrammar':'0.2.8'},'python_sources':{'TEST_ONLY':{}}},
        'tokenizer_class':'TEST_ONLY','tokenizer_path':'TEST_ONLY',
        'cases':[{'name':r['name'],'text_sha256':hashlib.sha256(r['text'].encode()).hexdigest(),
            'complete':r['expected_complete'],'token_complete':r['expected_complete']}
            for r in action_schema.conformance_examples()]}
    manifest=json.loads((ROOT/'proposals/adam_schema_v2/SOURCE_RECORD.json').read_text())['source_manifest']
    variants={str(scale):[{'case_id':c.case_id,'success':scale==.5,'invalid_actions':0,
        'completion_tokens':10,'token_usage_complete':True} for c in development_contract.cases()]
        for scale in development_contract.SCALES}
    chosen=development_contract.select_development(variants)
    report={'run':'adam-schema-miniwob-20261104-v2','status':'completed','source_manifest':manifest,
        'child_reaped':True,'child_exitcode':0,'child':{'status':'completed_development_only',
        'complete_coverage':True,'browser_close_returned':True,'verification_performed':False,
        'experiment_spec':development_contract.experiment_spec(),'structural_decoding':action_schema.schema_receipt(),
        'grammar_preflight':grammar,'variants':variants,'development_selection':chosen}}
    selection={'schema':1,'purpose':'reserved-miniwob-verification','development_report_sha256':'a'*64,
        'source_manifest_sha256':DEVELOPMENT_SOURCE_SHA256,'selected_attention_scale':.5,
        'development_selection':chosen,'protocol_quality_reviewed':True,'verification_previously_performed':False,
        'quality_review_sha256':'b'*64,'quality_review_path':'evidence/TEST_ONLY.json'}
    quality={'schema':1,'development_report_sha256':'a'*64,'source_manifest_sha256':DEVELOPMENT_SOURCE_SHA256,
        'complete_episodes':36,'malformed_outputs':0,'missing_envelope_outputs':0,'unknown_operations':0,
        'semantic_invalid_actions_reviewed':True,'carry_forward_to_miniwob_verification':True,'promotion':'NONE'}
    return {'development_report':report,'development_report_sha256':'a'*64,
        'candidate_selection':selection,'candidate_selection_sha256':'c'*64,
        'quality_review':quality,'quality_review_sha256':'b'*64,'experiment_spec':experiment_spec(.5)}


class EditTests(unittest.TestCase):
    def test_inspect_then_actual_quarter_edit_and_bitwise_restore(self):
        model,adapter,config,weights=inputs()
        original={k:bytes_of(v) for k,v in weights.items()}
        torch=FakeTorch();editor=MergedWeightEditor(model,adapter,config,torch_module=torch)
        with self.assertRaisesRegex(ValueError,'inspection'):
            editor.apply(.25)
        report=editor.inspect_values()
        self.assertEqual(report['groups']['all']['matrices'],192)
        self.assertEqual(len(report['entries']),192)
        self.assertEqual({k:bytes_of(v) for k,v in weights.items()},original)
        self.assertEqual(report['parameter_mutations'],0)
        for scale in (.25,.5,.25):
            audit=editor.apply(scale)
            self.assertEqual(audit['totals']['changed_elements'],768)
            for weight in weights.values():
                self.assertEqual(weight.rows,[[10+(scale-1)*8,(scale-1)*-8],
                    [20+(scale-1)*16,30+(scale-1)*-16]])
        editor.apply(1)
        self.assertEqual({k:bytes_of(v) for k,v in weights.items()},original)
        with self.assertRaises(ValueError):editor.inspect_values()

    def test_nonfinite_inspection_and_partial_edit_stop(self):
        model,adapter,config,weights=inputs();torch=FakeTorch()
        editor=MergedWeightEditor(model,adapter,config,torch_module=torch)
        editor.inspect_values();torch.fail_after=torch.matmul_calls+2
        with self.assertRaises(RuntimeError):editor.apply(.25)
        with self.assertRaisesRegex(ValueError,'incomplete'):editor.apply(.5)
        torch.fail_after=None;editor.apply(1)
        for scale in (True,.75,0,float('nan'),float('inf')):
            with self.assertRaises(ValueError):validate_scale(scale)


class ContractTests(unittest.TestCase):
    def test_repeated_pair_scale_still_gets_fresh_caches(self):
        from types import SimpleNamespace
        from verify_benchmark import prepare_variant
        events=[]
        engine=SimpleNamespace(reset_prefix_cache=lambda:events.append('prefix') or True,
            reset_mm_cache=lambda:events.append('mm'),
            llm_engine=SimpleNamespace(has_unfinished_requests=lambda:False,
                reset_encoder_cache=lambda:events.append('encoder')))
        def edit(value):events.append(('edit',value));return {'test_audit':value}
        receipt=prepare_variant(engine,.5,.5,.5,edit)
        self.assertFalse(receipt['weight_edit_performed'])
        self.assertIsNone(receipt['audit'])
        self.assertEqual(events,['prefix','mm','encoder'])
        events.clear();receipt=prepare_variant(engine,1.0,.5,.5,edit)
        self.assertTrue(receipt['weight_edit_performed'])
        self.assertEqual(events,[('edit',1.0),'prefix','mm','encoder'])
        events.clear()
        with self.assertRaises(ValueError):prepare_variant(engine,.25,.5,.5,edit)
        self.assertEqual(events,[])
        engine.llm_engine.has_unfinished_requests=lambda:None
        with self.assertRaises(RuntimeError):prepare_variant(engine,.5,.5,.5,edit)
        self.assertEqual(events,[])
        engine.llm_engine.has_unfinished_requests=lambda:False
        engine.reset_prefix_cache=lambda:events.append('prefix') or False
        with self.assertRaises(RuntimeError):prepare_variant(engine,.5,.5,.5,edit)
        self.assertEqual(events,['prefix'])

    def test_equal_cache_reset_and_failed_prefix_refuses_later_calls(self):
        from types import SimpleNamespace
        events=[]
        engine=SimpleNamespace(reset_prefix_cache=lambda:events.append('prefix') or True,
            reset_mm_cache=lambda:events.append('mm'),
            llm_engine=SimpleNamespace(reset_encoder_cache=lambda:events.append('encoder')))
        self.assertTrue(reset_variant_caches(engine)['encoder_reset_returned'])
        self.assertEqual(events,['prefix','mm','encoder'])
        events.clear();engine.reset_prefix_cache=lambda:events.append('prefix') or False
        with self.assertRaises(RuntimeError):reset_variant_caches(engine)
        self.assertEqual(events,['prefix'])

    def test_whole_actual_sdk_parent_under_shallow_remount(self):
        sdk_python=ROOT.parents[2]/'.venv/bin/python'
        if not sdk_python.is_file():
            self.skipTest('installed SDK interpreter unavailable')
        folder=Path(__file__).parent
        code=f'''import os,sys
sys.path[:0]=[{str(ROOT/'src')!r},{str(folder)!r}]
os.environ['ADAM_VERIFY_PACKAGED_RUNTIME']='1'
from pathlib import Path
source=Path({str(folder/'modal_verify.py')!r}).read_text()
namespace={{'__name__':'remounted_parent','__file__':'/root/modal_verify.py'}}
exec(compile(source,'/root/modal_verify.py','exec'),namespace)
assert str(namespace['ROOT'])=='/opt/weight-space'
assert str(namespace['PROPOSAL_ROOT'])=='/opt/weight-space/adam_verify'
import verify_entry,verify_benchmark,verify_patch,verify_native
assert callable(verify_entry.worker_entry)
assert 'torch' not in sys.modules and 'vllm' not in sys.modules
'''
        subprocess.run([str(sdk_python),'-I','-c',code],check=True,capture_output=True,timeout=20)

    def rows(self,success=True):
        return [{'case_id':c.case_id,'success':success,'invalid_actions':0,
                 'completion_tokens':10,'token_usage_complete':True} for c in cases()]

    def test_reserved_seed_and_balanced_fixed_pair(self):
        self.assertEqual(VERIFICATION_SEED,2026110397)
        self.assertEqual(len(cases()),12)
        self.assertTrue(all(case.seed==VERIFICATION_SEED for case in cases()))
        for position in range(2):
            self.assertEqual([variant_order(i,.5)[position] for i in range(12)].count(.5),6)
        admission=fake_candidate_admission()
        self.assertEqual(validate_spec(admission),.5)
        admission['experiment_spec']['scales']=[1.0,.25]
        with self.assertRaises(ValueError):validate_spec(admission)
        for candidate in (1.0,True,.75,.25,0):
            with self.assertRaises(ValueError):scales(candidate)

    def test_complete_comparison_never_reselects_losing_candidate(self):
        variants={str(scale):self.rows() for scale in scales(.5)}
        variants['0.5'][0]['success']=False
        result=compare_verification(variants,.5)
        self.assertEqual(result['fixed_candidate_scale'],.5)
        self.assertEqual(result['paired_regressions'],[cases()[0].case_id])
        self.assertFalse(result['candidate_reselected'])
        self.assertFalse(result['retry_authorized'])
        self.assertFalse(result['candidate_perfect_on_this_subset'])
        variants['1.0'].pop()
        with self.assertRaises(ValueError):compare_verification(variants,.5)

    def test_candidate_requires_clean_exact_development_and_root_freeze(self):
        admission=fake_candidate_admission()
        self.assertEqual(validate_candidate(admission),.5)
        for field in ('malformed_outputs','missing_envelope_outputs','unknown_operations'):
            bad=copy.deepcopy(admission);bad['quality_review'][field]=1
            with self.assertRaises(ValueError):validate_candidate(bad)
        for mutation in ('selected_attention_scale','quality_review_sha256','verification_previously_performed'):
            bad=copy.deepcopy(admission);bad['candidate_selection'][mutation]='wrong'
            with self.assertRaises(ValueError):validate_candidate(bad)
        bad=copy.deepcopy(admission);bad['development_report']['run']='adam-dev-miniwob-20261103-v1'
        with self.assertRaises(ValueError):validate_candidate(bad)
        bad=copy.deepcopy(admission);bad['development_report']['child']['variants']['0.25'].pop()
        with self.assertRaises(ValueError):validate_candidate(bad)

    def test_concise_protocol_rejects_reasoning_or_duplicate_fields(self):
        visible=visible_observation(observation())
        for text in ('{"memory":"reasoning","action":[{"click":{"index":3}}]}',
                     '{"action":[],"action":[]}'):
            with self.assertRaises(ValueError):parse_batch(text,visible)

    def test_real_packaged_manifest_and_executing_source_mismatch(self):
        folder=Path(__file__).parent
        manifest=source_manifest(ROOT/'src/weight_space_lab',folder)
        self.assertIn('src/weight_space_lab/__init__.py',manifest)
        self.assertIn('proposals/adam_verify_v1/verify_host.py',manifest)
        executing_source_join(manifest,{'verify_patch.py':folder/'verify_patch.py'})
        manifest['proposals/adam_verify_v1/verify_patch.py']='0'*64
        with self.assertRaises(ValueError):executing_source_join(manifest,{'verify_patch.py':folder/'verify_patch.py'})


class HostTests(unittest.TestCase):
    def test_post_spawn_download_floor_failure_cancels_known_handle(self):
        events=[]
        class Call:
            object_id='fc-test'
            def cancel(self,**kwargs):events.append('cancel')
            def get(self,**kwargs):raise AssertionError('must refuse before download')
        class Function:
            object_id='fu-test'
            def hydrate(self):pass
            def spawn(self,*args):events.append('spawn');return Call()
        def floor(path,amount):
            if amount==(200+16)*1024**2:raise RuntimeError('disk floor')
        with self.assertRaisesRegex(RuntimeError,'disk floor'):
            dispatch_once({'run_id':IMPORT_RUN,'source_manifest':{},'root_allocation':{}},
                {'app_name':APP_NAME,'app_id':'ap-test','function_id':'fu-test','source_manifest':{}},
                '/unused',lambda *a,**kw:Function(),sdk_receipt={},source_sha256='a'*64,
                save=lambda *a:None,now=lambda:0,disk_guard=floor)
        self.assertEqual(events,['spawn','cancel'])

    def test_returned_handle_cancelled_if_call_receipt_write_fails(self):
        events=[]
        class Call:
            object_id='fc-test'
            def cancel(self,**kwargs):events.append('cancel')
        class Function:
            object_id='fu-test'
            def hydrate(self):events.append('hydrate')
            def spawn(self,admission):events.append('spawn');return Call()
        def save(path,value):
            if str(path).endswith('.call.json'):raise OSError('receipt write failed')
        admission={'run_id':IMPORT_RUN,'source_manifest':{},'root_allocation':{}}
        deployment={'app_name':APP_NAME,'app_id':'ap-test','function_id':'fu-test','source_manifest':{}}
        with self.assertRaises(OSError):
            dispatch_once(admission,deployment,'/unused',lambda *a,**kw:Function(),
                sdk_receipt={},source_sha256='a'*64,save=save,now=lambda:0,disk_guard=lambda *a:None)
        self.assertEqual(events,['hydrate','spawn','cancel'])

    def test_wrong_function_id_never_spawns(self):
        class Function:
            object_id='fu-wrong'
            def hydrate(self):pass
            def spawn(self,*a):raise AssertionError('not permitted')
        with self.assertRaises(ValueError):
            dispatch_once({'run_id':IMPORT_RUN,'source_manifest':{},'root_allocation':{}},
                {'app_name':APP_NAME,'app_id':'ap-test','function_id':'fu-test','source_manifest':{}},
                '/unused',lambda *a,**kw:Function(),sdk_receipt={},source_sha256='a'*64,
                save=lambda *a:None,now=lambda:0,disk_guard=lambda *a:None)

    def test_cancellation_alarm_never_rearms(self):
        arms=[];handler,cancel=alarm_handlers(arms.append);cancel()
        with self.assertRaises(TimeoutError):handler(None,None)
        self.assertEqual(arms,[20])


if __name__=='__main__':unittest.main()
