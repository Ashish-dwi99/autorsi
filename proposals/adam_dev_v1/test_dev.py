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
from dev_patch import MergedWeightEditor, validate_scale
from dev_contract import (SCALES,cases,variant_order,select_development,experiment_spec,
    validate_spec,DEVELOPMENT_SEED,VERIFICATION_SEED,source_manifest,executing_source_join)
from dev_contract import reset_variant_caches
from dev_native import parse_batch,visible_observation
from dev_host import dispatch_once,alarm_handlers,IMPORT_RUN,APP_NAME
from test_dev_native import observation


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
os.environ['ADAM_DEV_PACKAGED_RUNTIME']='1'
from pathlib import Path
source=Path({str(folder/'modal_dev.py')!r}).read_text()
namespace={{'__name__':'remounted_parent','__file__':'/root/modal_dev.py'}}
exec(compile(source,'/root/modal_dev.py','exec'),namespace)
assert str(namespace['ROOT'])=='/opt/weight-space'
assert str(namespace['PROPOSAL_ROOT'])=='/opt/weight-space/adam_dev'
import dev_entry,dev_benchmark,dev_patch,dev_native
assert callable(dev_entry.worker_entry)
assert 'torch' not in sys.modules and 'vllm' not in sys.modules
'''
        subprocess.run([str(sdk_python),'-I','-c',code],check=True,capture_output=True,timeout=20)

    def rows(self,success=True):
        return [{'case_id':c.case_id,'success':success,'invalid_actions':0,
                 'completion_tokens':10,'token_usage_complete':True} for c in cases()]

    def test_fresh_seed_verification_refused_and_balanced_order(self):
        self.assertNotIn(DEVELOPMENT_SEED,(2026110317,2026110353,VERIFICATION_SEED))
        self.assertEqual(len(cases()),12)
        for position in range(3):
            self.assertEqual([variant_order(i)[position] for i in range(12)].count(.25),4)
        spec=experiment_spec();validate_spec({'experiment_spec':spec})
        spec['development_seed']=VERIFICATION_SEED
        with self.assertRaises(ValueError):validate_spec({'experiment_spec':spec})

    def test_complete_selection_ties_and_no_partial_promotion(self):
        variants={str(scale):self.rows() for scale in SCALES}
        self.assertEqual(select_development(variants)['selected_scale'],1)
        variants['0.25'][0]['completion_tokens']=1
        result=select_development(variants)
        self.assertEqual(result['selected_scale'],.25)
        self.assertFalse(result['verification_performed'])
        self.assertEqual(result['official_prior_score_unchanged'],'21/24')
        variants['1.0'].pop()
        with self.assertRaises(ValueError):select_development(variants)

    def test_concise_protocol_rejects_reasoning_or_duplicate_fields(self):
        visible=visible_observation(observation())
        for text in ('{"memory":"reasoning","action":[{"click":{"index":3}}]}',
                     '{"action":[],"action":[]}'):
            with self.assertRaises(ValueError):parse_batch(text,visible)

    def test_real_packaged_manifest_and_executing_source_mismatch(self):
        folder=Path(__file__).parent
        manifest=source_manifest(ROOT/'src/weight_space_lab',folder)
        self.assertIn('src/weight_space_lab/__init__.py',manifest)
        self.assertIn('proposals/adam_dev_v1/dev_host.py',manifest)
        executing_source_join(manifest,{'dev_patch.py':folder/'dev_patch.py'})
        manifest['proposals/adam_dev_v1/dev_patch.py']='0'*64
        with self.assertRaises(ValueError):executing_source_join(manifest,{'dev_patch.py':folder/'dev_patch.py'})


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
