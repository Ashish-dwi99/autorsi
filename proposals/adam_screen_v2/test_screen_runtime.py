"""Pure integration tests; simulated records are never actual model/browser results."""
import asyncio
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'proposals/adam_screen_v2'),str(ROOT/'proposals/adam_schema_v2')]
import screen_contract as contract
from screen_cpu import preflight,validate_preflight
from screen_worker import inspect_visual_values,require_idle
from screen_host import dispatch_once,APP_NAME,IMPORT_RUN
from dev_contract import cases as attention_cases,select_development,experiment_spec
import action_schema


def fake_grammar(module):
    return {'status':'passed','schema':module.schema_receipt(),'vocab_size':151936,
        'model_calls':0,'model_instantiations':0,'cuda_initialized':False,
        'runtime':{'versions':{'vllm':'0.18.0','xgrammar':'0.2.8'},'python_sources':{'TEST_ONLY':{}}},
        'tokenizer_class':'TEST_ONLY','tokenizer_path':'TEST_ONLY',
        'cases':[{'name':r['name'],'text_sha256':hashlib.sha256(r['text'].encode()).hexdigest(),
            'complete':r['expected_complete'],'token_complete':r['expected_complete']}
            for r in module.conformance_examples()]}


def fake_quality():
    return {'schema':1,'development_report_sha256':'a'*64,
        'source_manifest_sha256':contract.DEV_SOURCE_HASH,'complete_episodes':36,
        'malformed_outputs':0,'missing_envelope_outputs':0,'unknown_operations':0,
        'semantic_invalid_actions_reviewed':True,'carry_forward_to_screen_development':True,
        'promotion':'NONE'}


def fake_attention_report():
    source=json.loads((ROOT/'proposals/adam_schema_v2/SOURCE_RECORD.json').read_text())['source_manifest']
    variants={str(g):[{'case_id':c.case_id,'success':True,'invalid_actions':0,
        'completion_tokens':10,'token_usage_complete':True} for c in attention_cases()] for g in (1.,.5,.25)}
    return {'status':'completed','run':'adam-schema-miniwob-20261104-v2','child_reaped':True,
        'child_exitcode':0,'source_manifest':source,'child':{'status':'completed_development_only',
        'complete_coverage':True,'browser_close_returned':True,'variants':variants,
        'experiment_spec':experiment_spec(),'structural_decoding':action_schema.schema_receipt(),
        'verification_performed':False,'grammar_preflight':fake_grammar(action_schema),
        'development_selection':select_development(variants)}}


class ContractTests(unittest.TestCase):
    def test_attention_selection_requires_actual_complete_matching_receipt(self):
        report=fake_attention_report();digest='a'*64
        selection={'schema':1,'development_report_sha256':digest,
            'source_manifest_sha256':contract.DEV_SOURCE_HASH,'selected_attention_scale':1.0,
            'development_selection':report['child']['development_selection'],
            'verification_performed':False,'protocol_quality_reviewed':True,
            'quality_review_path':'evidence/TEST_ONLY.json','quality_review_sha256':'b'*64,'scope':'development-only attention choice for separate visual calibration'}
        self.assertEqual(contract.validate_attention_selection(selection,report,digest,quality_review=fake_quality(),quality_sha256='b'*64),1)
        wrong=copy.deepcopy(selection);wrong['selected_attention_scale']=.5
        with self.assertRaises(ValueError):contract.validate_attention_selection(wrong,report,digest,quality_review=fake_quality(),quality_sha256='b'*64)
        report['child']['variants']['0.25'].pop()
        with self.assertRaises(ValueError):contract.validate_attention_selection(selection,report,digest,quality_review=fake_quality(),quality_sha256='b'*64)

    def test_gain_selection_complete_only_and_baseline_tie(self):
        variants={str(g):[{'case_id':c.case_id,'success':True,'action_errors':0,
            'completion_tokens':5,'token_usage_complete':True,'abort_session':False,
            'context_close_returned':True,'video':{'fake_test_only':True}} for c in contract.cases('development')]
            for g in contract.GAINS}
        self.assertEqual(contract.select_gain(variants)['selected_gain'],1)
        variants['0.98'][0]['success']=False
        self.assertEqual(contract.select_gain(variants)['selected_gain'],1)
        variants['1.02'][0]['case_id']='verification-case'
        with self.assertRaises(ValueError):contract.select_gain(variants)

    def test_no_verification_in_executable_spec(self):
        self.assertEqual(len(contract.spec()['case_manifests']),8)
        self.assertTrue(all(c['split']=='development' for c in contract.spec()['case_manifests']))
        bad=contract.spec();bad['phase']='verification'
        with self.assertRaises(ValueError):contract.validate_spec({'experiment_spec':bad})

    def test_artifacts_paths_and_aggregate_bounds(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory);(path/'one.webm').write_bytes(b'x'*17)
            with patch.object(contract,'MAX_VIDEO_BYTES',16),self.assertRaises(ValueError):contract.artifact_inventory(path)
            (path/'one.webm').unlink();(path/'bad.safetensors').write_bytes(b'no')
            with self.assertRaises(ValueError):contract.artifact_inventory(path)

    def test_visual_inspection_reads_real_tiny_cpu_values_without_mutating(self):
        # Tiny synthetic CPU arrays test arithmetic/hash publication only; no model is loaded.
        import torch
        originals={'visual.merger.linear_fc2.weight':torch.tensor([[1.,-2.],[3.,0.]],dtype=torch.bfloat16),
                   'visual.merger.linear_fc2.bias':torch.tensor([0.,4.],dtype=torch.bfloat16)}
        before={k:v.clone() for k,v in originals.items()}
        report=inspect_visual_values(SimpleNamespace(originals=originals),torch)
        self.assertEqual(report['parameter_mutations'],0)
        for row in report['targets']:
            self.assertEqual(row['sha256'],hashlib.sha256(originals[row['name']].view(torch.uint8).numpy().tobytes()).hexdigest())
            self.assertTrue(torch.equal(before[row['name']],originals[row['name']]))
        self.assertAlmostEqual(report['targets'][0]['l2'],14**.5)
        originals['visual.merger.linear_fc2.bias'][0]=float('nan')
        with self.assertRaises(ValueError):inspect_visual_values(SimpleNamespace(originals=originals),torch)

    def test_idle_receipt_must_be_actual_false(self):
        for value in (True,None,0):
            with self.assertRaises(RuntimeError):require_idle(SimpleNamespace(llm_engine=SimpleNamespace(has_unfinished_requests=lambda:value)))
        require_idle(SimpleNamespace(llm_engine=SimpleNamespace(has_unfinished_requests=lambda:False)))

    def test_shallow_actual_sdk_module_import_and_source_join(self):
        folder=Path(__file__).parent;python=ROOT.parents[2]/'.venv/bin/python'
        code=f'''import os,sys
from pathlib import Path
sys.path[:0]=[{str(ROOT/'src')!r},{str(folder)!r},{str(ROOT/'proposals/adam_screen_v2')!r},{str(ROOT/'proposals/adam_schema_v2')!r}]
os.environ['ADAM_SCREEN_PACKAGED_RUNTIME']='1'
namespace={{'__name__':'screen_remount','__file__':'/root/modal_screen.py'}}
exec(compile(Path({str(folder/'modal_screen.py')!r}).read_text(),'/root/modal_screen.py','exec'),namespace)
assert str(namespace['ROOT'])=='/opt/weight-space'
assert str(namespace['PROPOSAL_ROOT'])=='/opt/weight-space/screen_runtime'
import screen_entry,screen_worker,screen_cpu
assert callable(screen_entry.worker_entry)
assert 'torch' not in sys.modules and 'vllm' not in sys.modules
'''
        subprocess.run([str(python),'-I','-c',code],check=True,timeout=20,capture_output=True)
        manifest=contract.source_manifest(ROOT)
        contract.executing_join(manifest,{'proposals/adam_screen_v2/screen_cpu.py':folder/'screen_cpu.py'})


class FakePage:
    def __init__(self,directory,viewport):
        self.path=Path(directory)/'fake.webm';self.path.write_bytes(b'FAKE_TEST_VIDEO_NOT_MEDIA')
        async def path():return str(self.path)
        self.video=SimpleNamespace(path=path);self.viewport=viewport
    async def set_content(self,*args,**kwargs):pass
    async def evaluate(self,*args):return None
    async def screenshot(self,*args,**kwargs):
        from PIL import Image
        import io
        stream=io.BytesIO();Image.new('RGB',(self.viewport['width'],self.viewport['height']),'white').save(stream,format='PNG')
        return stream.getvalue()


class FakeContext:
    def __init__(self,kwargs):self.kwargs=kwargs;self.closed=False
    async def route(self,*args):pass
    async def new_page(self):return FakePage(self.kwargs['record_video_dir'],self.kwargs['viewport'])
    async def close(self):self.closed=True


class FakeBrowser:
    version='TEST_ONLY'
    def __init__(self):self.contexts=[];self.closed=False
    async def new_context(self,**kwargs):
        context=FakeContext(kwargs);self.contexts.append(context);return context
    async def close(self):self.closed=True


class CPUPreflightTests(unittest.IsolatedAsyncioTestCase):
    async def test_pairs_development_only_and_never_invokes_policy(self):
        browser=FakeBrowser()
        async def launch(**kwargs):return browser
        class Manager:
            async def __aenter__(self):return SimpleNamespace(chromium=SimpleNamespace(executable_path='test-only',launch=launch))
            async def __aexit__(self,*args):pass
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(sys.modules,{'playwright.async_api':SimpleNamespace(async_playwright=lambda:Manager())}), \
                patch('screen_cpu.browser_runtime_manifest',return_value={'fake_runtime':True}):
                saved=[];receipt=await preflight(Path(directory)/'artifacts',lambda value:saved.append(copy.deepcopy(value)))
            self.assertEqual(len(browser.contexts),16)
            self.assertTrue(all(c.closed for c in browser.contexts));self.assertTrue(browser.closed)
            self.assertEqual(len(validate_preflight(receipt)),8)
            self.assertEqual(receipt['model_calls'],0);self.assertEqual(receipt['game_actions'],0)
            self.assertFalse(receipt['verification_rendered'])
            receipt['cases'][0]['renders'][1]['screenshot_sha256']='0'*64
            with self.assertRaises(ValueError):validate_preflight(receipt)


class HostTests(unittest.TestCase):
    def test_partial_artifact_download_is_preserved_without_retry(self):
        import screen_recover
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);runs=root/'runs';runs.mkdir()
            rows=[{'relative_path':'case/video.webm','bytes':3,'sha256':hashlib.sha256(b'abc').hexdigest()},
                  {'relative_path':'case/frame.png','bytes':3,'sha256':hashlib.sha256(b'xyz').hexdigest()}]
            report={'run':IMPORT_RUN,'function_call_id':'fc-test','source_manifest':{},
                'status':'completed','container_termination_required':False,'child':{'artifacts':{'files':rows}}}
            (runs/(IMPORT_RUN+'.remote-report.json')).write_text(json.dumps(report))
            (runs/(IMPORT_RUN+'.call.json')).write_text(json.dumps({'function_call_id':'fc-test'}))
            calls=[]
            class Volume:
                def read_file(self,path):calls.append(path);yield b'ab'
            with patch.object(screen_recover,'source_manifest',return_value={}), \
                patch.object(screen_recover.shutil,'disk_usage',return_value=SimpleNamespace(free=1024**3)), \
                self.assertRaises(ValueError):
                screen_recover.recover_once(IMPORT_RUN,Volume(),root=root)
            self.assertEqual(len(calls),1)
            self.assertEqual((runs/(IMPORT_RUN+'.artifacts')/'case/video.webm').read_bytes(),b'ab')
            failure=json.loads((runs/(IMPORT_RUN+'.artifacts')/'recovery-failed.json').read_text())
            self.assertFalse(failure['retry_authorized'])

    def test_known_handle_is_cancelled_on_receipt_failure(self):
        events=[]
        class Call:
            object_id='fc-test'
            def cancel(self,**kwargs):events.append('cancel')
        class Function:
            object_id='fu-test'
            def hydrate(self):pass
            def spawn(self,*args):events.append('spawn');return Call()
        def save(path,value):
            if str(path).endswith('.call.json'):raise OSError('disk error')
        with self.assertRaises(OSError):
            dispatch_once({'run_id':IMPORT_RUN,'source_manifest':{},'root_allocation':{}},
                {'app_name':APP_NAME,'app_id':'ap-test','function_id':'fu-test','source_manifest':{}},'/unused',
                lambda *a,**kw:Function(),sdk_receipt={},source_sha256='a'*64,save=save,
                now=lambda:0,disk_guard=lambda *a:None)
        self.assertEqual(events,['spawn','cancel'])


if __name__=='__main__':unittest.main()
