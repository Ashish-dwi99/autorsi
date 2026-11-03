"""Offline structural/schema/integration tests. No model, browser or provider."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import jsonschema

from action_schema import (action_schema, schema_json, schema_receipt, sampling_parameters,
    conformance_examples, engine_config, cpu_backend_preflight, validate_cpu_grammar)
from dev_native import parse_batch, visible_observation
from test_dev_native import observation


class StructuralTests(unittest.TestCase):
    def test_frozen_structural_examples_and_prefix_failures(self):
        validator = jsonschema.Draft202012Validator(action_schema())
        validator.check_schema(action_schema())
        for case in conformance_examples():
            with self.subTest(case=case['name']):
                if case['name'] == 'duplicate_key':
                    # Ordinary json.loads discards duplicates; original parser
                    # and the actual future xgrammar gate independently refuse.
                    with self.assertRaises(ValueError):
                        parse_batch(case['text'], visible_observation(observation()))
                    continue
                try:
                    value = json.loads(case['text'])
                    valid = validator.is_valid(value)
                except json.JSONDecodeError:
                    valid = False
                self.assertIs(valid, case['expected_complete'])

    def test_parser_constraints_are_not_claimed_as_grammar_guarantees(self):
        visible = visible_observation(observation())
        for case in conformance_examples():
            if case['name'].startswith('parser_rejects_'):
                jsonschema.validate(json.loads(case['text']), action_schema())
                with self.subTest(case=case['name']), self.assertRaises(ValueError):
                    parse_batch(case['text'], visible)
        valid = next(c for c in conformance_examples() if c['name'] == 'escaped_ascii')
        self.assertEqual(parse_batch(valid['text'], visible)[0].text, 'quote" slash\\')
        with self.assertRaises(ValueError):
            parse_batch(next(c['text'] for c in conformance_examples() if c['name']=='eight_actions'), visible, remaining=7)

    def test_same_static_sampling_schema_and_no_observation_argument(self):
        class Params:
            def __init__(self, **kwargs): self.__dict__.update(kwargs)
        values = [sampling_parameters(cap, sampling_class=Params, constraint_class=Params)
                  for cap in (64, 96, 96, 96)]
        self.assertEqual({p.structured_outputs.json for p in values}, {schema_json()})
        self.assertEqual([p.max_tokens for p in values], [64,96,96,96])
        self.assertTrue(all(p.temperature==0 and p.top_p==1 and p.n==1 for p in values))
        for cap in (True, 128, 0, 65):
            with self.assertRaises(ValueError):
                sampling_parameters(cap, sampling_class=Params, constraint_class=Params)
        initial=schema_json(); changed=action_schema(); changed['properties']['action']['maxItems']=1
        self.assertEqual(schema_json(),initial)
        self.assertEqual(engine_config()['backend'],'xgrammar')
        self.assertEqual(engine_config()['reasoning_parser'],'')
        self.assertTrue(engine_config()['enable_in_reasoning'])

    def test_prior_parser_patch_prompt_and_build_bytes_unchanged(self):
        here=Path(__file__).parent; old=here.parent/'adam_dev_v1'
        for name in ('dev_native.py','dev_patch.py','build_miniwob_runtime.py'):
            self.assertEqual((here/name).read_bytes(),(old/name).read_bytes(),name)
        from dev_contract import DEVELOPMENT_SEED, VERIFICATION_SEED
        self.assertEqual(DEVELOPMENT_SEED,2026110413)
        self.assertNotIn(DEVELOPMENT_SEED,(2026110317,2026110353,2026110389,VERIFICATION_SEED))
        self.assertEqual(VERIFICATION_SEED,2026110397)

    def test_gpu_and_cpu_use_bound_structural_interfaces(self):
        here=Path(__file__).parent
        gpu=ast.parse((here/'dev_benchmark.py').read_text())
        calls=[n for n in ast.walk(gpu) if isinstance(n,ast.Call)]
        llm=[n for n in calls if isinstance(n.func,ast.Name) and n.func.id=='LLM']
        self.assertEqual(len(llm),1)
        self.assertIn('structured_outputs_config',{k.arg for k in llm[0].keywords})
        self.assertEqual(sum(isinstance(n.func,ast.Name) and n.func.id=='sampling_parameters' for n in calls),1)
        self.assertFalse(any(isinstance(n.func,ast.Name) and n.func.id=='SamplingParams' for n in calls))
        parent=(here/'modal_dev.py').read_text()
        self.assertIn('--no-deps xgrammar==0.2.8',parent)
        self.assertIn('validate_cpu_grammar(admission.get("grammar_preflight", {}))',parent)

    def test_cpu_counterexample_is_persisted_before_failure(self):
        # This deliberately incorrect fake backend tests only journaling. It is
        # never evidence that real xgrammar compiled or accepted the schema.
        class Tokenizer:
            def __len__(self): return 10
            def encode(self,text,add_special_tokens=False): return list(text.encode())
        class Matcher:
            def __init__(self,*a,**kw): pass
            def accept_string(self,text): return True
            def accept_token(self,token): return True
            def is_completed(self): return True
            def is_terminated(self): return True
        class Compiler:
            def __init__(self,*a,**kw): pass
            def compile_json_schema(self,*a,**kw): return object()
        fake_xgr=SimpleNamespace(GrammarMatcher=Matcher,GrammarCompiler=Compiler,
            TokenizerInfo=SimpleNamespace(from_huggingface=lambda *a,**kw:object()))
        modules={'torch':SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda:False)),
            'xgrammar':fake_xgr,
            'transformers':SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a,**kw:Tokenizer())),
            'vllm.config':SimpleNamespace(StructuredOutputsConfig=lambda **kw:SimpleNamespace(**kw)),
            'vllm.v1.structured_output.backend_xgrammar':SimpleNamespace(validate_xgrammar_grammar=lambda _:None)}
        saved=[]
        with patch.dict(sys.modules,modules), patch('importlib.metadata.version',side_effect=lambda n:{'vllm':'0.18.0','xgrammar':'0.2.8'}[n]), \
             patch('action_schema.sampling_parameters',return_value=object()), \
             patch('action_schema.runtime_backend_identity',return_value={'versions':{'vllm':'0.18.0','xgrammar':'0.2.8'},'python_sources':{'fake':'not-evidence'}}):
            with self.assertRaisesRegex(ValueError,'missing_envelope'):
                cpu_backend_preflight('/fake/metadata-only-tokenizer',151936,
                    save_partial=lambda row:saved.append(copy.deepcopy(row)))
        final=saved[-1]
        self.assertEqual(final['status'],'failed')
        self.assertEqual(final['failed_case'],'missing_envelope')
        self.assertTrue(final['cases'][-1]['complete'])
        self.assertTrue(final['cases'][-1]['token_complete'])
        self.assertGreater(len(final['cases']),1)
        self.assertEqual(final['model_calls'],0)
        with self.assertRaises(ValueError): validate_cpu_grammar(final)


if __name__=='__main__': unittest.main()
