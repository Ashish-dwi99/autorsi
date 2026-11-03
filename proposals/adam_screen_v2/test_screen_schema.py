"""Pure protocol/geometry/schema checks, not browser or model results."""
import copy
import json
import unittest
from types import SimpleNamespace
from jsonschema import Draft202012Validator
from screen_protocol import parse_actions
import screen_schema as schema
from screen_cpu import coordinate_conformance


class NormalizedProtocolTests(unittest.TestCase):
    def test_exact_grid_both_viewports_and_no_clamping(self):
        raw='{"actions":[{"click":{"x":999,"y":999}},{"scroll":{"x":500,"y":0,"dy":-700}}]}'
        self.assertEqual(parse_actions(raw,1024,768),[
            {'click':{'x':1022,'y':767}},{'scroll':{'x':512,'y':0,'dy':-700}}])
        self.assertEqual(parse_actions(raw,390,844),[
            {'click':{'x':389,'y':843}},{'scroll':{'x':195,'y':0,'dy':-700}}])
        self.assertEqual(json.loads(raw)['actions'][0]['click']['x'],999)
        for value in (-1,1000,True,2.5,'50',None):
            with self.subTest(value=value),self.assertRaises(ValueError):
                parse_actions(json.dumps({'actions':[{'click':{'x':0,'y':0}},
                    {'click':{'x':value,'y':10}}]}),1024,768)
        self.assertEqual(len(coordinate_conformance()['cases']),6)

    def test_static_schema_validates_examples_and_strict_semantic_boundary(self):
        validator=Draft202012Validator(schema.action_schema())
        for row in schema.conformance_examples():
            if row['name'] in {'duplicate_key','truncated_prefix','trailing_prose'}:continue
            value=json.loads(row['text'])
            self.assertEqual(validator.is_valid(value),row['expected_complete'],row['name'])
            if row['name'].startswith('parser_rejects_'):
                with self.assertRaises(ValueError):parse_actions(row['text'],390,844)
        self.assertNotIn('goal',schema.schema_json())
        self.assertNotIn('viewport',schema.schema_json())
        self.assertNotIn('maxLength',schema.schema_json())
        self.assertNotIn('pattern',schema.schema_json())

    def test_sampling_has_one_explicit_backend_schema_for_both_token_caps(self):
        class Constraint:
            def __init__(self,**kw):self.kw=kw
        class Sampling:
            def __init__(self,**kw):self.kw=kw
        for cap in (64,128):
            result=schema.sampling_parameters(cap,sampling_class=Sampling,constraint_class=Constraint)
            self.assertEqual(result.kw['max_tokens'],cap)
            self.assertEqual(result.kw['structured_outputs'].kw,{'json':schema.schema_json()})
        with self.assertRaises(ValueError):schema.sampling_parameters(96)
        self.assertEqual(schema.engine_config()['backend'],'xgrammar')
        self.assertEqual(schema.engine_config()['reasoning_parser'],'')
        self.assertTrue(schema.engine_config()['enable_in_reasoning'])

    def test_cpu_receipt_requires_exact_string_and_token_conformance(self):
        from test_screen_runtime import fake_grammar
        receipt=fake_grammar(schema)
        schema.validate_cpu_grammar(receipt)
        bad=copy.deepcopy(receipt);bad['cases'][0]['token_complete']=False
        with self.assertRaises(ValueError):schema.validate_cpu_grammar(bad)
        bad=copy.deepcopy(receipt);bad['schema']['max_completion_tokens']=96
        with self.assertRaises(ValueError):schema.validate_cpu_grammar(bad)

    def test_actual_engine_config_requires_no_reasoning_bypass(self):
        config=SimpleNamespace(**schema.engine_config())
        engine=SimpleNamespace(llm_engine=SimpleNamespace(vllm_config=SimpleNamespace(
            structured_outputs_config=config)))
        self.assertEqual(schema.verify_engine_config(engine),schema.engine_config())
        config.reasoning_parser='unreviewed'
        with self.assertRaises(ValueError):schema.verify_engine_config(engine)

    def test_quality_review_cannot_revive_old_or_malformed_result(self):
        from test_screen_runtime import fake_attention_report,fake_quality,contract
        report=fake_attention_report();quality=fake_quality()
        selection={'schema':1,'development_report_sha256':'a'*64,
            'source_manifest_sha256':contract.DEV_SOURCE_HASH,'selected_attention_scale':1.0,
            'development_selection':report['child']['development_selection'],
            'verification_performed':False,'protocol_quality_reviewed':True,
            'quality_review_path':'evidence/TEST_ONLY.json','quality_review_sha256':'b'*64,
            'scope':'development-only attention choice for separate visual calibration'}
        def validate():return contract.validate_attention_selection(selection,report,'a'*64,
            quality_review=quality,quality_sha256='b'*64)
        self.assertEqual(validate(),1.0)
        for field in ('malformed_outputs','missing_envelope_outputs','unknown_operations'):
            quality[field]=1
            with self.assertRaises(ValueError):validate()
            quality[field]=0
        quality['semantic_invalid_actions_reviewed']=False
        with self.assertRaises(ValueError):validate()
        quality['semantic_invalid_actions_reviewed']=True
        report['run']='adam-dev-miniwob-20261103-v1'
        with self.assertRaises(ValueError):validate()


if __name__=='__main__':unittest.main()
