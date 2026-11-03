"""Metadata-only adversarial checks; fake receipts are not model outcomes."""
import copy
import unittest

from summarize_adam_schema_v2 import (reconcile_receipts, cases, SCALES, variant_order,
    audit_raw_outputs, reconcile_schema_receipts, frozen_source, schema_receipt,
    engine_config, experiment_spec, expected_timer_ms, IMPORT_RUN)
from unittest.mock import patch


def fake_report():
    child = {'planned_cases': 12, 'planned_episodes': 36, 'complete_coverage': True,
             'environments_constructed': 36, 'environment_close_returns': 36,
             'generation_records': [], 'warmups': [], 'variants': {str(s): [] for s in SCALES},
             'artifacts': [], 'elapsed_s': 100.0}
    for scale in SCALES:
        child['generation_records'].append({'call': len(child['generation_records']) + 1,
            'scope': 'unscored_neutral_warmup', 'case_id': None, 'scale': scale,
            'max_tokens': 64, 'prompt_tokens': 3, 'output_tokens': 2, 'elapsed_s': .1})
        child['warmups'].append({'scale': scale, 'prompt_tokens': 3, 'completion_tokens': 2})
    for index, case in enumerate(cases()):
        for scale in variant_order(index):
            artifact = {'relative_path': f'case-{index:02d}-scale-{scale}-step-1.png',
                        'bytes': 2, 'sha256': 'a' * 64, 'case_id': case.case_id,
                        'scale': scale, 'dispatch': 1, 'elapsed_s': len(child['artifacts']) + 1,
                        'width': 160, 'height': 210, 'kind': 'actual_pre_action_observation'}
            child['artifacts'].append(artifact)
            child['variants'][str(scale)].append({'case_id': case.case_id, 'env_id': case.env_id,
                'seed': case.seed, 'elapsed_s': .5, 'model_calls': 1, 'policy_dispatches': 1,
                'environment_steps': 1, 'invalid_actions': 0, 'success': True, 'reward': 1.0,
                'terminated': True, 'truncated': False, 'stop_reason': 'native_success',
                'prompt_tokens': 3, 'completion_tokens': 2, 'token_usage_complete': True,
                'events': [{'dispatch': 1, 'kind': 'inference', 'native_action': {'click': {'index': 1}},
                            'observation_artifact': copy.deepcopy(artifact)}]})
            child['generation_records'].append({'call': len(child['generation_records']) + 1,
                'scope': 'scored_miniwob_episode', 'case_id': case.case_id, 'scale': scale,
                'max_tokens': 96, 'prompt_tokens': 3, 'output_tokens': 2, 'elapsed_s': .1})
    child['model_calls'] = len(child['generation_records'])
    return {'child': child, 'artifact_preservation': {'files': 36, 'bytes': 72,
            'kind': 'actual pre-action screenshots with elapsed timestamps; not continuous video'}}


class SavedReceiptTests(unittest.TestCase):
    def test_complete_synthetic_metadata_join(self):
        result = reconcile_receipts(fake_report())
        self.assertEqual(result['episodes'], 36)
        self.assertEqual(result['scored_model_requests'], 36)
        self.assertEqual(result['neutral_warmup_requests'], 3)
        self.assertEqual(result['observation_receipts'], 36)

    def test_wrong_case_is_rejected_even_when_variant_totals_match(self):
        report = fake_report()
        requests = report['child']['generation_records']
        first = next(r for r in requests if r['scale'] == 1.0 and r['scope'] == 'scored_miniwob_episode')
        second = next(r for r in requests if r['scale'] == 1.0 and r['scope'] == 'scored_miniwob_episode' and r['case_id'] != first['case_id'])
        first['case_id'], second['case_id'] = second['case_id'], first['case_id']
        with self.assertRaisesRegex(ValueError, 'per-case generation'):
            reconcile_receipts(report)

    def test_unknown_warmup_scope_rejected(self):
        report = fake_report()
        report['child']['generation_records'][0]['scope'] = 'unaccounted'
        with self.assertRaisesRegex(ValueError, 'warmup'):
            reconcile_receipts(report)

    def test_artifact_preservation_and_event_mismatch_rejected(self):
        for mutation in ('copy', 'event'):
            report = fake_report()
            if mutation == 'copy':
                report['artifact_preservation']['bytes'] -= 1
            else:
                report['child']['variants']['1.0'][0]['events'][0]['observation_artifact']['sha256'] = 'b' * 64
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                reconcile_receipts(report)


def raw_fake_report():
    report=fake_report();child=report['child']
    for request in child['generation_records']:
        request.update(finish_reason='stop',stop_reason=None,schema_sha256=schema_receipt()['sha256'])
    for warmup in child['warmups']:
        warmup['native_response']='{"action":[{"done":{"text":"ready","success":false}}]}'
    for rows in child['variants'].values():
        for row in rows:
            row['events'][0].update(native_response='{"action":[{"click":{"index":1}}]}',native_response_truncated=False)
    return report


class SchemaSavedReceiptTests(unittest.TestCase):
    def test_actual_frozen_source_identity_and_full_raw_coverage(self):
        self.assertEqual(frozen_source()['source_manifest_sha256'],'b44dbbf430face18a694a4a8a3f4f6d8e10f1abf841806a2b9fba5ef401fde85')
        report=raw_fake_report();reconcile_receipts(report)
        audit=audit_raw_outputs(report['child'])
        self.assertEqual(len(audit['episodes']),36)
        self.assertEqual(len(audit['neutral_warmups']),3)
        self.assertEqual(audit['quality_metrics']['malformed_outputs'],0)
        self.assertEqual(audit['quality_metrics']['semantic_invalid_actions'],0)
        self.assertFalse(audit['rescored'])
        self.assertFalse(audit['response_repair'])

    def test_malformed_missing_envelope_and_semantic_errors_are_distinct(self):
        cases_to_check=[('{"click":{"index":1}}','concise protocol requires exactly the action key',1,1,0),
            ('{"action":[','one complete native JSON object required',1,0,0),
            ('{"action":[{"click":{"index":999}}]}','index must identify a current positive visible real element',0,0,1)]
        for raw,error,malformed,missing,semantic in cases_to_check:
            report=raw_fake_report();row=report['child']['variants']['1.0'][0]
            row.update(invalid_actions=1,success=False,reward=0.0,terminated=False,environment_steps=0,stop_reason='invalid_native_batch')
            event=row['events'][0];event.pop('native_action');event.update(native_response=raw,error=error)
            reconcile_receipts(report)
            audit=audit_raw_outputs(report['child'])
            with self.subTest(raw=raw):
                self.assertEqual(audit['quality_metrics']['malformed_outputs'],malformed)
                self.assertEqual(audit['quality_metrics']['missing_envelope_outputs'],missing)
                self.assertEqual(audit['quality_metrics']['semantic_invalid_actions'],semantic)
                self.assertEqual(sum(audit['invalid_categories'].values()),1)

    def test_partial_or_missing_raw_data_never_becomes_zero_error_evidence(self):
        for mutation in ('partial','raw','flag'):
            report=raw_fake_report();child=report['child']
            if mutation=='partial':child['partial_episode']={}
            elif mutation=='raw':child['variants']['1.0'][0]['events'][0].pop('native_response')
            else:child['variants']['1.0'][0]['events'][0].pop('native_response_truncated')
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):audit_raw_outputs(child)
        report=raw_fake_report();report['child']['warmups'][0]['native_response']='{"done":{}}'
        self.assertEqual(audit_raw_outputs(report['child'])['quality_metrics']['malformed_outputs'],1)

    def test_fresh_CPU_grammar_configuration_and_timer_joins_are_required(self):
        report=raw_fake_report();child=report['child'];manifest={'fake':'not-runtime-evidence'}
        child.update(grammar_preflight={'fake':'not-runtime-evidence'},native_preflight={'fake':'not-reset-evidence'},
            runtime_manifest={'fake':'not-browser-evidence'},structural_decoding=schema_receipt(),
            actual_structured_output_config=engine_config())
        for index,case in enumerate(cases()):
            for scale in SCALES:
                row=child['variants'][str(scale)][index]
                row['native_timer_ms']=expected_timer_ms(case)
                row['reset']={'case_id':case.case_id,'env_id':case.env_id,'seed':case.seed,'status':'accepted',
                    'observed_timer_type':'int','observed_timer_encoding':'json_scalar',
                    'observed_timer_value':expected_timer_ms(case),'expected_native_timer_ms':expected_timer_ms(case)}
        child['grammar_preflight']['cases']=[]
        cpu_child={'status':'imports_passed','run':IMPORT_RUN,'source_manifest':manifest,
            'experiment_spec':experiment_spec(),'model_calls':0,'tensor_payloads_loaded':0,'cuda_initialized':False,
            'grammar_preflight':child['grammar_preflight'],'miniwob_preflight':child['native_preflight'],
            'runtime_manifest':child['runtime_manifest']}
        cpu={'status':'completed','run':IMPORT_RUN,'child_reaped':True,'child_exitcode':0,
            'container_termination_required':False,'source_manifest':manifest,'child':cpu_child}
        with patch('summarize_adam_schema_v2.validate_cpu_grammar'):
            self.assertTrue(reconcile_schema_receipts(report,cpu,{'source_manifest':manifest})['same_schema_all_requests'])
            child['generation_records'][0]['schema_sha256']='0'*64
            with self.assertRaisesRegex(ValueError,'different schema'):
                reconcile_schema_receipts(report,cpu,{'source_manifest':manifest})
            child['generation_records'][0]['schema_sha256']=schema_receipt()['sha256']
            child['variants']['1.0'][0]['reset']['observed_timer_value']=999999
            with self.assertRaisesRegex(ValueError,'timer'):
                reconcile_schema_receipts(report,cpu,{'source_manifest':manifest})

    def test_nonfinite_or_negative_episode_times_rejected(self):
        for value in (float('nan'), float('inf'), -1, True):
            report = fake_report()
            report['child']['variants']['1.0'][0]['elapsed_s'] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'elapsed'):
                reconcile_receipts(report)

    def test_reward_and_dispatch_claims_cannot_replace_receipts(self):
        for mutation in ('reward', 'action'):
            report = fake_report()
            row = report['child']['variants']['1.0'][0]
            if mutation == 'reward':
                row['reward'] = -1.0
            else:
                row['events'][0].pop('native_action')
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                reconcile_receipts(report)


if __name__ == '__main__':
    unittest.main()
