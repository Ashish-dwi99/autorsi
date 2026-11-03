"""Metadata-only adversarial checks; fake receipts are not model outcomes."""
import copy
import unittest

from summarize_adam_development import reconcile_receipts, cases, SCALES, variant_order


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
