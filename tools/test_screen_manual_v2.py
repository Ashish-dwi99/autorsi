"""Source/receipt contract checks only; no manual actor, browser or media."""
import copy
import hashlib
import json
import unittest
from unittest.mock import patch
import screen_manual_v2 as cli


def fake():
    source={'test':'metadata-only, not an actual runtime'}
    digest=hashlib.sha256(json.dumps(source,sort_keys=True).encode()).hexdigest()
    row={'case_id':'screen-development-01','context_close_returned':True,'policy_pending':False,
         'abort_session':False,'video':{'not_media':'synthetic receipt only'},
         'actor_identity_declared':{'attention_scale':.25,'visual_gain':1.02}}
    variants={str(gain):[copy.deepcopy(row) for _ in range(8)] for gain in (.98,1.,1.02)}
    for rows in variants.values():
        for index,item in enumerate(rows):item['case_id']=f'screen-development-{index+1:02d}'
    row=copy.deepcopy(variants['1.02'][0])
    report={'source_manifest':source,'run':'adam-screen-visual-dev-20261104-v2','status':'completed',
        'child_reaped':True,'child_exitcode':0,'container_termination_required':False,
        'child':{'source_manifest':source,'status':'completed_screen_development_only',
            'complete_coverage':True,'browser_close_returned':True,'attention_scale':.25,
            'visual_selection':{'selected_gain':1.02},'variants':variants}}
    freeze={'schema':1,'decision':'frozen_for_demonstration_only','verification_claim':False,
        'development_report_sha256':'a'*64,'source_manifest_sha256':digest,
        'selected_attention_scale':.25,'selected_visual_gain':1.02}
    identity={'schema':1,'model_agent':'Astra','attribution':'external_workspace_runtime',
        'runtime_identity':{'TEST':'not an actual identity claim'}}
    return digest,freeze,report,row,identity


class ManualSourceTests(unittest.TestCase):
    def test_exact_v2_sources_and_same_limits_without_browser(self):
        _,protocol,_,_=cli.source_modules()
        self.assertIn('normalized integers from 0 through 999',protocol.SYSTEM_PROMPT)
        limits=protocol.Limits()
        self.assertEqual((limits.call_timeout_s,limits.case_timeout_s,limits.max_actions),(30,90,12))
        self.assertEqual(protocol.parse_actions('{"actions":[{"click":{"x":999,"y":999}}]}',1024,768),
                         [{'click':{'x':1022,'y':767}}])
        with self.assertRaises(ValueError):protocol.parse_actions('{"actions":[{"click":{"x":1024,"y":768}}]}',1024,768)

    def test_changed_source_refuses(self):
        with patch.dict(cli.SOURCE_PINS,{'screen_protocol.py':'0'*64}):
            with self.assertRaisesRegex(ValueError,'source changed'):cli.source_modules()

    def test_root_freeze_joins_actual_selected_case_not_a_new_winner(self):
        digest,freeze,report,row,identity=fake()
        with patch.object(cli,'SOURCE_MANIFEST_SHA256',digest):
            cli.validate_freeze(freeze,report,'a'*64,row,identity)
            for key,value in [('selected_visual_gain',.98),('selected_attention_scale',.5),
                               ('development_report_sha256','b'*64),('verification_claim',True)]:
                changed=copy.deepcopy(freeze);changed[key]=value
                with self.subTest(key=key),self.assertRaises(ValueError):
                    cli.validate_freeze(changed,report,'a'*64,row,identity)

    def test_no_missing_rows_different_case_or_unknown_operation(self):
        digest,freeze,report,row,identity=fake()
        with patch.object(cli,'SOURCE_MANIFEST_SHA256',digest):
            for mutation in ('partial','missing','different_case','unknown','runtime'):
                changed=copy.deepcopy(report);case=copy.deepcopy(row);actor=copy.deepcopy(identity)
                if mutation=='partial':changed['child']['partial_case']={}
                elif mutation=='missing':changed['child']['variants']['0.98'].pop()
                elif mutation=='different_case':case['case_id']='screen-verification-01'
                elif mutation=='unknown':case['policy_pending']=True
                else:actor['model_agent']='unverified replacement'
                with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                    cli.validate_freeze(freeze,changed,'a'*64,case,actor)


if __name__=='__main__':unittest.main()
