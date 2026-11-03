"""Metadata and mocked-tool tests only. No synthetic or actual video rendered."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from compose_screen_comparison import validate_record, recipe_caption, media_path, probe_video


def receipt():
    final={'saved':'example metadata, not a model outcome'}
    return {'context_close_returned':True,'policy_pending':False,'abort_session':False,
        'policy_calls':1,'actions_dispatched':1,'success':True,'frames':[{'metadata_only':True}],
        'split':'development','actor_label':'Metadata test actor',
        'video':{'actual_browser_capture':True,'record_video_size':{'width':1024,'height':768}},
        'viewport':{'width':1024,'height':768},'stop_reason':'independent_final_state_match',
        'final_receipt':final,'oracle_sha256':hashlib.sha256(json.dumps(final,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        'actor_identity_declared':{'attention_scale':.25,'visual_gain':.98}}


class CompositorTests(unittest.TestCase):
    def test_outcome_boolean_and_final_receipt_must_match(self):
        validate_record(receipt())
        for value in ('false',1,None):
            row=receipt();row['success']=value
            with self.subTest(value=value),self.assertRaises(ValueError):validate_record(row)
        row=receipt();row['final_receipt']={'different':True}
        with self.assertRaisesRegex(ValueError,'final-state'):validate_record(row)

    def test_pending_aborted_and_nonfinal_receipts_are_refused(self):
        for key,value in (('policy_pending',True),('abort_session',True),('context_close_returned',False),
                          ('policy_calls',False),('artifact_error','incomplete video')):
            row=receipt();row[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):validate_record(row)
        row=receipt();row.update(success=False,stop_reason='case_native_disposition_unknown')
        with self.assertRaises(ValueError):validate_record(row)

    def test_failed_task_is_permitted_as_failed_without_claiming_completion(self):
        row=receipt();row.update(success=False,stop_reason='invalid_response_or_ui_error',actions_dispatched=0)
        row.pop('final_receipt');validate_record(row)
        row['actor_label']='hidden\nsecond line'
        with self.assertRaises(ValueError):validate_record(row)

    def test_recipe_identity_is_visible_and_never_inferred(self):
        self.assertEqual(recipe_caption(receipt()),'attention=0.25 | vision gain=0.98')
        row=receipt();row['actor_identity_declared']['visual_gain']=1.02
        self.assertNotEqual(recipe_caption(row),recipe_caption(receipt()))
        row['actor_identity_declared']={}
        self.assertIn('external review required',recipe_caption(row))
        row['actor_identity_declared']={'visual_gain':float('nan')}
        with self.assertRaises(ValueError):recipe_caption(row)

    def test_media_paths_do_not_escape_or_allow_symlinks(self):
        for relative in ('../outside.webm','/absolute.webm',''):
            with self.subTest(path=relative),self.assertRaises(ValueError):media_path(Path('/receipt'),relative)
        with patch.object(Path,'is_symlink',return_value=True):
            with self.assertRaisesRegex(ValueError,'symlink'):media_path(Path('/receipt'),'video/capture.webm')

    def test_probe_geometry_duration_errors_and_rotation(self):
        base={'streams':[{'width':1024,'height':768,'nb_read_frames':'100','sample_aspect_ratio':'1:1'}],
              'format':{'duration':'4.0'}}
        def returned(data,stderr=b''):
            return SimpleNamespace(stdout=json.dumps(data).encode(),stderr=stderr)
        with patch('compose_screen_comparison.subprocess.run',return_value=returned(base)):
            probe,duration=probe_video(Path('/no-file-opened.webm'),{'width':1024,'height':768})
            self.assertEqual(duration,4)
            with self.assertRaisesRegex(ValueError,'duration'):
                probe_video(Path('/no-file-opened.webm'),{'width':1024,'height':768},expected_duration=5)
        for mutation in ('rotation','sar','nan','second_stream','geometry'):
            value=copy.deepcopy(base)
            if mutation=='rotation':value['streams'][0]['side_data_list']=[{'rotation':90}]
            elif mutation=='sar':value['streams'][0]['sample_aspect_ratio']='2:1'
            elif mutation=='nan':value['format']['duration']='nan'
            elif mutation=='second_stream':value['streams'].append(copy.deepcopy(value['streams'][0]))
            else:value['streams'][0]['width']=1
            with patch('compose_screen_comparison.subprocess.run',return_value=returned(value)):
                with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                    probe_video(Path('/no-file-opened.webm'),{'width':1024,'height':768})
        with patch('compose_screen_comparison.subprocess.run',return_value=returned(base,b'decoding error')):
            with self.assertRaisesRegex(ValueError,'errors'):probe_video(Path('/no-file-opened.webm'),{'width':1024,'height':768})


if __name__=='__main__':unittest.main()
