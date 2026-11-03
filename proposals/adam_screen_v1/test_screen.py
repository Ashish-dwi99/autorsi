"""Fresh-fixture validation only; scripted references are not model results.

The optional local browser test executes eight DEVELOPMENT UI references and
only renders the eight verification variants. It makes no policy/model calls.
Reference DOM lookup is confined to this test and never passed to a policy.
"""
import asyncio
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from screen_cases import cases, check_receipt, demo_cases
from screen_harness import ScreenSession, policy_messages, execute_action, canonical, RECEIPT_JS
from screen_manual import ManualPolicy, next_observation, submit, write_json
from screen_protocol import Limits, Reply, SessionAborted, parse_actions

PNG = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+j9CkAAAAASUVORK5CYII=')


class ProtocolTests(unittest.TestCase):
    def test_valid_batch_and_full_batch_refusal(self):
        text = '{"actions":[{"click":{"x":10,"y":12}},{"key":{"name":"SELECT_ALL"}},{"type":{"text":"Sample"}},{"key":{"name":"ENTER"}}]}'
        self.assertEqual(len(parse_actions(text, 390, 844)), 4)
        invalid = [
            '{"actions":[{"click":{"x":10,"y":12}},{"click":{"x":390,"y":12}}]}',
            '{"actions":[{"click":{"x":true,"y":12}}]}',
            '{"actions":[{"click":{"x":1,"x":2,"y":12}}]}',
            '{"actions":[{"key":{"name":"Control+L"}}]}',
            '{"actions":[{"key":{"name":[]}}]}',
            '{"actions":[{"type":{"text":"hello\\nworld"}}]}',
            '{"actions":[{"done":{}},{"click":{"x":1,"y":1}}]}',
            '{"actions":[{"click":{"x":1,"y":NaN}}]}',
            '{"actions":[{"click":{"x":1,"y":1,"selector":"#answer"}}]}',
        ]
        for bad in invalid:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse_actions(bad, 390, 844)
        with self.assertRaises(ValueError):
            parse_actions(text, 390, 844, remaining=3)

    def test_frozen_cases_and_independent_oracle(self):
        dev, verification = cases('development'), cases('verification')
        self.assertEqual(len(dev), 8)
        self.assertEqual(len(verification), 8)
        # Generic contact/counter apps can share HTML while their goal and
        # viewport differ. The complete scenario identity must be distinct.
        self.assertEqual(len({hashlib.sha256(canonical(c.manifest())).hexdigest() for c in dev + verification}), 16)
        self.assertEqual(len({c.seed for c in dev + verification}), 16)
        self.assertEqual(sum(c.layout == 'desktop-like' for c in dev), 4)
        for a, b in zip(dev, verification):
            self.assertNotEqual(a.layout, b.layout)
            self.assertNotEqual(a.goal, b.goal)
            self.assertFalse(check_receipt(a, None))
            self.assertFalse(check_receipt(a, {'wrong': True}))
            self.assertTrue(check_receipt(a, a.expected))
        self.assertTrue(all(c.split == 'development' for c in demo_cases()))
        self.assertNotIn('screen-verification', ''.join(c.case_id for c in demo_cases()))

    def test_policy_observation_contains_no_dom_or_expected_state(self):
        case = cases('development')[0]
        messages = policy_messages(case.goal, PNG, case.viewport, 1, 12, [])
        payload = json.loads(messages[-1]['content'][0]['text'])
        self.assertEqual(set(payload), {'goal', 'viewport', 'sequence', 'remaining_actions', 'coordinates'})
        self.assertEqual(base64.b64decode(messages[-1]['content'][1]['image_url']['url'].split(',')[1]), PNG)
        self.assertNotIn('expected', json.dumps(payload))
        self.assertNotIn('elements', json.dumps(payload))
        self.assertNotIn('data-id', json.dumps(messages))


class ManualTests(unittest.IsolatedAsyncioTestCase):
    async def test_late_returned_policy_is_retained_but_never_dispatched(self):
        clock=[0.0]
        async def policy(*args):
            clock[0]=91.0
            return Reply('{"actions":[{"click":{"x":1,"y":1}}]}',1,1)
        session=ScreenSession(policy,actor_label='test double')
        result={'policy_calls':0,'policy_pending':False,'prompt_tokens':0,'completion_tokens':0,'token_usage_complete':True}
        with self.assertRaises(SessionAborted):
            await session._infer([],result,lambda:90-clock[0])
        self.assertEqual(result['policy_calls'],1)
        self.assertEqual(result['stop_reason'],'policy_return_observed_after_case_deadline')
        self.assertFalse(result['policy_pending'])
        self.assertEqual(result['completion_tokens'],1)
        self.assertIn('click',result['late_policy_reply']['text'])
        self.assertIsNone(session.pending)
        self.assertTrue(session.aborted)

    async def test_expired_original_clock_never_creates_policy_task(self):
        calls=[]
        async def policy(_messages,_cap):
            calls.append(1)
            return Reply('{"actions":[{"done":{}}]}')
        session=ScreenSession(policy,actor_label='test double')
        result={'policy_calls':0,'policy_pending':False}
        with self.assertRaises(SessionAborted):
            await session._infer([],result,lambda:0)
        await asyncio.sleep(0)
        self.assertEqual(calls,[])
        self.assertEqual(result['policy_calls'],0)
        self.assertFalse(result['policy_pending'])
        self.assertIsNone(session.pending)
        self.assertTrue(session.aborted)

    async def wait_pending(self, folder):
        for _ in range(100):
            if (folder/'waiting.json').exists():
                return next_observation(folder)
            await asyncio.sleep(.005)
        self.fail('manual observation was not published')

    async def test_manual_once_command_and_unknown_usage(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)/'bridge'
            policy = ManualPolicy(folder, 'test-session', timeout_s=1)
            policy.case_id = 'demo-case'
            task = asyncio.create_task(policy(policy_messages('Example goal', PNG, {'width':390,'height':844}, 1, 12, []), 128))
            record = await self.wait_pending(folder)
            self.assertEqual(record['screenshot_sha256'], hashlib.sha256(PNG).hexdigest())
            submit(folder, '{"actions":[{"click":{"x":10,"y":20}}]}')
            with self.assertRaises(FileExistsError):
                submit(folder, '{"actions":[{"done":{}}]}')
            reply = await task
            self.assertIsNone(reply.prompt_tokens)
            self.assertIsNone(reply.completion_tokens)
            self.assertTrue((folder/'consumed/000001.json').exists())
            self.assertEqual(next_observation(folder)['status'], 'consumed')

    async def test_manual_wrong_identity_aborts(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)/'bridge';policy = ManualPolicy(folder, 'test-session', timeout_s=1)
            policy.case_id = 'demo-case'
            task = asyncio.create_task(policy(policy_messages('Goal', PNG, {'width':390,'height':844}, 1, 12, []), 128))
            record = await self.wait_pending(folder)
            command = {k:record[k] for k in ('session_id','case_id','sequence','observation_sha256')}
            command.update(session_id='different',response={'actions':[{'done':{}}]})
            write_json(folder/'commands/000001.json',command,exclusive=True)
            with self.assertRaisesRegex(ValueError, 'mismatched'):
                await task
            self.assertTrue(policy.aborted)
            with self.assertRaises(RuntimeError):
                await policy([], 128)

    async def test_manual_expiry_is_not_a_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            policy = ManualPolicy(Path(tmp)/'bridge', 'test-session', timeout_s=.02)
            policy.case_id = 'demo-case'
            with self.assertRaises(TimeoutError):
                await policy(policy_messages('Goal', PNG, {'width':390,'height':844},1,12,[]),128)
            self.assertTrue(policy.aborted)

    async def test_unknown_policy_halts_session_before_next_call(self):
        calls = []
        async def blocked(_messages, _cap):
            calls.append(1)
            await asyncio.sleep(10)
        session = ScreenSession(blocked, actor_label='test double', limits=Limits(call_timeout_s=.02))
        result = {'policy_calls':0,'policy_pending':False}
        with self.assertRaises(SessionAborted):
            await session._infer([], result, 1)
        self.assertTrue(session.aborted)
        with self.assertRaises(SessionAborted):
            await session._infer([], result, 1)
        self.assertEqual(len(calls), 1)

    async def test_late_cancelled_reply_never_clears_fatal_session(self):
        async def ignores_cancellation(_messages,_cap):
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                return Reply('{"actions":[{"done":{}}]}',1,1)
        session=ScreenSession(ignores_cancellation,actor_label='test double',limits=Limits(call_timeout_s=.02))
        result={'policy_calls':0,'policy_pending':False}
        with self.assertRaises(SessionAborted):await session._infer([],result,1)
        await asyncio.sleep(0)
        self.assertTrue(session.aborted)
        self.assertTrue(result['policy_pending'])
        with self.assertRaises(SessionAborted):await session._infer([],result,1)

    async def test_existing_pending_command_is_not_deleted(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'bridge';policy=ManualPolicy(folder,'test-session',timeout_s=1)
            policy.case_id='demo-case'
            task=asyncio.create_task(policy(policy_messages('Goal',PNG,{'width':390,'height':844},1,12,[]),128))
            await self.wait_pending(folder)
            existing=folder/'commands/000001.pending';existing.write_text('another writer')
            with self.assertRaises(FileExistsError):submit(folder,'{"actions":[{"done":{}}]}')
            self.assertEqual(existing.read_text(),'another writer')
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):await task


class NativeBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def test_late_actual_receipt_does_not_grant_success(self):
        clock=[0.0];case=cases('development')[0]
        class Page:
            video=None
            count=0
            def set_default_timeout(self,_):pass
            async def set_content(self,*args,**kwargs):pass
            async def screenshot(self,**kwargs):return PNG
            async def evaluate(self,*args):
                self.count+=1
                if self.count==1:return None
                clock[0]=91.0
                return case.expected
        class Context:
            async def route(self,*args):pass
            async def new_page(self):return Page()
            async def close(self):pass
        async def new_context(**kwargs):return Context()
        async def policy(*args):return Reply('{"actions":[{"click":{"x":1,"y":1}}]}')
        async def action(*args):pass
        session=ScreenSession(policy,actor_label='test double')
        with tempfile.TemporaryDirectory() as tmp, patch('screen_harness.time',SimpleNamespace(monotonic=lambda:clock[0],time=lambda:123)), patch('screen_harness.execute_action',action):
            with self.assertRaises(SessionAborted):
                await session.run_case(SimpleNamespace(version='fake',new_context=new_context),case,Path(tmp)/'case')
            result=json.loads((Path(tmp)/'case/result.json').read_text())
        self.assertFalse(result['success'])
        self.assertEqual(result['late_final_receipt'],case.expected)
        self.assertEqual(result['stop_reason'],'receipt_observed_after_case_deadline')
        self.assertTrue(session.aborted)

    async def test_playwright_timeout_latches_session_and_refuses_next_case(self):
        from playwright.async_api import TimeoutError as PlaywrightTimeoutError
        class Page:
            video=None
            def set_default_timeout(self,_):pass
            async def set_content(self,*args,**kwargs):
                raise PlaywrightTimeoutError('native fake timeout')
        class Context:
            closed=False
            async def route(self,*args):pass
            async def new_page(self):return Page()
            async def close(self):self.closed=True
        context=Context();entered=[]
        async def new_context(**kwargs):entered.append(1);return context
        async def policy(*args):self.fail('policy must not be called')
        browser=SimpleNamespace(version='fake',new_context=new_context)
        session=ScreenSession(policy,actor_label='test double')
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SessionAborted) as caught:
                await session.run_case(browser,cases('development')[0],Path(tmp)/'case')
            result=json.loads((Path(tmp)/'case/result.json').read_text())
            self.assertEqual(result['stop_reason'],'case_native_disposition_unknown')
            self.assertTrue(result['context_close_returned'])
            self.assertTrue(session.aborted)
            with self.assertRaises(SessionAborted):
                await session.run_case(browser,cases('development')[1],Path(tmp)/'next')
        self.assertEqual(entered,[1])

    async def test_close_and_video_path_share_original_finalization_deadline(self):
        clock=[0.0];timeouts=[]
        class Video:
            async def path(self):raise TimeoutError('unreturned video receipt')
        class Page:
            video=Video()
            def set_default_timeout(self,_):pass
            async def set_content(self,*args,**kwargs):pass
            async def evaluate(self,*args):return None
            async def screenshot(self,**kwargs):return PNG
        class Context:
            async def route(self,*args):pass
            async def new_page(self):return Page()
            async def close(self):clock[0]+=.75
        async def new_context(**kwargs):return Context()
        async def policy(*args):return Reply('{"actions":[{"done":{}}]}')
        async def wait_for(awaitable,timeout):
            timeouts.append((awaitable.cr_code.co_name,timeout))
            return await awaitable
        session=ScreenSession(policy,actor_label='test double',limits=Limits(close_timeout_s=1))
        with tempfile.TemporaryDirectory() as tmp, patch('screen_harness.time',SimpleNamespace(monotonic=lambda:clock[0],time=lambda:123)), patch('screen_harness.asyncio.wait_for',wait_for):
            with self.assertRaises(SessionAborted):
                await session.run_case(SimpleNamespace(version='fake',new_context=new_context),cases('development')[0],Path(tmp)/'case')
            result=json.loads((Path(tmp)/'case/result.json').read_text())
        self.assertEqual([t for n,t in timeouts if n=='close'],[1])
        self.assertEqual([t for n,t in timeouts if n=='path'],[.25])
        self.assertTrue(result['context_close_returned'])
        self.assertTrue(session.aborted)
        self.assertIn('TimeoutError',result['artifact_error'])


@unittest.skipUnless(os.environ.get('ADAM_SCREEN_LOCAL_BROWSER_TESTS') == '1', 'opt-in fresh local fixture validation')
class BrowserTests(unittest.IsolatedAsyncioTestCase):
    async def test_actual_harness_video_and_manual_bridge_with_done_test_double(self):
        from playwright.async_api import async_playwright
        async with async_playwright() as pw:
            launch={'headless':True}
            if os.environ.get('ADAM_SCREEN_BROWSER_EXECUTABLE'):
                launch['executable_path']=os.environ['ADAM_SCREEN_BROWSER_EXECUTABLE']
            browser=await pw.chromium.launch(**launch)
            try:
                with tempfile.TemporaryDirectory() as tmp:
                    folder=Path(tmp)
                    policy=ManualPolicy(folder/'bridge','video-test-session',timeout_s=10)
                    case=cases('development')[5]
                    policy.case_id=case.case_id
                    session=ScreenSession(policy,actor_label='Scripted done test double; not a model',limits=Limits(call_timeout_s=10))
                    task=asyncio.create_task(session.run_case(browser,case,folder/'case',demonstration=True))
                    for _ in range(300):
                        if (folder/'bridge/waiting.json').exists() or task.done():break
                        await asyncio.sleep(.02)
                    if task.done():await task
                    record=next_observation(folder/'bridge')
                    self.assertEqual(record['observation']['viewport'],case.viewport)
                    submit(folder/'bridge','{"actions":[{"done":{}}]}')
                    result=await task
                    self.assertFalse(result['success'])
                    self.assertEqual(result['stop_reason'],'policy_done_without_oracle_success')
                    self.assertEqual(result['policy_calls'],1)
                    self.assertFalse(result['token_usage_complete'])
                    self.assertTrue(result['context_close_returned'])
                    self.assertEqual(len(result['frames']),2)
                    self.assertNotIn('artifact_error',result)
                    self.assertGreater(result['video']['bytes'],0)
                    self.assertTrue(result['video']['actual_browser_capture'])
            finally:
                await browser.close()

    async def test_development_ui_references_and_verification_render_only(self):
        from PIL import Image
        from playwright.async_api import async_playwright
        async with async_playwright() as pw:
            launch = {'headless':True}
            if os.environ.get('ADAM_SCREEN_BROWSER_EXECUTABLE'):
                launch['executable_path']=os.environ['ADAM_SCREEN_BROWSER_EXECUTABLE']
            browser = await pw.chromium.launch(**launch)
            print('Fresh fixture browser:',browser.version,'explicit binary:',launch.get('executable_path','Playwright installed default'))
            try:
                for split in ('development','verification'):
                    for case in cases(split):
                        context = await browser.new_context(viewport=case.viewport,device_scale_factor=1,service_workers='block')
                        await context.route('**/*',lambda route:route.abort())
                        page = await context.new_page()
                        await page.set_content(case.html)
                        image = Image.open(io.BytesIO(await page.screenshot()))
                        self.assertEqual(image.size,(case.viewport['width'],case.viewport['height']))
                        self.assertIsNone(await page.evaluate(RECEIPT_JS))
                        if split == 'development':
                            await self.reference(page,case)
                            actual=await page.evaluate(RECEIPT_JS)
                            self.assertTrue(check_receipt(case,actual),(case.case_id,actual))
                        await context.close()
            finally:
                await browser.close()

    async def click(self,page,selector):
        # Validator only. The runtime policy never receives selectors/bboxes.
        element=page.locator(selector)
        await element.scroll_into_view_if_needed()
        box=await element.bounding_box()
        await execute_action(page,{'click':{'x':int(box['x']+box['width']/2),'y':int(box['y']+box['height']/2)}})

    async def enter(self,page,selector,text):
        await self.click(page,selector)
        await execute_action(page,{'key':{'name':'SELECT_ALL'}})
        await execute_action(page,{'type':{'text':text}})

    async def reference(self,page,case):
        expected=case.expected
        if case.family=='contact':
            await self.enter(page,'#name',expected['name']);await self.enter(page,'#ticket',expected['ticket']);await self.click(page,'#form button:not([type])')
        elif case.family=='checklist':
            for text in expected['selected']:await self.click(page,f'input[value="{text}"]')
            await self.click(page,'#form button')
        elif case.family=='choice':
            await self.click(page,f'input[value="{expected["plan"]}"]');await self.click(page,'#form button')
        elif case.family=='dialog':
            await self.click(page,f'[data-project="{expected["archived"]}"]');await self.click(page,'#confirm')
        elif case.family=='tabs':
            await self.click(page,'#files');await self.click(page,f'[data-file="{expected["favorite"]}"]')
        elif case.family=='counter':
            for _ in range(expected['quantity']-1):await self.click(page,'#plus')
            await self.click(page,'#save')
        elif case.family=='scroll':
            await execute_action(page,{'scroll':{'x':200,'y':400,'dy':700}})
            await self.click(page,f'[data-destination="{expected["destination"]}"]')
        elif case.family=='search':
            await self.enter(page,'#query',expected['opened']);await self.click(page,'#form button');await self.click(page,'#results button')


if __name__ == '__main__':
    unittest.main()
