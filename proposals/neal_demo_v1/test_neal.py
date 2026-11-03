"""Pure fake-backend tests; no site, browser, model, provider or artifact capture."""
import base64
import copy
import json
from types import SimpleNamespace
import unittest

from neal_protocol import (ACTORS, LEVELS, MAX_ACTIONS, MAX_REQUESTS, IDENTITY_KEYS, SYSTEM_PROMPT, DemoAbort,
    accept_completion, make_observation, validate_command, parse_batch, revalidate)
from neal_harness import preflight, run_demo, _trusted_completion


SESSION='test_session_123'


def visible(actor='adam'):
    return {'text':'actor-'+actor,'elements':[{'index':1,'tag':'button','text':'Click',
        'value':'','enabled':True,'focused':False,'box':[0,0,100,30]}]}


def observation(actor='assistant',seq=1,level=1):
    return make_observation(session_id=SESSION,actor=actor,level=level,seq=seq,
        frame_id=f'{SESSION}:{level}:{actor}:fresh',png=('only-'+actor).encode(),visible_dom=visible(actor),
        remaining_actions=MAX_ACTIONS,remaining_wall_s=90,prior_action_result=None)


def raw(actions):
    return json.dumps({'memory':'Observed visible game only','action':actions})


class ProtocolTests(unittest.TestCase):
    def test_delayed_old_window_wrong_origin_and_claims_are_not_completion(self):
        old,new=object(),object()
        args=dict(origin='https://embed.neal.fun',source=new,expected_window=new,
                  message={'type':'captcha-completed'},active=True,expected_level=3,current_level=3)
        self.assertTrue(accept_completion(**args))
        for changes in ({'source':old},{'origin':'https://evil.example'},{'active':False},
                        {'current_level':1},{'message':{'success':True}}):
            self.assertFalse(accept_completion(**{**args,**changes}))

    def test_command_is_bound_to_all_identity_fields(self):
        obs=observation()
        command={key:obs[key] for key in IDENTITY_KEYS}
        command['native_action_json']=raw([{'click':{'index':1}}])
        self.assertEqual(validate_command(command,obs),command['native_action_json'])
        for key in IDENTITY_KEYS:
            changed=copy.deepcopy(command)
            changed[key]=2 if key in ('level','seq') else 'wrong'
            with self.subTest(key=key),self.assertRaises(ValueError):
                validate_command(changed,obs)

    def test_atomic_prefix_rejection_and_no_unknown_tools(self):
        for bad in ({'click':{'index':999}},{'evaluate':{'code':'fake_success()'}},
                    {'send_keys':{'keys':'Control+L'}},{'input':{'index':1,'text':'x'}}):
            with self.subTest(bad=bad),self.assertRaises(ValueError):
                parse_batch(raw([{'click':{'index':1}},bad]),visible(),12)

    def test_coordinates_cannot_cross_actor_frame_or_mix_index(self):
        self.assertEqual(parse_batch(raw([{'click':{'coordinate_x':464,'coordinate_y':599}}]),visible(),12)[0].kind,'click')
        for params in ({'coordinate_x':465,'coordinate_y':10},{'coordinate_x':0,'coordinate_y':600},
                       {'coordinate_x':True,'coordinate_y':10},{'index':1,'coordinate_x':0,'coordinate_y':0}):
            with self.assertRaises(ValueError):
                parse_batch(raw([{'click':params}]),visible(),12)

    def test_stale_disabled_identity_and_done_suffix(self):
        action=parse_batch(raw([{'click':{'index':1}}]),visible(),12)[0]
        changed=visible();changed['elements'][0]['enabled']=False
        with self.assertRaises(ValueError): revalidate(action,changed)
        changed=visible();changed['elements'][0]['text']='A different target'
        with self.assertRaises(ValueError): revalidate(action,changed)
        with self.assertRaises(ValueError):
            parse_batch(raw([{'done':{'text':'won'}},{'click':{'index':1}}]),visible(),12)
        with self.assertRaises(ValueError): parse_batch(raw([{'click':{'index':1}}]*8),visible(),7)

    def test_completion_cannot_credit_load_only_or_late_event(self):
        row={'session_id':SESSION,'actor':'adam','level':1,'frame_id':f'{SESSION}:1:adam:fresh','executed_actions':0,'success':False}
        event={**{k:row[k] for k in ('session_id','actor','level','frame_id')},'origin':'https://embed.neal.fun','source_verified':True,'received_ms':1000}
        self.assertFalse(_trusted_completion([event],row,0))
        row['executed_actions']=1
        row['first_executed_action_started_ms']=1500
        self.assertFalse(_trusted_completion([event],row,0))
        row['first_executed_action_started_ms']=500
        self.assertTrue(_trusted_completion([event],row,0))
        self.assertFalse(_trusted_completion([{**event,'received_ms':90001}],row,0))

    def test_transport_bounds_and_no_astra_identity_claim(self):
        self.assertNotIn('Astra',SYSTEM_PROMPT)
        obs=observation()
        self.assertEqual(base64.b64decode(obs['png_base64']),b'only-assistant')
        with self.assertRaises(ValueError):
            make_observation(session_id=SESSION,actor='assistant',level=1,seq=1,
                frame_id=f'{SESSION}:1:assistant:fresh',png=b'x'*(512*1024+1),visible_dom=visible(),
                remaining_actions=12,remaining_wall_s=90,prior_action_result=None)

    def test_prospective_shared32_action96_request_bounds_keep_batch8(self):
        self.assertEqual((MAX_ACTIONS,MAX_REQUESTS),(32,96))
        self.assertEqual(observation(seq=96)['seq'],96)
        with self.assertRaises(ValueError): observation(seq=97)
        self.assertEqual(len(parse_batch(raw([{'click':{'index':1}}]*8),visible(),32)),8)
        with self.assertRaises(ValueError): parse_batch(raw([{'click':{'index':1}}]*9),visible(),32)


class FakeScene:
    instances=[]
    def __init__(self,admission,artifact_dir):
        self.session=admission['session_id'];self.level=None
        self.network=[];self.blocked_requests=0;self.actions=[];self.events=[];self.closed=False
        self.instances.append(self)
    async def open(self): pass
    async def start_level(self,level):
        self.level=level;self.events=[]
        return {'frame_ids':{actor:f'{self.session}:{level}:{actor}:fresh' for actor in ACTORS},
                'wrapper_performance_timing':{'ready_ms':0,'loads':{actor:0 for actor in ACTORS}},
                'navigation_started_monotonic':0,'ready_returned_monotonic':0}
    async def observe(self,actor):
        return {'png':('only-'+actor).encode(),'visible_dom':visible(actor),
                'frame_id':f'{self.session}:{self.level}:{actor}:fresh',
                'geometry':{'x':10 if actor=='adam' else 495,'y':40,'width':465,'height':600}}
    async def completions(self): return copy.deepcopy(self.events)
    async def execute(self,actor,action,*,deadline_monotonic):
        self.actions.append((self.level,actor,action))
        if action.kind!='done':
            self.events.append({'session_id':self.session,'actor':actor,'level':self.level,
                'frame_id':f'{self.session}:{self.level}:{actor}:fresh','origin':'https://embed.neal.fun',
                'source_verified':True,'received_ms':1000})
        return {'executed':action.kind!='done','action':action.native,'claim_only':action.kind=='done','returned_monotonic':0,
                'wrapper_action_started_ms':500}
    async def stop_level(self): pass
    async def close(self):
        self.closed=True
        return [{'relative_path':'browser-video/fake-test.webm','bytes':0,'sha256':'0'*64,'test_double':True}]


class Bridge:
    def __init__(self,action=None,tamper=None):
        self.observations=[];self.action=action or [{'click':{'index':1}}];self.tamper=tamper
    async def exchange(self,obs,*,timeout_s):
        self.observations.append(copy.deepcopy(obs))
        command={k:obs[k] for k in IDENTITY_KEYS}
        command['native_action_json']=raw(self.action)
        if self.tamper:command[self.tamper]='wrong'
        return command


class HarnessTests(unittest.IsolatedAsyncioTestCase):
    async def test_only_current_level_exact_official_redirect_urls_are_admitted(self):
        from neal_browser import BrowserScene
        scene=BrowserScene({'session_id':SESSION},'/unused')
        scene.current_level=3
        main=object();scene.page=SimpleNamespace(main_frame=main)
        class Route:
            def __init__(self,url,parent=main):
                self.request=SimpleNamespace(url=url,resource_type='document',frame=SimpleNamespace(parent_frame=parent))
                self.allowed=None
            async def continue_(self): self.allowed=True
            async def abort(self,*args): self.allowed=False
        for url,expected in (
            ('https://embed.neal.fun/not-a-robot/3',True),
            ('https://embed.neal.fun/not-a-robot/embed/3',True),
            ('https://embed.neal.fun/not-a-robot/embed/5',False),
            ('https://embed.neal.fun/not-a-robot/embed/3?alternate=1',False),
            ('https://evil.example/not-a-robot/embed/3',False)):
            route=Route(url);await scene._route(route)
            self.assertEqual(route.allowed,expected,url)
        nested=Route('https://embed.neal.fun/not-a-robot/embed/3',object())
        await scene._route(nested)
        self.assertFalse(nested.allowed)

    async def test_append_reaches_full_multiline_textarea_end(self):
        from neal_browser import BrowserScene
        class Element:
            value='first line\nlast line'
            cursor=0
            async def is_visible(self): return True
            async def is_enabled(self): return True
            async def press(self,key,**kwargs):
                if key=='End': self.cursor=self.value.index('\n')
                elif key=='Control+End': self.cursor=len(self.value)
                else: raise AssertionError('unexpected keyboard action')
            async def type(self,text,**kwargs):
                self.value=self.value[:self.cursor]+text+self.value[self.cursor:]
        element=Element()
        observed=visible();observed['elements'][0].update(tag='textarea',text='',value=element.value)
        class State:
            async def evaluate(self,*args): return observed
            async def evaluate_handle(self,*args): return SimpleNamespace(as_element=lambda:element)
        class Page:
            async def evaluate(self,*args): return 500
        scene=BrowserScene({'session_id':SESSION},'/unused')
        scene.frames={'adam':{'state':State()}}
        scene.page=Page()
        async def geometry(actor): return {'x':0,'y':0,'width':465,'height':600}
        scene._geometry=geometry
        action=parse_batch(raw([{'input':{'index':1,'text':' appended','clear':False}}]),observed,12)[0]
        result=await scene.execute('adam',action,deadline_monotonic=float('inf'))
        self.assertTrue(result['executed'])
        self.assertEqual(element.value,'first line\nlast line appended')

    async def test_live_pair_shares_interface_and_each_actor_sees_only_own_pane(self):
        calls=[];bridge=Bridge();saved=[]
        async def policy(messages,cap):
            calls.append(messages)
            text=json.dumps(messages)
            self.assertNotIn('actor-assistant',text)
            self.assertEqual(cap,192)
            data=messages[-1]['content'][-1]['image_url']['url'].split(',',1)[1]
            self.assertEqual(base64.b64decode(data),b'only-adam')
            return SimpleNamespace(text=raw([{'click':{'index':1}},{'done':{'text':'clicked','success':True}}]),prompt_tokens=12,completion_tokens=9)
        result=await run_demo({'session_id':SESSION},'/unused','/unused',adam_policy=policy,
            assistant_bridge=bridge,publish=saved.append,scene_factory=FakeScene)
        self.assertEqual(result['status'],'completed')
        self.assertEqual(result['successes_by_actor'],{'adam':3,'assistant':3})
        self.assertEqual(result['completed_actor_episodes'],6)
        self.assertEqual(result['model_calls'],3)
        self.assertEqual([x['seq'] for x in bridge.observations],[1,2,3])
        self.assertEqual([x['level'] for x in bridge.observations],[1,3,5])
        for obs in bridge.observations:
            self.assertEqual(base64.b64decode(obs['png_base64']),b'only-assistant')
            self.assertNotIn('actor-adam',json.dumps(obs['visible_dom']))
        self.assertTrue(result['browser_close_returned'])
        self.assertEqual([level['actors']['adam']['discarded_queued_actions'] for level in result['levels']],[1,1,1])

    async def test_preflight_has_no_model_or_game_actions(self):
        result=await preflight({'session_id':SESSION},'/unused','/unused',publish=lambda report:None,scene_factory=FakeScene)
        self.assertEqual((result['model_calls'],result['game_actions']), (0,0))
        self.assertEqual(len(result['levels']),3)
        self.assertEqual(FakeScene.instances[-1].actions,[])
        self.assertTrue(FakeScene.instances[-1].closed)

    async def test_unknown_model_disposition_aborts_no_next_level(self):
        calls=[];saved=[]
        async def policy(*args):
            calls.append(True)
            raise TimeoutError('native owner may still be running')
        with self.assertRaises(DemoAbort):
            await run_demo({'session_id':SESSION},'/unused','/unused',adam_policy=policy,
                assistant_bridge=Bridge(),publish=saved.append,scene_factory=FakeScene)
        self.assertEqual(len(calls),1)
        self.assertEqual(len(saved[-1]['levels']),1)
        self.assertEqual(saved[-1]['planned_actor_episodes'],6)
        self.assertEqual(saved[-1]['status'],'failed')
        self.assertTrue(FakeScene.instances[-1].closed)

    async def test_stale_bridge_command_never_executes_assistant_action(self):
        saved=[]
        async def policy(*args):
            return SimpleNamespace(text=raw([{'done':{'text':'not solved','success':False}}]),prompt_tokens=1,completion_tokens=1)
        with self.assertRaises(DemoAbort):
            await run_demo({'session_id':SESSION},'/unused','/unused',adam_policy=policy,
                assistant_bridge=Bridge(tamper='frame_id'),publish=saved.append,scene_factory=FakeScene)
        self.assertFalse(any(actor=='assistant' for _,actor,_ in FakeScene.instances[-1].actions))

    async def test_done_claim_does_not_score(self):
        async def policy(*args):
            return SimpleNamespace(text=raw([{'done':{'text':'I won','success':True}}]),prompt_tokens=1,completion_tokens=1)
        result=await run_demo({'session_id':SESSION},'/unused','/unused',adam_policy=policy,
            assistant_bridge=Bridge([{'done':{'text':'I won','success':True}}]),publish=lambda report:None,scene_factory=FakeScene)
        self.assertEqual(result['successes_by_actor'],{'adam':0,'assistant':0})
        self.assertEqual(result['game_actions'],0)


if __name__=='__main__':unittest.main()
