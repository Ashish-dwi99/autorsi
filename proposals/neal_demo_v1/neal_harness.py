"""Live isolated Neal demo; optional browser/model/SDK integrations are injected.

No site, provider, browser or model call occurs on import. A separate reviewed
wrapper admits resources, verifies the target recipe and owns native watchdogs.
The interactive assistant and Adam receive the same actor-local policy surface.
"""
from __future__ import annotations
import asyncio
import copy
import hashlib
import json
import time

from neal_protocol import (ACTORS, LEVELS, LEVEL_SECONDS, MAX_ACTIONS, MAX_REQUESTS, EMBED_ORIGIN,
    DemoAbort, validate_session, make_observation, validate_command, parse_batch,
    revalidate, policy_messages)


def _trusted_completion(events, row, ready_ms):
    for event in events:
        if (event.get('session_id')==row['session_id'] and event.get('actor')==row['actor']
                and event.get('level')==row['level'] and event.get('frame_id')==row['frame_id']
                and event.get('origin')==EMBED_ORIGIN and event.get('source_verified') is True
                and type(event.get('received_ms')) in (int,float)):
            elapsed=(event['received_ms']-ready_ms)/1000
            row['trusted_completion']=event
            row['completion_elapsed_from_wrapper_ready_s']=elapsed
            row['trusted_completion_received']=True
            first_action=row.get('first_executed_action_started_ms')
            after_action=(type(first_action) in (int,float) and event['received_ms']>=first_action)
            row['success']=0<=elapsed<=LEVEL_SECONDS and row['executed_actions']>0 and after_action
            row['late_or_preexisting_completion']=not (0<=elapsed<=LEVEL_SECONDS and after_action)
            return row['success']
    return False


async def run_actor(scene, actor, level, timing, admission, counters, adam_policy,
                    assistant_bridge, row, publish, *, clock=time.monotonic):
    """One actor, one fixed level. Unknown policy transport aborts the whole demo."""
    queue,history=[],[]
    start=timing['level_start_monotonic']
    row.update({'session_id':admission['session_id'],'actor':actor,'level':level,
        'frame_id':timing['frame_ids'][actor],'status':'active','success':False,
        'trusted_completion_received':False,'actions':0,'executed_actions':0,
        'policy_requests':0,'model_calls':0 if actor=='adam' else None,
        'prompt_tokens':0 if actor=='adam' else None,'completion_tokens':0 if actor=='adam' else None,
        'events':[],'stop_reason':'running','time_limit_s':LEVEL_SECONDS,
        'completion_elapsed_from_wrapper_ready_s':None})
    publish()
    try:
        while row['actions']<MAX_ACTIONS:
            remaining=LEVEL_SECONDS-(clock()-start)
            if remaining<=0:
                row['stop_reason']='wall_limit'
                break
            if _trusted_completion(await scene.completions(),row,timing['wrapper_performance_timing']['ready_ms']):
                row['stop_reason']='trusted_game_completion'
                break
            observed=await scene.observe(actor)
            if 'initial_screenshot_sha256' not in row:
                row['initial_screenshot_sha256']=hashlib.sha256(observed['png']).hexdigest()
                row['first_observation_monotonic']=clock()
                row['initial_visible_dom_sha256']=hashlib.sha256(json.dumps(observed['visible_dom'],sort_keys=True).encode()).hexdigest()
            remaining=LEVEL_SECONDS-(clock()-start)
            if remaining<=0:
                row['stop_reason']='wall_limit'
                break
            event={'dispatch':row['actions']+1,'kind':'queued_dispatch' if queue else 'policy_request'}
            row['events'].append(event)
            if not queue:
                counters[actor]+=1
                observation=make_observation(session_id=admission['session_id'],actor=actor,level=level,
                    seq=counters[actor],frame_id=observed['frame_id'],png=observed['png'],
                    visible_dom=observed['visible_dom'],remaining_actions=MAX_ACTIONS-row['actions'],
                    remaining_wall_s=remaining,prior_action_result=row.get('last_action_result'))
                event.update({'observation_sha256':observation['observation_sha256'],'seq':observation['seq']})
                row['policy_requests']+=1
                row['task_phase']='awaiting_policy'
                publish()
                request_start=clock()
                try:
                    if actor=='adam':
                        row['model_calls']+=1
                        reply=await asyncio.wait_for(adam_policy(policy_messages(observation,history),192),min(40,remaining))
                        raw=reply.text
                        for field in ('prompt_tokens','completion_tokens'):
                            value=getattr(reply,field,None)
                            if type(value) is not int or value<0:
                                raise ValueError('actual Adam token usage receipt required')
                            row[field]+=value
                    else:
                        command=await asyncio.wait_for(assistant_bridge.exchange(observation,timeout_s=min(30,remaining)),min(30,remaining))
                        raw=validate_command(command,observation)
                except BaseException as exc:
                    event['error']=f'{type(exc).__name__}: {exc}'[:512]
                    row['stop_reason']='policy_or_transport_disposition_unknown'
                    row['status']='failed'
                    publish()
                    raise DemoAbort('No next request or level after unknown policy/transport disposition') from exc
                event['policy_elapsed_s']=clock()-request_start
                if not isinstance(raw,str):
                    raise DemoAbort('policy did not return native JSON text')
                event['native_response']=raw[:8192]
                event['native_response_truncated']=len(raw)>8192
                history.append({'role':'assistant','content':raw[:8192]})
                row['task_phase']='active'
                try:
                    queue=list(parse_batch(raw,observed['visible_dom'],MAX_ACTIONS-row['actions']))
                except (ValueError,TypeError) as exc:
                    event['error']=str(exc)[:512]
                    row['stop_reason']='invalid_native_batch'
                    break
            if clock()-start>=LEVEL_SECONDS:
                row['stop_reason']='wall_limit_after_policy'
                break
            if _trusted_completion(await scene.completions(),row,timing['wrapper_performance_timing']['ready_ms']):
                row['stop_reason']='trusted_game_completion'
                break
            try:
                # Browser executor makes a second fresh visibility/identity check
                # under its global physical-action lock before an actual action.
                action=revalidate(queue[0],observed['visible_dom'])
                queue.pop(0)
                row['actions']+=1
                result=await scene.execute(actor,action,deadline_monotonic=start+LEVEL_SECONDS)
            except (ValueError,TypeError) as exc:
                event['error']=str(exc)[:512]
                row['stop_reason']='stale_or_invalid_action'
                break
            event['native_action']=action.native
            event['actual_action_result']=result
            row['last_action_result']=result
            row['executed_actions']+=int(result['executed'])
            if result['executed'] and 'first_executed_action_started_ms' not in row:
                row['first_executed_action_started_ms']=result['wrapper_action_started_ms']
            history.append({'role':'user','content':'Action result: '+json.dumps(result,sort_keys=True)+'. Success requires the game completion receipt.'})
            publish()
            if action.kind=='done':
                row['stop_reason']='done_claim_without_trusted_completion'
                break
        else:
            row['stop_reason']='action_limit'
        _trusted_completion(await scene.completions(),row,timing['wrapper_performance_timing']['ready_ms'])
        if row['success']:
            row['stop_reason']='trusted_game_completion'
        row['status']='completed'
        return row
    finally:
        row['discarded_queued_actions']=len(queue)
        queue.clear()
        row['elapsed_until_actor_stop_s']=clock()-start
        row['failure_time_is_censored']=not row['success']
        publish()


async def _run(admission,progress_path,artifact_dir,*,adam_policy=None,assistant_bridge=None,
               publish=None,scene_factory=None,preflight_only=False,clock=time.monotonic):
    session=validate_session(admission['session_id'])
    if tuple(admission.get('levels',LEVELS))!=LEVELS:
        raise ValueError('only frozen levels1,3,5 are admitted')
    if not preflight_only and (adam_policy is None or assistant_bridge is None):
        raise ValueError('separate actual Adam policy and authenticated assistant bridge required')
    if scene_factory is None:
        from neal_browser import BrowserScene
        scene_factory=BrowserScene
    report={'schema':1,'run':'neal-demo-v1','mode':'cpu_preflight' if preflight_only else 'live_demo',
        'session_id':session,'phase':'browser_setup','status':'running','levels':[],
        'planned_levels':list(LEVELS),'planned_actor_episodes':6,
        'comparison':'UNPAIRED qualitative live entertainment-game demonstration; no verified Astra API benchmark',
        'model_label':'Adam-1-CU, BU-derived experimental edit','assistant_label':'Interactive assistant in this session',
        'actual_model_identity':admission.get('actual_model_identity','Must be supplied by the reviewed policy wrapper'),
        'assistant_model_identity':admission.get('assistant_model_identity','Not supplied by trusted runtime metadata'),
        'browser_close_returned':False,'artifacts':[],'model_calls':0,'game_actions':0,
        'model_call_counter_scope':'Policy callback attempts here; actual native entry/return and full warmup accounting belong to the separate supervised model wrapper',
        'limits':{'levels':[1,3,5],'actions_per_actor_level':MAX_ACTIONS,'wall_seconds_per_level':90,
                  'adam_output_tokens':192,'adam_requests_max':MAX_REQUESTS,'assistant_requests_max':MAX_REQUESTS},
        'native_timer_synchronization':'Not established; game has no documented shared start/pause/seed API',
        'success_is_model_self_report':False}
    started=clock()
    def save():
        report['elapsed_s']=clock()-started
        rows=[row for level in report['levels'] for row in level.get('actors',{}).values()]
        report['model_calls']=sum(row.get('model_calls') or 0 for row in rows)
        report['game_actions']=sum(row.get('executed_actions',0) for row in rows)
        report['completed_actor_episodes']=sum(row.get('status')=='completed' for row in rows)
        report['successes_by_actor']={actor:sum(row.get('success',False) for row in rows if row.get('actor')==actor) for actor in ACTORS}
        report['denominator_per_actor']=3
        snapshot=copy.deepcopy(report)
        if publish is not None:
            publish(snapshot)  # synchronous owner callback preserves native pending/watchdog phase
        else:
            from weight_space_lab.diagnostic import atomic_json
            atomic_json(progress_path,snapshot)
    scene=scene_factory(admission,artifact_dir)
    counters={actor:0 for actor in ACTORS}
    tasks=[]
    try:
        save()
        await scene.open()
        for level in LEVELS:
            report['phase']='level_load'
            save()
            timing=await scene.start_level(level)
            timing['level_start_monotonic']=clock()
            level_row={'level':level,'timing':timing,'actors':{}}
            report['levels'].append(level_row)
            report['phase']='browser_step'
            save()
            if preflight_only:
                for actor in ACTORS:
                    observed=await scene.observe(actor)
                    observation=make_observation(session_id=session,actor=actor,level=level,seq=LEVELS.index(level)+1,
                        frame_id=observed['frame_id'],png=observed['png'],visible_dom=observed['visible_dom'],
                        remaining_actions=MAX_ACTIONS,remaining_wall_s=90,prior_action_result=None)
                    level_row['actors'][actor]={'actor':actor,'frame_id':observed['frame_id'],
                        'status':'observed_only','geometry':observed['geometry'],
                        'observation_sha256':observation['observation_sha256'],
                        'screenshot_sha256':hashlib.sha256(observed['png']).hexdigest(),
                        'png_bytes':len(observed['png']),'visible_element_count':len(observed['visible_dom']['elements']),
                        'model_calls':0,'executed_actions':0}
                    save()
            else:
                for actor in ACTORS:
                    row=level_row['actors'][actor]={}
                    tasks.append(asyncio.create_task(run_actor(scene,actor,level,timing,admission,counters,
                        adam_policy,assistant_bridge,row,save,clock=clock)))
                try:
                    await asyncio.gather(*tasks)
                except BaseException:
                    for task in tasks:
                        if not task.done(): task.cancel()
                    await asyncio.gather(*tasks,return_exceptions=True)
                    raise
                finally:
                    tasks=[]
                events=await scene.completions()
                for row in level_row['actors'].values():
                    _trusted_completion(events,row,timing['wrapper_performance_timing']['ready_ms'])
                    row['failure_time_is_censored']=not row['success']
                    if row['success']: row['stop_reason']='trusted_game_completion'
            await scene.stop_level()
            save()
        report['status']='completed'
    except BaseException as exc:
        report['status']='failed'
        report['error']={'type':type(exc).__name__,'message':str(exc)[:2000]}
        save()
        raise
    finally:
        report['phase']='finalizing'
        save()
        try:
            report['artifacts']=await scene.close()
            report['browser_close_returned']=True
            report['network_requests']=scene.network
            report['blocked_browser_requests']=scene.blocked_requests
            if preflight_only and scene.blocked_requests:
                report['status']='failed'
                report['error']={'type':'AssetAdmissionFailed','message':'Blocked public asset requests require review before a live model run'}
        except BaseException as exc:
            report['status']='failed'
            report['close_error']={'type':type(exc).__name__,'message':str(exc)[:2000]}
        report['phase']='completed' if report['status']=='completed' else 'failed'
        save()
    return report


async def preflight(admission,progress_path,artifact_dir,*,publish=None,scene_factory=None):
    """Three official levels, two fresh panes each, recorded; zero game actions/model calls."""
    return await _run(admission,progress_path,artifact_dir,publish=publish,scene_factory=scene_factory,preflight_only=True)


async def run_demo(admission,progress_path,artifact_dir,*,adam_policy,assistant_bridge,publish=None,scene_factory=None):
    return await _run(admission,progress_path,artifact_dir,adam_policy=adam_policy,
        assistant_bridge=assistant_bridge,publish=publish,scene_factory=scene_factory)
