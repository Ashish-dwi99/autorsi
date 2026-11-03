"""Optional isolated Playwright backend. No browser starts on import.

All JavaScript below is fixed harness code for visible DOM observations and an
owned wrapper completion listener. No actor can supply code, selectors or URLs.
No game variables, HTML source or answer maps are read or modified.
"""
from pathlib import Path
import asyncio
import hashlib
import json
import time
from urllib.parse import urlsplit

from neal_protocol import ACTORS, LEVELS, WIDTH, HEIGHT, VIEWPORT, EMBED_ORIGIN, DemoAbort


WRAPPER = """<!doctype html><html><head><style>
html,body{margin:0;width:970px;height:650px;overflow:hidden;background:#121824;color:white;font:14px sans-serif}
main{display:flex;gap:20px;padding:10px}section{width:465px}header{height:30px;font-size:13px}
iframe{width:465px;height:600px;border:0;display:block;background:white}small{color:#bdc9dc}
</style></head><body><main>
<section><header>Adam-1-CU · BU-derived experimental model</header><div id="pane-adam"></div></section>
<section><header>Interactive assistant · this session</header><div id="pane-assistant"></div></section>
</main></body></html>"""

START_LEVEL = r"""async ({session,level,bindings}) => {
  if(window.__nealOwnedListener) window.removeEventListener('message',window.__nealOwnedListener);
  const state={session,level,active:true,events:[],seen:{},frames:{},loads:{},created_ms:performance.now()};
  window.__nealOwnedState=state;
  const loads=[];
  for(const actor of ['adam','assistant']) {
    const pane=document.getElementById('pane-'+actor);
    pane.replaceChildren();
    const frame=document.createElement('iframe');
    frame.name=bindings[actor]; frame.title=actor+' official game';
    frame.referrerPolicy='no-referrer';
    const loaded=new Promise((resolve,reject)=>{
      const timer=setTimeout(()=>reject(new Error('official iframe load timeout')),30000);
      frame.addEventListener('load',()=>{clearTimeout(timer);state.loads[actor]=performance.now();resolve();},{once:true});
    });
    state.frames[actor]=frame;
    frame.src='https://embed.neal.fun/not-a-robot/'+level;
    pane.appendChild(frame);
    loads.push(loaded);
  }
  window.__nealOwnedListener=(event)=>{
    if(window.__nealOwnedState!==state || !state.active || event.origin!=='https://embed.neal.fun'
        || !event.data || typeof event.data!=='object' || event.data.type!=='captcha-completed') return;
    for(const actor of ['adam','assistant']) {
      const frame=state.frames[actor];
      if(event.source===frame.contentWindow && !state.seen[actor]) {
        state.seen[actor]=true;
        state.events.push({session_id:session,actor,level,frame_id:bindings[actor],origin:event.origin,
          source_verified:true,received_ms:performance.now()});
      }
    }
  };
  window.addEventListener('message',window.__nealOwnedListener);
  await Promise.all(loads);
  return {created_ms:state.created_ms,loads:state.loads,ready_ms:performance.now()};
}"""

VISIBLE = r"""(state) => {
  const elements=[], nodes=new Map(), labels=new Map();
  const rendered=(node)=>{
    for(let item=node;item && item.nodeType===1;item=item.parentElement) {
      const style=getComputedStyle(item);
      if(style.visibility==='hidden' || style.display==='none' || Number(style.opacity)===0) return false;
    }
    return true;
  };
  const list=document.querySelectorAll('a,button,input,textarea,select,canvas,label,[role="button"],[role="checkbox"],[tabindex]');
  if(list.length>512) throw new Error('visible candidate inventory exceeds512');
  for(const node of list) {
    const rect=node.getBoundingClientRect();
    if(!rendered(node)
       || rect.width<=0 || rect.height<=0 || rect.right<=0 || rect.bottom<=0 || rect.x>=465 || rect.y>=600) continue;
    if(!state.ids.has(node)) {if(state.next>4096) throw new Error('element identity cap');state.ids.set(node,state.next++);}
    const index=state.ids.get(node),tag=node.tagName.toLowerCase();
    const text=(node.getAttribute('aria-label') || '').slice(0,512);
    const value=(tag==='input' && node.type==='password') ? '[masked password field]' :
      ((tag==='input'||tag==='textarea'||tag==='select') ? String(node.value).slice(0,256) : '');
    const enabled=!node.disabled && node.getAttribute('aria-disabled')!=='true';
    elements.push({index,tag,text,value,enabled,focused:document.activeElement===node,
      box:[rect.x,rect.y,rect.width,rect.height]});
    nodes.set(index,node);
    if(!text) labels.set(node,elements[elements.length-1]);
    if(elements.length>128) throw new Error('visible actionable inventory exceeds128');
  }
  state.nodes=nodes;
  const walker=document.createTreeWalker(document.body,NodeFilter.SHOW_TEXT),parts=[];
  let textNode,visited=0,total=0;
  while((textNode=walker.nextNode())) {
    if(++visited>4096) throw new Error('visible text inventory exceeds4096');
    const parent=textNode.parentElement;
    if(!parent || ['SCRIPT','STYLE','NOSCRIPT','TEMPLATE'].includes(parent.tagName) || !rendered(parent)) continue;
    const range=document.createRange();range.selectNodeContents(textNode);
    const rect=range.getBoundingClientRect();
    if(rect.width<=0 || rect.height<=0 || rect.right<=0 || rect.bottom<=0 || rect.x>=465 || rect.y>=600) continue;
    const text=textNode.textContent.trim();
    if(text) {
      for(const [node,item] of labels) if(node.contains(textNode) && item.text.length<512)
        item.text=(item.text+' '+text).trim().slice(0,512);
      parts.push(text.slice(0,8000-total));total+=parts[parts.length-1].length;if(total>=8000) break;
    }
  }
  return {text:parts.join('\n'),elements};
}"""


def document_urls(level):
    """Only documented initial and root-observed canonical official URLs."""
    if level not in LEVELS:
        return frozenset()
    initial=f'{EMBED_ORIGIN}/not-a-robot/{level}'
    canonical=f'{EMBED_ORIGIN}/not-a-robot/embed/{level}'
    return frozenset((initial,initial+'/',canonical,canonical+'/'))


class BrowserScene:
    def __init__(self, admission, artifact_dir):
        self.session = admission["session_id"]
        allowed = admission.get("allowed_asset_origins", ["https://neal.fun", EMBED_ORIGIN])
        if (not isinstance(allowed, list) or EMBED_ORIGIN not in allowed or len(allowed)>8
                or any(urlsplit(origin).scheme!="https" or not urlsplit(origin).netloc
                       or urlsplit(origin).path not in ("", "/") or urlsplit(origin).query
                       or urlsplit(origin).fragment or urlsplit(origin).username for origin in allowed)):
            raise ValueError("explicit bounded HTTPS public-asset origin allowlist required")
        self.allowed_origins=set(origin.rstrip('/') for origin in allowed)
        self.artifact_dir=Path(artifact_dir)
        self.lock=asyncio.Lock()
        self.network=[]
        self.frames={}
        self.current_level=None
        self.playwright=self.browser=self.context=self.page=None
        self.closed=False
        self.video_started=False
        self.blocked_requests=0

    async def open(self):
        from playwright.async_api import async_playwright
        video_dir=self.artifact_dir/'browser-video'
        video_dir.mkdir(parents=True,exist_ok=False)
        self.playwright=await async_playwright().start()
        self.browser=await self.playwright.chromium.launch(headless=True)
        self.context=await self.browser.new_context(viewport=VIEWPORT,device_scale_factor=1,
            record_video_dir=str(video_dir),record_video_size=VIEWPORT,accept_downloads=False)
        await self.context.route('**/*',self._route)
        self.page=await self.context.new_page()
        self.video_started=True
        self.page.set_default_timeout(3000)
        await self.page.set_content(WRAPPER)

    async def _route(self,route):
        request=route.request
        url=request.url
        parsed=urlsplit(url)
        origin=f'{parsed.scheme}://{parsed.netloc}'
        allowed=origin in self.allowed_origins
        if request.resource_type=='document':
            allowed=allowed and url in document_urls(self.current_level) and request.frame.parent_frame==self.page.main_frame
        if len(self.network)<512:
            self.network.append({'url':url[:2048],'resource_type':request.resource_type,'allowed':bool(allowed)})
        else:
            allowed=False
        if allowed:
            await route.continue_()
        else:
            self.blocked_requests+=1
            await route.abort('blockedbyclient')

    async def start_level(self,level):
        if level not in LEVELS:
            raise ValueError('level outside frozen subset')
        async with self.lock:
            self.current_level=level
            bindings={actor:f'{self.session}:{level}:{actor}:fresh' for actor in ACTORS}
            navigation_started=time.monotonic()
            timing=await self.page.evaluate(START_LEVEL,{'session':self.session,'level':level,'bindings':bindings})
            self.frames={}
            for actor in ACTORS:
                node=await self.page.query_selector('#pane-'+actor+' iframe')
                frame=await node.content_frame()
                if frame is None or frame.url not in document_urls(level):
                    raise DemoAbort('iframe did not retain its official frozen URL')
                state=await frame.evaluate_handle('({ids:new WeakMap(),next:1,nodes:new Map()})')
                self.frames[actor]={'node':node,'frame':frame,'state':state,'frame_id':bindings[actor]}
                await self._geometry(actor)
            return {'navigation_started_monotonic':navigation_started,'ready_returned_monotonic':time.monotonic(),
                    'wrapper_performance_timing':timing,'frame_ids':bindings,
                    'native_timer_synchronization':'Not established; no documented start/pause API is used'}

    async def _geometry(self,actor):
        box=await self.frames[actor]['node'].bounding_box()
        if box is None or abs(box['width']-WIDTH)>.01 or abs(box['height']-HEIGHT)>.01:
            raise DemoAbort('iframe geometry changed from465x600 CSS pixels')
        if box['x']<0 or box['y']<0 or box['x']+WIDTH>VIEWPORT['width'] or box['y']+HEIGHT>VIEWPORT['height']:
            raise DemoAbort('iframe outside the owned video viewport')
        return box

    async def observe(self,actor):
        async with self.lock:
            box=await self._geometry(actor)
            state=self.frames[actor]['state']
            visible=await state.evaluate(VISIBLE)
            png=await self.frames[actor]['node'].screenshot(type='png')
            return {'png':png,'visible_dom':visible,'frame_id':self.frames[actor]['frame_id'],
                    'geometry':box,'captured_monotonic':time.monotonic()}

    async def completions(self):
        async with self.lock:
            return await self.page.evaluate('window.__nealOwnedState ? window.__nealOwnedState.events.slice() : []')

    async def execute(self,actor,action,*,deadline_monotonic):
        async with self.lock:
            if time.monotonic()>=deadline_monotonic:
                raise ValueError('actor wall expired before physical action admission')
            info=self.frames[actor]
            box=await self._geometry(actor)
            from neal_protocol import revalidate
            # Fresh DOM inspection and execution share the same global lock.
            visible=await info['state'].evaluate(VISIBLE)
            action=revalidate(action,visible)
            if time.monotonic()>=deadline_monotonic:
                raise ValueError('actor wall expired during fresh target verification')
            params=action.native[action.kind]
            if action.index is not None:
                handle=await info['state'].evaluate_handle('(state,index)=>state.nodes.get(index)',action.index)
                element=handle.as_element()
                if element is None or not await element.is_visible() or not await element.is_enabled():
                    raise ValueError('indexed target is stale, hidden or disabled')
            action_started_ms=await self.page.evaluate('performance.now()')
            if time.monotonic()>=deadline_monotonic:
                raise ValueError('actor wall expired before actual action expression')
            if action.kind=='click':
                if action.index is not None:
                    await element.click(timeout=3000)
                else:
                    await self.page.mouse.click(box['x']+params['coordinate_x'],box['y']+params['coordinate_y'])
            elif action.kind=='input':
                if params.get('clear',True):
                    await element.fill(params['text'],timeout=3000)
                else:
                    # The admitted browser is Linux Chromium. End alone means
                    # line-end in a textarea; Control+End reaches field-end.
                    await element.press('Control+End',timeout=3000)
                    await element.type(params['text'],timeout=3000)
            elif action.kind=='scroll':
                await self.page.mouse.move(box['x']+WIDTH/2,box['y']+HEIGHT/2)
                await self.page.mouse.wheel(0,HEIGHT*params['pages']*(1 if params['down'] else -1))
            elif action.kind=='send_keys':
                focused=await info['frame'].evaluate_handle('document.activeElement')
                element=focused.as_element()
                if element is None:
                    raise ValueError('no actor-frame focus target')
                await element.press(' ' if params['keys']=='Space' else params['keys'],timeout=3000)
            elif action.kind!='done':
                raise ValueError('unsupported validated action')
            return {'executed':action.kind!='done','action':action.native,
                    'claim_only':action.kind=='done','returned_monotonic':time.monotonic(),
                    'wrapper_action_started_ms':action_started_ms}

    async def stop_level(self):
        async with self.lock:
            if self.page is not None:
                await self.page.evaluate('if(window.__nealOwnedState) window.__nealOwnedState.active=false')

    async def close(self):
        video=self.page.video if self.page is not None else None
        if self.context is not None:
            await self.context.close()
        if self.browser is not None:
            await self.browser.close()
        if self.playwright is not None:
            await self.playwright.stop()
        self.closed=True
        if video is None:
            return []
        path=Path(await video.path())
        if not path.resolve().is_relative_to(self.artifact_dir.resolve()) or path.stat().st_size>256*1024*1024:
            raise DemoAbort('video path or256MiB artifact bound failed')
        digest=hashlib.sha256()
        with path.open('rb') as stream:
            while chunk:=stream.read(1024*1024):
                digest.update(chunk)
        return [{'relative_path':str(path.relative_to(self.artifact_dir)),'bytes':path.stat().st_size,
                 'sha256':digest.hexdigest(),'type':'video/webm','viewport':VIEWPORT,
                 'dimensions_scope':'configured capture dimensions; actual media metadata inspection remains required',
                 'size_bound_scope':'finalized artifact acceptance gate, not a filesystem write quota',
                 'content':'one continuous live browser-only side-by-side recording; no synthetic cursor'}]
