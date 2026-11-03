"""Fresh source/browser/render admission for development only; no model policy."""
import asyncio
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
from screen_cases import cases
from screen_contract import artifact_inventory, spec


def browser_runtime_manifest(executable):
    if importlib.metadata.version('playwright')!='1.60.0':raise ValueError('pinned Playwright1.60 required')
    path=Path(executable)
    if path.is_symlink() or not path.is_file() or path.stat().st_size>600*1024**2:
        raise ValueError('ordinary bounded installed Chromium binary required')
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):digest.update(block)
    return {'playwright_version':'1.60.0','chromium_executable':str(path),
        'chromium_executable_bytes':path.stat().st_size,'chromium_executable_sha256':digest.hexdigest(),
        'scope':'Installed executable hash/version and actual browser.version below; not complete OS/native library provenance'}


def validate_preflight(receipt):
    expected=cases('development')
    if (receipt.get('status')!='completed' or receipt.get('phase')!='screen_development_preflight'
        or receipt.get('model_calls')!=0 or receipt.get('game_actions')!=0
        or receipt.get('browser_close_returned') is not True or receipt.get('experiment_spec')!=spec()
        or len(receipt.get('cases',[]))!=8):
        raise ValueError('complete fresh screenshot CPU preflight required')
    output={}
    for case,row in zip(expected,receipt['cases'],strict=True):
        renders=row.get('renders',[])
        if row.get('manifest')!=case.manifest() or len(renders)!=2:
            raise ValueError('paired development render manifests differ')
        for render in renders:
            if (render.get('context_close_returned') is not True or render.get('initial_receipt') is not None
                or render.get('network_requests_blocked')!=0 or not render.get('video')
                or render.get('viewport')!=case.viewport or not render.get('screenshot_sha256')):
                raise ValueError('fresh normal render/source/video/close receipts required')
        if renders[0]['screenshot_sha256']!=renders[1]['screenshot_sha256']:
            raise ValueError('paired initial pixels differ')
        output[case.case_id]=renders[0]['screenshot_sha256']
    return output


async def preflight(artifact_dir, save):
    from playwright.async_api import async_playwright
    started=time.monotonic();directory=Path(artifact_dir);directory.mkdir(exist_ok=False)
    report={'schema':1,'phase':'screen_development_preflight','status':'running',
        'experiment_spec':spec(),'model_calls':0,'game_actions':0,'cases':[],
        'browser_close_returned':False,'verification_rendered':False}
    browser=None
    try:
        async with async_playwright() as pw:
            report['runtime_manifest']=browser_runtime_manifest(pw.chromium.executable_path)
            browser=await asyncio.wait_for(pw.chromium.launch(headless=True,executable_path=pw.chromium.executable_path),30)
            report['browser_version']=browser.version
            save(report)
            for index,case in enumerate(cases('development')):
                row={'manifest':case.manifest(),'renders':[]};report['cases'].append(row)
                for member in range(2):
                    case_dir=directory/f'case-{index:02d}-pair-{member}';case_dir.mkdir()
                    record={'viewport':case.viewport,'context_close_returned':False,
                        'network_requests_blocked':0};row['renders'].append(record);save(report)
                    context=await asyncio.wait_for(browser.new_context(viewport=case.viewport,
                        device_scale_factor=1,record_video_dir=str(case_dir),record_video_size=case.viewport,
                        accept_downloads=False,service_workers='block'),10)
                    page=None
                    try:
                        async def block(route):
                            record['network_requests_blocked']+=1;await route.abort('blockedbyclient')
                        await context.route('**/*',block)
                        page=await context.new_page()
                        await page.set_content(case.html,wait_until='load',timeout=3000)
                        record['initial_receipt']=await page.evaluate('() => window.__screenReceipt')
                        if record['initial_receipt'] is not None:raise ValueError('unexpected initial task outcome')
                        png=await page.screenshot(type='png',full_page=False,animations='disabled',timeout=5000)
                        if len(png)>1024**2:raise ValueError('preflight PNG exceeds1MiB')
                        with (case_dir/'initial.png').open('xb') as stream:stream.write(png)
                        record['screenshot_sha256']=hashlib.sha256(png).hexdigest()
                        record['screenshot_bytes']=len(png);save(report)
                    finally:
                        deadline=time.monotonic()+10
                        await asyncio.wait_for(context.close(),max(.001,deadline-time.monotonic()))
                        record['context_close_returned']=True;save(report)
                        if page is None or page.video is None:raise ValueError('actual CPU video missing')
                        video=Path(await asyncio.wait_for(page.video.path(),max(.001,deadline-time.monotonic())))
                        if video.parent.resolve()!=case_dir.resolve() or video.is_symlink():raise ValueError('owned video required')
                        if not 0<video.stat().st_size<=8*1024**2:raise ValueError('CPU video exceeds8MiB')
                        raw=video.read_bytes()
                        record['video']={'relative_path':video.relative_to(directory).as_posix(),
                            'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'actual_browser_capture':True}
                        save(report)
            await asyncio.wait_for(browser.close(),10);report['browser_close_returned']=True;browser=None
        report['artifacts']=artifact_inventory(directory)
        report['status']='completed';validate_preflight(report);save(report)
        return report
    except BaseException as exc:
        report.update(status='failed',error={'type':type(exc).__name__,'message':str(exc)[:1000]})
        save(report);raise
    finally:
        if browser is not None:
            try:
                await asyncio.wait_for(browser.close(),10);report['browser_close_returned']=True
            except BaseException as exc:report['browser_close_error']=type(exc).__name__
        report['elapsed_s']=time.monotonic()-started;save(report)
