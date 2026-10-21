import asyncio
import contextlib
import logging
import time
import numpy as np
from fastapi import WebSocketDisconnect
from src.pipeline import ModelsUnavailable
log=logging.getLogger(__name__)
async def live_socket(ws,pipeline,clients):
    cfg=pipeline.cfg;sc=cfg['server'];a=cfg['audio'];await ws.accept()
    if clients['count']>=sc['max_live_clients']:await ws.send_json({'type':'error','message':'All live slots are busy. Try again shortly.'});await ws.close(code=1013);return
    clients['count']+=1;worker=None;send_lock=asyncio.Lock()
    async def send(obj):
        async with send_lock:await ws.send_json(obj)
    try:
        hello=await asyncio.wait_for(ws.receive_json(),10)
        if hello.get('sample_rate')!=16000 or hello.get('encoding')!='pcm_s16le':raise ValueError('Expected 16000 Hz mono signed PCM16 little-endian.')
        await asyncio.to_thread(pipeline.load);await send({'type':'ready','window_seconds':15,'hop_seconds':a['hop_seconds']})
        n=240000;hop=int(16000*a['hop_seconds']);buffer=np.empty(0,dtype=np.float32);total=0;boundary=n;latest=None;dropped=0;available=asyncio.Event();started=time.monotonic()
        async def consume():
            nonlocal latest
            while True:
                await available.wait();available.clear();item=latest;latest=None
                if item is None:continue
                audio,offset=item;t=time.monotonic()
                try:
                    r=await asyncio.to_thread(pipeline.analyze_window,audio,offset);await send({'type':'result','result':r,'processing_seconds':round(time.monotonic()-t,3),'skipped_windows':dropped})
                except Exception as e:log.exception('Live inference failed');await send({'type':'error','message':str(e)})
        worker=asyncio.create_task(consume())
        while True:
            message=await asyncio.wait_for(ws.receive(),sc['live_timeout_seconds'])
            if message['type']=='websocket.disconnect':break
            if message.get('text') is not None:
                if message['text']=='stop':break
                raise ValueError('Send binary PCM chunks or the text stop.')
            raw=message.get('bytes')
            if not raw or len(raw)%2 or len(raw)>sc['max_ws_message_bytes']:raise ValueError('Invalid or oversized PCM chunk.')
            x=np.frombuffer(raw,dtype='<i2').astype(np.float32)/32768;old_total=total;total+=len(x);combined=np.concatenate((buffer,x));origin=old_total-len(buffer)
            while boundary<=total:
                start=boundary-n-origin;end=boundary-origin
                if start>=0:
                    if latest is not None:dropped+=1
                    latest=(combined[start:end].copy(),(boundary-n)/16000);available.set()
                boundary+=hop
            buffer=combined[-n:]
            if total/16000>sc['live_max_seconds'] or time.monotonic()-started>sc['live_max_seconds']+60:await send({'type':'info','message':'Session limit reached. Start a new session.'});break
    except WebSocketDisconnect:
        log.info('Live client disconnected')
    except (ValueError,ModelsUnavailable,asyncio.TimeoutError) as e:
        with contextlib.suppress(Exception):await send({'type':'error','message':str(e) or 'Live connection timed out.'})
    except Exception:
        log.exception('Live socket failed')
        with contextlib.suppress(Exception):await send({'type':'error','message':'Live audio failed. Check server logs.'})
    finally:
        if worker:
            worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):await worker
        clients['count']-=1
        with contextlib.suppress(Exception):await ws.close()
