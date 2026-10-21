import asyncio
import logging
from pathlib import Path
import tempfile
from fastapi import FastAPI,File,HTTPException,UploadFile,WebSocket
from fastapi.responses import FileResponse,JSONResponse
from fastapi.staticfiles import StaticFiles
from src.settings import ROOT,load_config
from src.pipeline import Pipeline,ModelsUnavailable
from .websocket import live_socket

class LimitBody:
    def __init__(self,app,limit):self.app=app;self.limit=limit
    async def __call__(self,scope,receive,send):
        if scope['type']!='http':return await self.app(scope,receive,send)
        size=0;headers=dict(scope.get('headers',[]))
        try:declared=int(headers.get(b'content-length',b'0'))
        except ValueError:declared=self.limit+1
        if declared>self.limit:return await JSONResponse({'detail':'Upload exceeds file-size limit.'},status_code=413)(scope,receive,send)
        async def limited():
            nonlocal size
            message=await receive();size+=len(message.get('body',b''))
            if size>self.limit:raise HTTPException(413,'Upload exceeds file-size limit.')
            return message
        await self.app(scope,limited,send)

def create_app(pipeline=None):
    cfg=pipeline.cfg if pipeline else load_config();logging.basicConfig(level=cfg['server']['log_level'],format='%(asctime)s %(levelname)s %(name)s: %(message)s');pipe=pipeline or Pipeline(cfg);app=FastAPI(title='SoundSentinel',version='1.0.0');app.state.pipeline=pipe;app.add_middleware(LimitBody,limit=int(cfg['audio']['max_upload_mb']*1024**2)+65536);clients={'count':0};gate=asyncio.Semaphore(1)
    @app.get('/api/health')
    def health():return {'service':'SoundSentinel','status':'ok',**pipe.status()}
    @app.post('/api/analyze')
    async def analyze(file:UploadFile=File(...)):
        suffix=Path(file.filename or '').suffix.lower();path=None
        if suffix.lstrip('.') not in cfg['audio']['extensions']:await file.close();raise HTTPException(415,'Unsupported format. Use wav, mp3, ogg, flac, m4a or webm.')
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix,delete=False) as tmp:
                path=Path(tmp.name);count=0
                while True:
                    chunk=await file.read(1024*1024)
                    if not chunk:break
                    count+=len(chunk)
                    if count>cfg['audio']['max_upload_mb']*1024**2:raise HTTPException(413,'Audio file is too large.')
                    tmp.write(chunk)
            async with gate:return await asyncio.to_thread(pipe.analyze_file,path)
        except ModelsUnavailable as e:raise HTTPException(503,str(e)) from e
        except ValueError as e:raise HTTPException(422,str(e)) from e
        except HTTPException:raise
        except Exception as e:logging.getLogger(__name__).exception('Analysis failed');raise HTTPException(500,'Analysis failed. Check logs and run python -m scripts.check.') from e
        finally:
            await file.close()
            if path is not None:path.unlink(missing_ok=True)
    @app.websocket('/ws/live')
    async def live(ws:WebSocket):await live_socket(ws,pipe,clients)
    @app.get('/')
    def index():return FileResponse(ROOT/'app/static/index.html')
    app.mount('/static',StaticFiles(directory=ROOT/'app/static'),name='static');return app
app=create_app()
