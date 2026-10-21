from fastapi.testclient import TestClient
from app.main import create_app
from src.pipeline import Pipeline

def test_health_without_models(cfg):
    with TestClient(create_app(Pipeline(cfg))) as c:
        r=c.get('/api/health');assert r.status_code==200 and not r.json()['ready'];assert c.get('/').status_code==200
def test_missing_upload_models(cfg,wav_bytes):
    with TestClient(create_app(Pipeline(cfg))) as c:
        r=c.post('/api/analyze',files={'file':('x.wav',wav_bytes,'audio/wav')});assert r.status_code==503 and 'train_emotion' in r.json()['detail']
def test_upload_success(pipeline,wav_bytes):
    with TestClient(create_app(pipeline)) as c:
        r=c.post('/api/analyze',files={'file':('x.wav',wav_bytes,'audio/wav')});assert r.status_code==200 and r.json()['summary']['verdict']=='Normal'
def test_bad_extension(pipeline):
    with TestClient(create_app(pipeline)) as c:assert c.post('/api/analyze',files={'file':('x.txt',b'bad')}).status_code==415
def test_bad_audio(pipeline):
    with TestClient(create_app(pipeline)) as c:assert c.post('/api/analyze',files={'file':('x.wav',b'bad')}).status_code==422
def test_size_limit(pipeline):
    pipeline.cfg['audio']['max_upload_mb']=.001
    with TestClient(create_app(pipeline)) as c:assert c.post('/api/analyze',files={'file':('x.wav',b'x'*2000)}).status_code==413
def test_ws_protocol(pipeline):
    with TestClient(create_app(pipeline)) as c:
        with c.websocket_connect('/ws/live') as ws:
            ws.send_json({'sample_rate':44100,'encoding':'pcm_s16le'});assert ws.receive_json()['type']=='error'
def test_ws_windows(pipeline):
    import numpy as np
    with TestClient(create_app(pipeline)) as c:
        with c.websocket_connect('/ws/live') as ws:
            ws.send_json({'sample_rate':16000,'encoding':'pcm_s16le'});assert ws.receive_json()['type']=='ready';chunk=(np.ones(16000,dtype='<i2')*1000).tobytes()
            for i in range(15):ws.send_bytes(chunk)
            r=ws.receive_json();assert r['type']=='result' and r['result']['start']==0 and r['result']['end']==15
            ws.send_bytes(chunk);r=ws.receive_json();assert r['result']['start']==1 and r['result']['end']==16;ws.send_text('stop')
