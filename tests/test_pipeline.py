import numpy as np
import pytest
from src.audio_utils import fixed_window,windows,resample,read_audio
from src.pipeline import Pipeline,ModelsUnavailable

def test_fixed_preprocess(cfg):
    y,n=fixed_window(np.ones(32000)*.5,cfg);assert y.shape==(240000,) and n==32000 and y[0]==1 and not y[32000:].any()
def test_stereo_resample():
    y=resample(np.ones((48000,2)),48000);assert y.shape==(16000,) and np.isfinite(y).all()
def test_tail_window(cfg):
    pairs=list(windows(np.ones(256100),cfg));assert len(pairs)==3 and pairs[-1][0]+len(pairs[-1][1])/16000==pytest.approx(16.00625)
def test_nonfinite(cfg):
    with pytest.raises(ValueError):fixed_window(np.array([np.nan]),cfg)
def test_offsets(pipeline):
    r=pipeline.analyze_window(np.ones(16000)*.1,12);assert r['verdict']=='Normal' and r['words'][0]['start']==pytest.approx(12.1) and r['end']==13
def test_silence(pipeline):
    r=pipeline.analyze_window(np.zeros(240000));assert r['verdict']=='No signal' and r['confidence'] is None
def test_missing_models(cfg):
    with pytest.raises(ModelsUnavailable,match='train_emotion'):Pipeline(cfg).analyze_window(np.ones(240000))
def test_decode(cfg,wav_bytes,tmp_path):
    p=tmp_path/'sample.wav';p.write_bytes(wav_bytes);assert len(read_audio(p,cfg))==32000
