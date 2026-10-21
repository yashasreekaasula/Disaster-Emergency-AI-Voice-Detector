import copy
import io
import numpy as np
import pytest
import soundfile as sf
from src.settings import load_config
from src.pipeline import Pipeline
class FakeEmotion:
    def predict(self,y,valid):return {'neutral':.85,'fear':.05,'angry':.1}
class FakeEnvironment:
    def predict(self,y,valid):return {'rain':.9,'siren':.1},[]
class FakeTranscript:
    def predict(self,y,valid):return [{'word':'hello','start':.1,'end':.3,'probability':.9}],[]
@pytest.fixture
def cfg(tmp_path):
    c=copy.deepcopy(load_config());c['model']['models_dir']=str(tmp_path/'models');return c
@pytest.fixture
def pipeline(cfg):return Pipeline(cfg,FakeEmotion(),FakeEnvironment(),FakeTranscript())
@pytest.fixture
def wav_bytes():
    b=io.BytesIO();sf.write(b,.1*np.sin(2*np.pi*440*np.arange(32000)/16000),16000,format='WAV');return b.getvalue()
