import json
from pathlib import Path
import numpy as np
from src.settings import signature
from src.fusion import Fusion,feature_names,vectorize,apply_policy
from src.data import balanced_indices
from scripts.train_fusion import choose_threshold

def event(label):return {'label':label,'probability':.97,'start':1.,'end':3.,'source':'test'}
def test_normal(cfg):assert Fusion(cfg).predict({'neutral':.9},{'rain':.8},[],[])['verdict']=='Normal'
def test_anger_alone(cfg):assert Fusion(cfg).predict({'angry':.99},{},[],[])['verdict']=='Normal'
def test_siren_alone(cfg):assert Fusion(cfg).predict({'neutral':.9},{},[],[event('siren')])['verdict']=='Normal'
def test_gunshot_explained(cfg):
    r=Fusion(cfg).predict({},{},[],[event('gunshot')]);assert r['verdict']=='Distress' and r['explanation'][0]['signal']=='gunshot'
def test_help_phrase(cfg):assert Fusion(cfg).predict({},{},[{'phrase':'help me','score':.95,'start':0.,'end':1.}],[])['verdict']=='Distress'
def test_weak_word(cfg):assert Fusion(cfg).predict({},{},[{'phrase':'stop','score':.45,'start':0.,'end':1.}],[])['verdict']=='Normal'
def test_fear_with_context(cfg):assert Fusion(cfg).predict({'fear':.9},{},[],[event('glass_breaking')])['verdict']=='Distress'
def test_feature_shape():assert vectorize({},{},[],[]).shape==(len(feature_names()),)
def test_exact_balance():
    y=np.array([0]*100+[1]*7+[2]*3);assert np.bincount(y[balanced_indices(y,42)]).tolist()==[100,100,100];assert np.bincount(y[balanced_indices(y,42,cap=3)]).tolist()==[3,3,3]
def test_threshold():
    y=np.array([0,0,1,1]);p=np.array([.2,.4,.8,.9]);g=np.ones(4,dtype=bool);r=np.zeros(4);t,f=choose_threshold(y,p,g,r,0.);assert f==0 and np.array_equal(apply_policy(p,g,r,t),y)
def test_trained_policy_consistency(cfg):
    n=len(feature_names());p=Path(cfg['model']['models_dir']);p.mkdir();ck={'features':feature_names(),'signature':signature(cfg),'mean':[0.]*n,'scale':[1.]*n,'coef':[0.]*n,'intercept':0.,'threshold':.65};(p/'fusion.json').write_text(json.dumps(ck));r=Fusion(cfg).predict({},{},[{'phrase':'help me','score':.95,'start':0.,'end':1.}],[]);assert r['verdict']=='Distress' and r['raw_fusion_score']==.5 and r['distress_score']==.95
def test_ignore_stale_when_rebuilding(cfg):
    p=Path(cfg['model']['models_dir']);p.mkdir();(p/'fusion.json').write_text('{}');assert Fusion(cfg,False).trained is None
