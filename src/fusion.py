import hashlib
import json
import numpy as np
from .settings import load_yaml,model_dir,signature

def feature_names():
    m=load_yaml('label_maps.yaml');return ['emotion:'+x for x in m['emotion_labels']]+['environment:'+x for x in m['environment_labels']]+['event:scream','keyword_max','keyword_count']
def vectorize(emotion,environment,hits,events):
    v={'emotion:'+k:p for k,p in emotion.items()};v.update({'environment:'+k:p for k,p in environment.items()})
    v['event:scream']=max([e['probability'] for e in events if e['label']=='scream'] or [0.]);v['keyword_max']=max([h['score'] for h in hits] or [0.]);v['keyword_count']=min(len(hits),5)/5.
    return np.array([v.get(k,0.) for k in feature_names()],dtype=np.float32)
def evidence(emotion,hits,events,cfg):
    f=cfg['fusion'];fear=emotion.get('fear',0.)>=f['fear_threshold'];words=[h for h in hits if h['score']>=f['keyword_threshold']]
    strong=[e for e in events if e['label'] in f['strong_events'] and e['probability']>=f['event_threshold']]
    context=[e for e in events if e['label'] in f['contextual_events'] and e['probability']>=f['event_threshold']]
    gate=bool(words or strong or (fear and (hits or context)))
    reasons=[{'type':'keyword','signal':h['phrase'],'score':h['score'],'start':h['start'],'end':h['end']} for h in hits]
    reasons += [{'type':'acoustic','signal':e['label'],'score':e['probability'],'start':e['start'],'end':e['end'],'source':e['source']} for e in strong+context]
    if fear:reasons.append({'type':'emotion','signal':'fear','score':emotion['fear'],'start':None,'end':None})
    rule=max([h['score'] for h in words]+[e['probability'] for e in strong]+([.8] if fear and (hits or context) else [])+[.15])
    return gate,reasons,rule

def apply_policy(scores,gates,rules,threshold,require_gate=True):
    scores=np.maximum(scores,np.where(rules>=.9,rules,0.))
    return ((scores>=threshold)&(gates if require_gate else np.ones(len(scores),dtype=bool))).astype(int)
class Fusion:
    def __init__(self,cfg,load_trained=True):
        self.cfg=cfg;self.trained=None;p=model_dir(cfg)/'fusion.json'
        if load_trained and p.exists():
            ck=json.loads(p.read_text())
            if ck['features']!=feature_names() or ck['signature']!=signature(cfg):raise ValueError('Fusion features changed; rebuild and retrain fusion.')
            for kind,digest in ck.get('base_sha256',{}).items():
                base=model_dir(cfg)/f'{kind}.pt'
                if not base.exists() or hashlib.sha256(base.read_bytes()).hexdigest()!=digest:raise ValueError('Base models changed; rebuild and retrain fusion.')
            self.trained=ck
    def predict(self,emotion,environment,hits,events):
        gate,reasons,rule=evidence(emotion,hits,events,self.cfg);score=rule;t=self.cfg['fusion']['threshold'];method='rule_fallback'
        if self.trained:
            ck=self.trained;x=(vectorize(emotion,environment,hits,events)-np.array(ck['mean']))/np.array(ck['scale']);z=float(x@np.array(ck['coef'])+ck['intercept']);score=float(1/(1+np.exp(-np.clip(z,-50,50))));t=ck['threshold'];method='trained_logistic'
        raw=score
        if self.trained and rule>=.9:score=max(score,rule);method='trained_logistic_with_evidence_rules'
        allowed=gate or not self.cfg['fusion']['require_evidence_gate'];positive=bool(score>=t and allowed)
        decision=score if allowed else min(score,.49)
        return {'verdict':'Distress' if positive else 'Normal','confidence':float(decision if positive else 1-decision),'distress_score':float(decision),'raw_fusion_score':raw,'threshold':t,'method':method,'evidence_gate_passed':gate,'confidence_kind':'uncalibrated decision score','explanation':reasons if positive else [],'observed_signals':reasons}
