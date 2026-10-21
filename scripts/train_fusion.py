if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score,confusion_matrix
from tqdm import tqdm
from src.settings import ROOT,load_config,model_dir
from src.data import balanced_indices,write_json
from src.metrics import report
from src.fusion import apply_policy

def choose_threshold(y,scores,gates,rules,max_fpr,require_gate=True):
    best=None
    for t in np.linspace(.35,.99,65):
        pred=apply_policy(scores,gates,rules,t,require_gate);tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel();fpr=fp/max(tn+fp,1);rank=(fpr<=max_fpr,float(balanced_accuracy_score(y,pred)),tp/max(tp+fn,1),-float(t))
        if best is None or rank>best[0]:best=(rank,float(t),float(fpr))
    return best[1],best[2]
def main():
    p=argparse.ArgumentParser(description='Exactly balanced logistic fusion with validation threshold tuning');p.add_argument('--data',type=Path,default=ROOT/'data/processed/fusion.npz');p.add_argument('--fast',action='store_true');args=p.parse_args();cfg=load_config();data=np.load(args.data,allow_pickle=False);meta=json.loads(args.data.with_suffix('.json').read_text());x,y,s=data['X'],data['y'],data['split'];train=np.flatnonzero(s=='fusion_train');val=np.flatnonzero(s=='fusion_val');test=np.flatnonzero(s=='test')
    for name,inds in [('train',train),('val',val),('test',test)]:
        if set(y[inds])!={0,1}:raise ValueError(name+' requires both classes.')
    n=min(Counter(y[train]).values());n=min(n,32) if args.fast else n;take=train[balanced_indices(y[train],cfg['training']['seed'],cap=n)];scaler=StandardScaler().fit(x[take]);xt=scaler.transform(x[take]);xv=scaler.transform(x[val]);best=None;stale=0
    for c in tqdm([.1,1.] if args.fast else [.01,.1,1.,10.,100.],desc='Fusion regularization'):
        model=LogisticRegression(C=c,max_iter=2000,tol=1e-5,random_state=42).fit(xt,y[take]);scores=model.predict_proba(xv)[:,1];t,fpr=choose_threshold(y[val],scores,data['gate'][val],data['rule'][val],cfg['fusion']['target_false_positive_rate'],cfg['fusion']['require_evidence_gate']);pred=apply_policy(scores,data['gate'][val],data['rule'][val],t,cfg['fusion']['require_evidence_gate']);rank=(fpr<=cfg['fusion']['target_false_positive_rate'],balanced_accuracy_score(y[val],pred))
        if best is None or rank>best[0]:best=(rank,model,t,fpr);stale=0
        else:
            stale+=1
            if stale>=3:break
    rank,model,t,fpr=best;ck={'features':meta['features'],'signature':meta['signature'],'base_sha256':meta['base_sha256'],'mean':scaler.mean_.tolist(),'scale':scaler.scale_.tolist(),'coef':model.coef_[0].tolist(),'intercept':float(model.intercept_[0]),'threshold':t,'fast':bool(args.fast or meta['fast']),'training_counts':{'Normal':n,'Distress':n},'validation_fpr':fpr}
    write_json(model_dir(cfg)/'fusion.json',ck);write_json(model_dir(cfg)/'fusion_labels.json',{'0':'Normal','1':'Distress'});write_json(ROOT/'reports/fusion_balance.json',{'before':dict(Counter(map(str,y[train]))),'after':ck['training_counts']})
    for name,inds in [('validation',val),('test',test)]:
        scores=model.predict_proba(scaler.transform(x[inds]))[:,1];pred=apply_policy(scores,data['gate'][inds],data['rule'][inds],t,cfg['fusion']['require_evidence_gate']);report(y[inds],pred,['Normal','Distress'],ROOT/'reports','fusion_'+name);count=min(Counter(y[inds]).values());bal=balanced_indices(y[inds],42,cap=count);report(y[inds][bal],pred[bal],['Normal','Distress'],ROOT/'reports','fusion_'+name+'_balanced')
    if not rank[0]:print('WARNING: validation false-positive target not achieved.')
    print('Saved fusion.json. Proxy-label scores are not real emergency accuracy.')
if __name__=='__main__':main()
