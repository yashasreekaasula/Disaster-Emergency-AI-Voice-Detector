if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import argparse
import json
from pathlib import Path
import numpy as np
from tqdm import tqdm
from src.settings import ROOT,load_config
from src.data import read_manifest,select_rows
from src.audio_utils import read_audio,fixed_window
from src.emotion import EmotionModel
from src.environment import EnvironmentModel
from src.fusion import Fusion,apply_policy
from src.metrics import report

def main():
    p=argparse.ArgumentParser(description='Evaluate untouched test partitions');p.add_argument('--task',choices=['all','emotion','environment','fusion'],default='all');p.add_argument('--manifest',type=Path,default=ROOT/'data/processed/manifest.csv');p.add_argument('--fusion-data',type=Path,default=ROOT/'data/processed/fusion.npz');p.add_argument('--fast',action='store_true');args=p.parse_args();cfg=load_config()
    for task in ['emotion','environment','fusion'] if args.task=='all' else [args.task]:
        if task=='fusion':
            data=np.load(args.fusion_data,allow_pickle=False);ck=Fusion(cfg).trained
            if ck is None:raise ValueError('Train fusion first.')
            meta=json.loads(args.fusion_data.with_suffix('.json').read_text())
            if meta['signature']!=ck['signature'] or meta['base_sha256']!=ck['base_sha256']:raise ValueError('Stale fusion features; rebuild them.')
            inds=np.flatnonzero(data['split']=='test');x=(data['X'][inds]-np.array(ck['mean']))/np.array(ck['scale']);scores=1/(1+np.exp(-np.clip(x@np.array(ck['coef'])+ck['intercept'],-50,50)));pred=apply_policy(scores,data['gate'][inds],data['rule'][inds],ck['threshold'],cfg['fusion']['require_evidence_gate']);report(data['y'][inds],pred,['Normal','Distress'],ROOT/'reports','evaluation_fusion')
        else:
            model=EmotionModel(cfg) if task=='emotion' else EnvironmentModel(cfg);ys=[];ps=[]
            for row in tqdm(select_rows(read_manifest(args.manifest),task,'test',args.fast,2),desc='Evaluate '+task):
                y,valid=fixed_window(read_audio(row['path'],cfg),cfg);probs=model.predict(y,valid);probs=probs[0] if task=='environment' else probs;ys.append(model.labels.index(row['label']));ps.append(model.labels.index(max(probs,key=probs.get)))
            report(ys,ps,model.labels,ROOT/'reports','evaluation_'+task)
if __name__=='__main__':main()
