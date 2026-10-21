"""Build fusion data from held-out base-model predictions with explicit weak labels."""
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import argparse
import csv
import hashlib
from pathlib import Path
import numpy as np
from tqdm import tqdm
from src.settings import ROOT,load_config,model_dir,signature
from src.data import read_manifest,write_json,manifest_digest
from src.pipeline import Pipeline
from src.audio_utils import read_audio,windows
from src.fusion import feature_names,vectorize,evidence

def build(args):
    import torch
    cfg=load_config();pipe=Pipeline(cfg,use_trained_fusion=False);pipe.load()
    if pipe.transcriber is None:raise ValueError('Transcription is required: run python -m scripts.download_models.')
    rows=read_manifest(args.manifest)
    for task in ['emotion','environment']:
        ck=torch.load(model_dir(cfg)/f'{task}.pt',map_location='cpu',weights_only=True)
        if ck['manifest_sha256']!=manifest_digest(args.manifest):raise ValueError('Manifest changed since base model training; retrain both base models.')
    selected=[dict(r) for r in rows if r['split'] in ['fusion_train','fusion_val','test']]
    for r in selected:r['distress']=int((r['task']=='emotion' and r['label'] in cfg['fusion']['weak_positive_emotions']) or (r['task']=='environment' and r['label'] in cfg['fusion']['weak_positive_events']));r['label_source']='weak_dataset_proxy'
    if args.custom_labels:
        with args.custom_labels.open(encoding='utf-8',newline='') as f:custom=list(csv.DictReader(f))
        known={r['sha256'] for r in rows};groups={}
        for r in custom:
            if r['distress'] not in ['0','1'] or r['split'] not in ['fusion_train','fusion_val','test']:raise ValueError('Custom CSV requires distress=0/1 and a fusion_train/fusion_val/test split.')
            path=Path(r['path']);path=path if path.is_absolute() else args.custom_labels.parent/path;digest=hashlib.sha256(path.read_bytes()).hexdigest()
            if digest in known:raise ValueError('Custom file duplicates an existing recording.')
            if r['group'] in groups and groups[r['group']]!=r['split']:raise ValueError('Custom group leaks across splits.')
            known.add(digest);groups[r['group']]=r['split'];selected.append({'path':str(path.resolve()),'split':r['split'],'distress':int(r['distress']),'group':'custom:'+r['group'],'sha256':digest,'label_source':'human'})
    if args.fast:
        rng=np.random.default_rng(cfg['training']['seed']);small=[]
        for split in ['fusion_train','fusion_val','test']:
            for label in [0,1]:
                bucket=[r for r in selected if r['split']==split and r['distress']==label];small.extend(bucket[i] for i in rng.choice(len(bucket),min(8,len(bucket)),replace=False))
        selected=small
    X=[];Y=[];splits=[];gates=[];rules=[];provenance=[]
    for row in tqdm(selected,desc='Fusion features'):
        audio=read_audio(row['path'],cfg)
        for offset,chunk in windows(audio,cfg):
            r=pipe.analyze_window(chunk)
            if r['verdict']=='No signal':continue
            gate,reasons,rule=evidence(r['emotion'],r['keywords'],r['events'],cfg);X.append(vectorize(r['emotion'],r['environment'],r['keywords'],r['events']));Y.append(row['distress']);splits.append(row['split']);gates.append(gate);rules.append(rule);provenance.append({'sha256':row['sha256'],'group':row['group'],'source':row['label_source'],'offset':offset})
    for split in ['fusion_train','fusion_val','test']:
        if {y for y,s in zip(Y,splits) if s==split}!={0,1}:raise ValueError(f'{split} needs both labels.')
    args.output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(args.output,X=np.stack(X),y=np.array(Y),split=np.array(splits),gate=np.array(gates),rule=np.array(rules))
    write_json(args.output.with_suffix('.json'),{'features':feature_names(),'signature':signature(cfg),'fast':args.fast,'base_sha256':{k:hashlib.sha256((model_dir(cfg)/f'{k}.pt').read_bytes()).hexdigest() for k in ['emotion','environment']},'rows':provenance,'label_warning':'Weak proxies are not verified emergency labels.'});print('Saved',args.output,'Normal',Y.count(0),'Distress',Y.count(1))
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,default=ROOT/'data/processed/manifest.csv');p.add_argument('--output',type=Path,default=ROOT/'data/processed/fusion.npz');p.add_argument('--custom-labels',type=Path);p.add_argument('--fast',action='store_true');build(p.parse_args())
if __name__=='__main__':main()
