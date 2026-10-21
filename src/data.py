import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from .settings import ROOT
PARTITIONS=['base_train','base_val','fusion_train','fusion_val','test']
def read_manifest(path=None):
    p=Path(path or ROOT/'data/processed/manifest.csv')
    if not p.exists():raise ValueError('Manifest missing. Run python -m scripts.prepare_data.')
    with p.open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def manifest_digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_json(path,obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,indent=2),encoding='utf-8')
def balanced_indices(labels,seed,cap=None):
    labels=np.asarray(labels);groups=[np.flatnonzero(labels==c) for c in np.unique(labels)]
    if len(groups)<2:raise ValueError('At least two training classes are required.')
    n=max(map(len,groups)) if cap is None else cap;rng=np.random.default_rng(seed);indices=np.concatenate([rng.choice(g,n,replace=len(g)<n) for g in groups]);rng.shuffle(indices);return indices
def select_rows(rows,task,split,fast=False,cap=2,seed=42):
    selected=[r for r in rows if r['task']==task and r['split']==split]
    if not selected:raise ValueError(f'No {task}/{split} data.')
    if fast:
        rng=np.random.default_rng(seed);out=[]
        for label in sorted({r['label'] for r in selected}):
            group=[r for r in selected if r['label']==label];out.extend(group[i] for i in rng.choice(len(group),min(cap,len(group)),replace=False))
        return out
    return selected
