import argparse
from collections import Counter
import hashlib
from pathlib import Path
import random
import numpy as np
from tqdm import tqdm
from .settings import ROOT,load_config,load_yaml,signature,device_for,model_dir
from .data import read_manifest,select_rows,balanced_indices,manifest_digest,write_json
from .audio_utils import read_audio,fixed_window
from .features import log_mel,ASTFeatures
from .emotion import make_cnn
from .environment import make_head
from .metrics import report

def load_features(row,task,cfg,ast=None):
    key=hashlib.sha256((row['sha256']+signature(cfg)+task).encode()).hexdigest();cache=ROOT/'data/cache'/task/(key+'.npy')
    if cache.exists():return np.load(cache,allow_pickle=False,mmap_mode='r')
    y,valid=fixed_window(read_audio(row['path'],cfg),cfg);x=log_mel(y,cfg) if task=='emotion' else ast(y,valid)[0];cache.parent.mkdir(parents=True,exist_ok=True);np.save(cache,x,allow_pickle=False);return x

def train(task,argv=None):
    import torch
    from torch.utils.data import Dataset,DataLoader
    from sklearn.metrics import balanced_accuracy_score
    p=argparse.ArgumentParser(description=f'Train balanced {task} model');p.add_argument('--manifest',type=Path,default=ROOT/'data/processed/manifest.csv');p.add_argument('--epochs',type=int);p.add_argument('--batch-size',type=int);p.add_argument('--fast',action='store_true');args=p.parse_args(argv)
    cfg=load_config();tc=cfg['training'];seed=tc['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);device=device_for(cfg)
    rows=read_manifest(args.manifest);labels=load_yaml('label_maps.yaml')[task+'_labels'];chosen={s:select_rows(rows,task,s,args.fast,tc['fast_per_class'],seed) for s in ['base_train','base_val','test']}
    if set(labels)!={r['label'] for r in chosen['base_train']}:raise ValueError('Training data lacks required classes.')
    ast=ASTFeatures(cfg) if task=='environment' else None
    for split,records in chosen.items():
        for r in tqdm(records,desc=task+' '+split+' features'):load_features(r,task,cfg,ast)
    del ast
    class CachedDataset(Dataset):
        def __init__(self,records):self.records=records
        def __len__(self):return len(self.records)
        def __getitem__(self,i):
            row=self.records[i];x=load_features(row,task,cfg)
            if task=='environment':x=x.mean(0)
            return torch.from_numpy(np.array(x,copy=True)),labels.index(row['label'])
    datasets={s:CachedDataset(r) for s,r in chosen.items()};model=(make_cnn(len(labels)) if task=='emotion' else make_head(len(labels))).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=tc['learning_rate'],weight_decay=.01);loss_fn=torch.nn.CrossEntropyLoss();epochs=args.epochs or (tc['fast_epochs'] if args.fast else tc['epochs']);batch=args.batch_size or tc['batch_size']
    if epochs<1 or batch<1:raise ValueError('Epochs and batch size must be positive.')
    out=model_dir(cfg);out.mkdir(parents=True,exist_ok=True);best=-1.;stale=0;history=[]
    def predictions(split):
        model.eval();probs=[];truth=[]
        with torch.inference_mode():
            for xb,yb in DataLoader(datasets[split],batch_size=batch,num_workers=0):probs.append(model(xb.to(device)).softmax(-1).cpu().numpy());truth.extend(yb.tolist())
        return np.array(truth),np.concatenate(probs).argmax(-1)
    for epoch in range(epochs):
        y=np.array([labels.index(r['label']) for r in chosen['base_train']]);inds=balanced_indices(y,seed+epoch);print('Exact class exposure:',dict(Counter(labels[i] for i in y[inds])));model.train();total=0.
        for xb,yb in tqdm(DataLoader(datasets['base_train'],batch_size=batch,sampler=inds.tolist(),num_workers=0),desc=f'Epoch {epoch+1}/{epochs}'):
            xb,yb=xb.to(device),yb.to(device)
            if task=='emotion':
                xb=xb.clone();width=min(12,xb.shape[-1]//10);t=random.randrange(xb.shape[-1]-width+1);xb[:,:,:,t:t+width]=-1
            optimizer.zero_grad(set_to_none=True);loss=loss_fn(model(xb),yb);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step();total+=float(loss.detach())*len(yb)
        truth,pred=predictions('base_val');score=float(balanced_accuracy_score(truth,pred));history.append({'epoch':epoch+1,'loss':total/len(inds),'val_balanced_accuracy':score});print(history[-1])
        if score>best:
            best=score;stale=0;torch.save({'state_dict':{k:v.cpu() for k,v in model.state_dict().items()},'labels':labels,'signature':signature(cfg),'manifest_sha256':manifest_digest(args.manifest),'fast':args.fast,'task':task},out/f'{task}.pt')
        else:
            stale+=1
            if stale>=tc['patience']:print('Early stopping');break
    ck=torch.load(out/f'{task}.pt',map_location=device,weights_only=True);model.load_state_dict(ck['state_dict']);write_json(out/f'{task}_labels.json',{str(i):v for i,v in enumerate(labels)});write_json(ROOT/f'reports/{task}_history.json',history)
    for split in ['base_val','test']:
        truth,pred=predictions(split);report(truth,pred,labels,ROOT/'reports',task+'_'+split+('_fast' if args.fast else ''))
    print('Saved',out/f'{task}.pt')
