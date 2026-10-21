"""Parse six datasets and split by speakers or original recording folds."""
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import argparse
from collections import Counter
import csv
import hashlib
from pathlib import Path
import re
import numpy as np
import soundfile as sf
from tqdm import tqdm
from src.settings import ROOT,load_config,load_yaml
from src.data import PARTITIONS,write_json

def speaker_partitions(groups,seed):
    groups=sorted(set(groups));np.random.default_rng(seed).shuffle(groups);n=len(groups)
    if n==1:return {groups[0]:'base_train'}
    if n==2:return dict(zip(groups,['base_train','test']))
    if n<5:return {g:('test' if i==n-1 else 'base_val' if i==n-2 else 'base_train') for i,g in enumerate(groups)}
    tail=max(1,n//10);counts=[n-4*tail,tail,tail,tail,tail];out={};pos=0
    for split,count in zip(PARTITIONS,counts):
        for g in groups[pos:pos+count]:out[g]=split
        pos+=count
    return out

def prepare(root,output,cfg):
    maps=load_yaml('label_maps.yaml');rows=[];errors=[]
    def add(path,dataset,task,label,group,split=''):
        rows.append({'path':str(path.resolve()),'dataset':dataset,'task':task,'label':label,'group':dataset+':'+str(group),'split':split})
    for ds in ['ravdess','crema','tess','savee']:
        files=sorted((root/ds).rglob('*.wav'))+sorted((root/ds).rglob('*.WAV'))
        if not files:errors.append(f'No WAV files under {root/ds}')
        for path in files:
            try:
                stem=path.stem
                if ds=='ravdess':
                    f=stem.split('-')
                    if len(f)!=7:raise ValueError('Expected seven-part RAVDESS filename')
                    if f[:2]!=['03','01']:continue
                    label,group=maps[ds][f[2]],f[6]
                elif ds=='crema':
                    f=stem.split('_');label,group=maps[ds][f[2]],f[0]
                elif ds=='tess':
                    f=stem.lower().split('_');label,group=maps[ds][f[-1]],f[0]
                else:
                    match=re.search(r'(sa|su|a|d|f|h|n)([0-9]+)$',stem.lower())
                    if not match:raise ValueError('Expected SAVEE filename')
                    label=maps[ds][match.group(1)];group=(stem[:match.start()].strip('_-') or path.parent.name).upper()
                    if group not in {'DC','JE','JK','KL'}:raise ValueError('Unknown SAVEE speaker')
                add(path,ds,'emotion',label,group)
            except (ValueError,KeyError,IndexError) as e:errors.append(f'{path}: {e}')
        selected=[r for r in rows if r['dataset']==ds];assign=speaker_partitions([r['group'] for r in selected],cfg['training']['seed'])
        for r in selected:r['split']=assign[r['group']]
    for ds,csvname in [('esc50','esc50.csv'),('urban','UrbanSound8K.csv')]:
        candidates=list((root/ds).rglob(csvname))
        if len(candidates)!=1:errors.append(f'Expected one {csvname} under {root/ds}; found {len(candidates)}');continue
        base=candidates[0].parent.parent
        with candidates[0].open(encoding='utf-8',newline='') as f:
            for r in csv.DictReader(f):
                fold=int(r['fold'])
                if ds=='esc50':
                    path=base/'audio'/r['filename'];label=maps[ds][r['category']];group=r.get('src_file') or r['filename'].split('-')[1]
                    split='base_train' if fold<=2 else 'base_val' if fold==3 else 'test' if fold==5 else 'fusion_train'
                    if fold==4 and int(hashlib.sha256(str(group).encode()).hexdigest()[:8],16)%3==0:split='fusion_val'
                else:
                    path=base/'audio'/f'fold{fold}'/r['slice_file_name'];label=maps[ds][r['class']];group=r['fsID'];split='base_train' if fold<=6 else {7:'base_val',8:'fusion_train',9:'fusion_val',10:'test'}[fold]
                add(path,ds,'environment',label,group,split)
    seen={};clean=[];group_splits={}
    for r in tqdm(rows,desc='Validate audio and deduplicate'):
        try:
            path=Path(r['path']);info=sf.info(path)
            if not cfg['audio']['min_seconds']<=info.duration<=cfg['audio']['max_duration_seconds']:raise ValueError('Invalid duration')
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            if digest in seen:
                if seen[digest]['label']!=r['label']:raise ValueError('Conflicting labels for identical audio')
                continue
            if r['group'] in group_splits and group_splits[r['group']]!=r['split']:raise ValueError('Source group leaks between partitions')
            group_splits[r['group']]=r['split'];r.update(sha256=digest,seconds=round(info.duration,5));seen[digest]=r;clean.append(r)
        except Exception as e:errors.append(f'{r["path"]}: {e}')
    output.mkdir(parents=True,exist_ok=True);write_json(output/'preparation_errors.json',errors)
    if errors:raise ValueError(f'{len(errors)} errors; see {output/"preparation_errors.json"}. No incomplete manifest written.')
    for task in ['emotion','environment']:
        expected=set(maps[task+'_labels'])
        for split in ['base_train','base_val','test']:
            found={r['label'] for r in clean if r['task']==task and r['split']==split}
            if expected-found:raise ValueError(f'Missing {task}/{split} labels: {sorted(expected-found)}')
    with (output/'manifest.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(clean[0]));writer.writeheader();writer.writerows(clean)
    counts=Counter((r['task'],r['split'],r['label']) for r in clean)
    write_json(output/'balance_report.json',{'unique_files':len(clean),'duplicates_removed':len(rows)-len(clean),'counts':{'/'.join(k):v for k,v in sorted(counts.items())},'training_balance':'Equal class exposure per epoch; held-out data keeps natural frequencies.'})
    print('Saved',len(clean),'files to',output/'manifest.csv')
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-root',type=Path,default=ROOT/'data/raw');p.add_argument('--output',type=Path,default=ROOT/'data/processed');a=p.parse_args();prepare(a.data_root,a.output,load_config())
if __name__=='__main__':main()
