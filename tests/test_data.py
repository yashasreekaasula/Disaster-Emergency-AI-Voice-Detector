import csv
import numpy as np
import soundfile as sf
from scripts.prepare_data import speaker_partitions,prepare
from src.data import PARTITIONS,read_manifest
from src.settings import load_yaml

def test_speaker_partitions():
    r=speaker_partitions([str(i) for i in range(24)],42);assert len(r)==24 and set(r.values())==set(PARTITIONS) and r==speaker_partitions([str(i) for i in range(24)],42)
def test_tess_no_leakage():assert set(speaker_partitions(['OAF','YAF'],42).values())=={'base_train','test'}
def test_six_dataset_parsers(tmp_path,cfg):
    maps=load_yaml('label_maps.yaml');raw=tmp_path/'raw';out=tmp_path/'processed';counter=0
    def audio(path):
        nonlocal counter
        counter+=1;path.parent.mkdir(parents=True,exist_ok=True);sf.write(path,np.random.default_rng(counter).normal(0,.1,800).astype('float32'),16000)
    for actor in range(1,25):
        for code in maps['ravdess']:audio(raw/'ravdess'/f'03-01-{code}-01-01-01-{actor:02}.wav')
    for actor in range(1001,1011):
        for code in maps['crema']:audio(raw/'crema'/f'{actor}_DFA_{code}_XX.wav')
    for actor in ['OAF','YAF']:
        for code in maps['tess']:audio(raw/'tess'/f'{actor}_back_{code}.wav')
    for actor in ['DC','JE','JK','KL']:
        for code in maps['savee']:audio(raw/'savee'/f'{actor}_{code}01.wav')
    for ds in ['esc50','urban']:
        meta=[];base=raw/ds
        for fold in range(1,6 if ds=='esc50' else 11):
            for i,label in enumerate(maps[ds]):
                name=f'{fold}-{i}.wav'
                if ds=='esc50':audio(base/'audio'/name);meta.append({'filename':name,'fold':fold,'category':label,'src_file':f'{fold}-{i}'})
                else:audio(base/'audio'/f'fold{fold}'/name);meta.append({'slice_file_name':name,'fold':fold,'class':label,'fsID':f'{fold}-{i}'})
        path=base/('meta/esc50.csv' if ds=='esc50' else 'metadata/UrbanSound8K.csv');path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('w',newline='') as f:writer=csv.DictWriter(f,fieldnames=list(meta[0]));writer.writeheader();writer.writerows(meta)
    prepare(raw,out,cfg);rows=read_manifest(out/'manifest.csv');groups={}
    for r in rows:groups.setdefault(r['group'],set()).add(r['split'])
    assert len(rows)==counter and all(len(s)==1 for s in groups.values())
    for task in ['emotion','environment']:
        for split in ['base_train','base_val','test']:assert {r['label'] for r in rows if r['task']==task and r['split']==split}==set(maps[task+'_labels'])
