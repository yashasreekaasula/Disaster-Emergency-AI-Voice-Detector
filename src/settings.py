from pathlib import Path
import hashlib
import json
import os
import yaml
ROOT=Path(__file__).resolve().parents[1]
def load_yaml(name):
    with (ROOT/'config'/name).open(encoding='utf-8') as f:return yaml.safe_load(f)
def load_config(path=None):
    with Path(path or os.environ.get('SOUNDSENTINEL_CONFIG',ROOT/'config/config.yaml')).open(encoding='utf-8') as f:cfg=yaml.safe_load(f)
    a=cfg['audio']
    if a['sample_rate']!=16000 or a['window_seconds']!=15:raise ValueError('Use 16000 Hz and 15-second windows for this checkpoint format.')
    if not 0<a['hop_seconds']<=15 or a['event_seconds']<=0:raise ValueError('Invalid hop or event length.')
    return cfg
def model_dir(cfg):
    p=Path(cfg['model']['models_dir']);return p if p.is_absolute() else ROOT/p
def signature(cfg):
    return hashlib.sha256(json.dumps({'sr':16000,'window':15,'event':cfg['audio']['event_seconds'],'mel':'64-400-160-v1','normalization':'peak-v1','ast':cfg['model']['ast_id']},sort_keys=True).encode()).hexdigest()
def device_for(cfg):
    import torch
    torch.set_num_threads(cfg['model']['torch_threads']);d=cfg['model']['device']
    if d=='auto':return 'cuda' if torch.cuda.is_available() else 'cpu'
    if d=='cuda' and not torch.cuda.is_available():raise ValueError('CUDA unavailable. Set model.device: cpu.')
    return d
