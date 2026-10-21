from .settings import device_for,signature,model_dir,load_yaml
from .features import ASTFeatures

def make_head(classes):
    import torch.nn as nn
    return nn.Sequential(nn.LayerNorm(768),nn.Dropout(.2),nn.Linear(768,classes))

class EnvironmentModel:
    def __init__(self,cfg):
        import torch
        self.cfg=cfg;self.device=device_for(cfg);path=model_dir(cfg)/'environment.pt'
        if not path.exists():raise FileNotFoundError('Run python -m scripts.train_environment')
        ck=torch.load(path,map_location='cpu',weights_only=True);self.labels=ck['labels'];self.fast=ck.get('fast',False)
        if ck['signature']!=signature(cfg) or self.labels!=load_yaml('label_maps.yaml')['environment_labels']:raise ValueError('Environment preprocessing/labels changed; retrain.')
        self.head=make_head(len(self.labels)).to(self.device);self.head.load_state_dict(ck['state_dict']);self.head.eval();self.features=ASTFeatures(cfg)
    def predict(self,y,valid):
        import torch
        emb,native,spans=self.features(y,valid)
        with torch.inference_mode():
            probs=self.head(torch.tensor(emb,device=self.device)).softmax(-1).cpu().numpy()
            p=self.head(torch.tensor(emb.mean(0),device=self.device).unsqueeze(0)).softmax(-1)[0].cpu().numpy()
        overall=dict(zip(self.labels,map(float,p)));events=[]
        relevant=set(self.cfg['fusion']['strong_events']+self.cfg['fusion']['contextual_events'])
        scream=[int(k) for k,v in self.features.labels.items() if v.lower()=='screaming']
        for i,(start,end) in enumerate(spans):
            for label in relevant.intersection(self.labels):events.append({'label':label,'probability':float(probs[i,self.labels.index(label)]),'start':start,'end':end,'source':'trained_environment'})
            if scream:events.append({'label':'scream','probability':float(native[i,scream].max()),'start':start,'end':end,'source':'pretrained_audioset_unvalidated'})
        return overall,events
