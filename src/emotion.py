from .settings import device_for,signature,model_dir,load_yaml
from .features import log_mel

def make_cnn(classes):
    import torch.nn as nn
    return nn.Sequential(nn.Conv2d(1,16,5,2,2),nn.BatchNorm2d(16),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(16,32,3,2,1),nn.BatchNorm2d(32),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(32,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(),nn.AdaptiveAvgPool2d((1,1)),nn.Flatten(),nn.Dropout(.25),nn.Linear(64,classes))

class EmotionModel:
    def __init__(self,cfg):
        import torch
        self.cfg=cfg;self.device=device_for(cfg);path=model_dir(cfg)/'emotion.pt'
        if not path.exists():raise FileNotFoundError('Run python -m scripts.train_emotion')
        ck=torch.load(path,map_location='cpu',weights_only=True);self.labels=ck['labels'];self.fast=ck.get('fast',False)
        if ck['signature']!=signature(cfg) or self.labels!=load_yaml('label_maps.yaml')['emotion_labels']:raise ValueError('Emotion preprocessing/labels changed; retrain.')
        self.model=make_cnn(len(self.labels)).to(self.device);self.model.load_state_dict(ck['state_dict']);self.model.eval()
    def predict(self,y,valid):
        import torch
        x=torch.from_numpy(log_mel(y,self.cfg)).unsqueeze(0).to(self.device)
        with torch.inference_mode():p=self.model(x).softmax(-1)[0].cpu().numpy()
        return dict(zip(self.labels,map(float,p)))
