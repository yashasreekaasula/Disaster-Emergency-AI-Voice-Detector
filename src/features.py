import numpy as np
from .settings import device_for
from .audio_utils import event_chunks

def log_mel(y,cfg):
    import librosa
    mel=librosa.feature.melspectrogram(y=y,sr=16000,n_fft=400,hop_length=160,n_mels=64,fmin=50,fmax=8000,power=2.)
    return ((librosa.power_to_db(mel,ref=np.max,top_db=80)+40)/40).astype(np.float32)[None]

class ASTFeatures:
    def __init__(self,cfg):
        from transformers import AutoFeatureExtractor,ASTForAudioClassification
        self.cfg=cfg;self.device=device_for(cfg);opts={'local_files_only':cfg['model']['local_files_only']};name=cfg['model']['ast_id']
        self.extractor=AutoFeatureExtractor.from_pretrained(name,**opts)
        self.model=ASTForAudioClassification.from_pretrained(name,**opts).to(self.device).eval();self.labels=self.model.config.id2label
        for p in self.model.parameters():p.requires_grad_(False)
    def __call__(self,y,valid):
        import torch
        emb=[];scores=[];spans=[]
        with torch.inference_mode():
            for start,end,chunk in event_chunks(y,valid,self.cfg):
                x=self.extractor(chunk,sampling_rate=16000,return_tensors='pt');x={k:v.to(self.device) for k,v in x.items()}
                h=self.model.audio_spectrogram_transformer(**x).pooler_output
                emb.append(h[0].cpu().numpy());scores.append(self.model.classifier(h).sigmoid()[0].cpu().numpy());spans.append((start,end))
        return np.stack(emb),np.stack(scores),spans
