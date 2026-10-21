import re
from difflib import SequenceMatcher
from .settings import load_yaml

def tokens(text):return re.findall(r"[a-z]+(?:'[a-z]+)?",text.lower())
class KeywordMatcher:
    def __init__(self,config=None):self.cfg=config or load_yaml('distress_keywords.yaml')
    def match(self,words):
        ws=[{**w,'token':t} for w in words for t in tokens(w['word'])];ts=[w['token'] for w in ws];blocked=set();used=set();hits=[]
        for phrase in self.cfg.get('suppress_phrases',[]):
            p=tokens(phrase)
            for i in range(len(ts)-len(p)+1):
                if ts[i:i+len(p)]==p:blocked.update(range(i,i+len(p)))
        for entry in sorted(self.cfg['phrases'],key=lambda e:len(tokens(e['text'])),reverse=True):
            phrase=tokens(entry['text']);n=len(phrase)
            for i in range(len(ts)-n+1):
                inds=set(range(i,i+n));part=ws[i:i+n]
                if inds&(blocked|used) or min(w.get('probability',1.) for w in part)<self.cfg['min_word_probability']:continue
                if any(part[j+1]['start']-part[j]['end']>self.cfg['max_gap_seconds'] for j in range(n-1)):continue
                actual=' '.join(ts[i:i+n]);target=' '.join(phrase);exact=actual==target;score=SequenceMatcher(None,actual,target).ratio()
                if not exact and (n==1 or score*100<self.cfg['fuzzy_threshold']):continue
                hits.append({'phrase':entry['text'],'text':actual,'start':float(part[0]['start']),'end':float(part[-1]['end']),'score':float(entry['weight']*score),'match':'exact' if exact else 'fuzzy'});used|=inds
        return sorted(hits,key=lambda h:h['start'])
class Transcriber:
    def __init__(self,cfg):
        import onnxruntime as ort
        ort.disable_telemetry_events()
        from faster_whisper import WhisperModel
        import torch
        d=cfg['model']['whisper_device']
        if d=='auto':d='cuda' if torch.cuda.is_available() else 'cpu'
        self.cfg=cfg;self.matcher=KeywordMatcher();self.model=WhisperModel(cfg['model']['whisper_size'],device=d,compute_type='float16' if d=='cuda' else 'int8',cpu_threads=cfg['model']['torch_threads'],local_files_only=cfg['model']['local_files_only'])
    def predict(self,y,valid):
        segments,info=self.model.transcribe(y[:valid],language=self.cfg['model']['whisper_language'],word_timestamps=True,vad_filter=True,condition_on_previous_text=False,beam_size=1,temperature=0.)
        words=[]
        for seg in segments:
            if seg.no_speech_prob>.6:continue
            for w in seg.words or []:
                start=max(0.,w.start);end=min(valid/16000,w.end)
                if end>start:words.append({'word':w.word.strip(),'start':start,'end':end,'probability':float(w.probability)})
        return words,self.matcher.match(words)
