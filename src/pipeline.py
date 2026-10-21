import logging
import threading
import numpy as np
from .settings import load_config,model_dir
from .audio_utils import read_audio,fixed_window,windows
from .fusion import Fusion
from .schemas import Analysis
log=logging.getLogger(__name__)
class ModelsUnavailable(RuntimeError):
    """Required local checkpoint is missing or incompatible."""
class Pipeline:
    def __init__(self,cfg=None,emotion=None,environment=None,transcriber=None,use_trained_fusion=True):
        self.cfg=cfg or load_config();self.emotion=emotion;self.environment=environment;self.transcriber=transcriber;self.use_trained_fusion=use_trained_fusion;self.fusion=None;self.initialized=False;self.errors=[];self.warnings=[];self.lock=threading.RLock()
    def status(self):
        missing=[f'python -m scripts.train_{k}' for k in ('emotion','environment') if getattr(self,k) is None and not (model_dir(self.cfg)/f'{k}.pt').exists()]
        return {'ready':not missing and not self.errors,'loaded':self.initialized,'missing_commands':missing,'errors':self.errors,'warnings':self.warnings,'window_seconds':15,'hop_seconds':self.cfg['audio']['hop_seconds'],'timeline_limit':self.cfg['server']['timeline_limit']}
    def load(self):
        with self.lock:
            if self.initialized:return
            missing=self.status()['missing_commands']
            if missing:raise ModelsUnavailable('Models not trained yet. Run: '+'; '.join(missing))
            try:
                from .emotion import EmotionModel
                from .environment import EnvironmentModel
                from .keywords import Transcriber
                self.emotion=self.emotion or EmotionModel(self.cfg);self.environment=self.environment or EnvironmentModel(self.cfg);self.fusion=Fusion(self.cfg,self.use_trained_fusion)
                if self.transcriber is None:
                    try:self.transcriber=Transcriber(self.cfg)
                    except Exception as exc:log.exception('Transcription unavailable');self.warnings.append('Transcription unavailable. Run python -m scripts.download_models and restart. '+str(exc))
                if not self.fusion.trained:self.warnings.append('Using conservative fusion rules. Build and train fusion for learned scores.')
                if getattr(self.emotion,'fast',False) or getattr(self.environment,'fast',False):self.warnings.append('FAST smoke-test checkpoints: predictions are unreliable. Run full training.')
                self.initialized=True;self.errors.clear()
            except Exception as exc:
                log.exception('Model loading failed');self.errors=[str(exc)];raise ModelsUnavailable('Model loading failed: '+str(exc)) from exc
    def analyze_window(self,samples,offset=0.):
        with self.lock:
            self.load();y,valid=fixed_window(samples,self.cfg);end=offset+valid/16000;warnings=list(self.warnings)
            if np.sqrt(np.mean(np.asarray(samples[:valid],dtype=np.float32)**2))<self.cfg['audio']['silence_rms']:
                return Analysis(verdict='No signal',confidence=None,distress_score=0.,emotion={},environment={},events=[],words=[],keywords=[],transcript='',explanation=[],warnings=['Audio is silent or too quiet.'],start=offset,end=end,method='silence_gate').model_dump()
            emotion=self.emotion.predict(y,valid);environment,events=self.environment.predict(y,valid);words=[];hits=[]
            if self.transcriber:
                try:words,hits=self.transcriber.predict(y,valid)
                except Exception:log.exception('Transcription failed');warnings.append('Transcription failed; keyword evidence is incomplete.')
            result=self.fusion.predict(emotion,environment,hits,events)
            for key in ('explanation','observed_signals'):
                result[key]=[{**r,'start':r['start']+offset if r.get('start') is not None else offset,'end':r['end']+offset if r.get('end') is not None else end} for r in result[key]]
            def shift(items):return [{**i,'start':min(end,i['start']+offset),'end':min(end,i['end']+offset)} for i in items]
            return Analysis(**result,emotion=emotion,environment=environment,events=shift(events),words=shift(words),keywords=shift(hits),transcript=' '.join(w['word'] for w in words),warnings=warnings,start=offset,end=end).model_dump()
    def analyze_file(self,path):
        y=read_audio(path,self.cfg);results=[self.analyze_window(x,t) for t,x in windows(y,self.cfg)];active=[r for r in results if r['verdict']!='No signal'];summary=max(active or results,key=lambda r:r['distress_score'])
        return {'summary':summary,'windows':results,'duration':len(y)/16000,'aggregation':'highest distress-score window; absolute timestamps'}
