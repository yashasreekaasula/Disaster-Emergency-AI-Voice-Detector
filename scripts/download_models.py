if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from src.settings import load_config

def main():
    import onnxruntime as ort
    ort.disable_telemetry_events()
    from transformers import AutoFeatureExtractor,ASTForAudioClassification
    from faster_whisper import WhisperModel
    cfg=load_config();name=cfg['model']['ast_id'];print('Downloading AST:',name);AutoFeatureExtractor.from_pretrained(name);ASTForAudioClassification.from_pretrained(name);print('Downloading Whisper:',cfg['model']['whisper_size']);WhisperModel(cfg['model']['whisper_size'],device='cpu',compute_type='int8');print('Backbones cached. Train the project heads next.')
if __name__=='__main__':main()
