if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import importlib
import platform
import shutil
from src.settings import ROOT
from src.pipeline import Pipeline

def main():
    print('Project:',ROOT,'Python:',platform.python_version());bad=[]
    for name in ['numpy','scipy','pandas','torch','torchaudio','librosa','transformers','faster_whisper','sklearn','fastapi','uvicorn','soundfile','yaml','multipart']:
        try:m=importlib.import_module(name);print(name,getattr(m,'__version__','OK'))
        except Exception as e:print(name,'FAILED:',e);bad.append(name)
    print('ffmpeg:',shutil.which('ffmpeg') or 'MISSING');print('Models:',Pipeline().status());return int(bool(bad))
if __name__=='__main__':raise SystemExit(main())
