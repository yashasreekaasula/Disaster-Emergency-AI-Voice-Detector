from pathlib import Path
import math
import shutil
import subprocess
import numpy as np
from scipy.signal import resample_poly

def validate_audio(y,sr):
    y=np.asarray(y,dtype=np.float32)
    if y.ndim==2:y=y.mean(axis=1)
    if y.ndim!=1 or not y.size or not np.isfinite(y).all():raise ValueError('Empty or malformed audio samples.')
    if not isinstance(sr,(int,np.integer)) or not 1000<=sr<=384000:raise ValueError('Invalid sample rate.')
    return y

def resample(y,sr,target=16000):
    y=validate_audio(y,sr)
    if sr!=target:
        g=math.gcd(sr,target);y=resample_poly(y,target//g,sr//g).astype(np.float32)
    return y

def fixed_window(y,cfg):
    y=validate_audio(y,16000);n=int(16000*cfg['audio']['window_seconds']);valid=min(n,len(y))
    out=np.zeros(n,dtype=np.float32);part=y[:valid];out[:valid]=part/max(float(np.abs(part).max()),.01)
    return out,valid

def read_audio(path,cfg):
    path=Path(path);a=cfg['audio'];limit=a['max_duration_seconds']
    if not path.is_file():raise ValueError(f'Audio not found: {path}')
    if path.suffix.lower().lstrip('.') not in a['extensions']:raise ValueError('Unsupported audio extension.')
    if path.stat().st_size>a['max_upload_mb']*1024**2:raise ValueError('File exceeds configured size limit.')
    if shutil.which('ffmpeg'):
        cmd=['ffmpeg','-v','error','-nostdin','-i',str(path),'-t',str(limit+1),'-vn','-ac','1','-ar','16000','-f','f32le','pipe:1']
        try:r=subprocess.run(cmd,capture_output=True,check=True,timeout=a['decode_timeout_seconds'])
        except subprocess.TimeoutExpired as e:raise ValueError('Audio decoding timed out.') from e
        except subprocess.CalledProcessError as e:raise ValueError('Could not decode audio; file may be damaged.') from e
        y=np.frombuffer(r.stdout,dtype='<f4').copy()
    else:
        import soundfile as sf
        try:
            info=sf.info(path)
            if info.duration>limit:raise ValueError('Audio exceeds duration limit.')
            y,sr=sf.read(path,dtype='float32',always_2d=True);y=resample(y,sr)
        except (sf.LibsndfileError,RuntimeError) as e:raise ValueError('Install ffmpeg and restart the terminal to decode this format.') from e
    y=validate_audio(y,16000)
    if not a['min_seconds']<=len(y)/16000<=limit:raise ValueError(f'Audio must last {a["min_seconds"]} to {limit} seconds.')
    return y

def windows(y,cfg):
    n=240000;hop=int(16000*cfg['audio']['hop_seconds'])
    starts=list(range(0,max(1,len(y)-n+1),hop))
    if len(y)>n and starts[-1]+n<len(y):starts.append(len(y)-n)
    for start in starts:yield start/16000,y[start:start+n]

def event_chunks(y,valid,cfg):
    n=int(16000*cfg['audio']['event_seconds'])
    for start in range(0,valid,n):
        end=min(valid,start+n);part=np.zeros(n,dtype=np.float32);part[:end-start]=y[start:end]
        yield start/16000,end/16000,part
