"""Optional downloader; original dataset licenses and Kaggle access apply."""
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import argparse
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile
import requests
from tqdm import tqdm
from src.settings import ROOT
SOURCES={'ravdess':'uwrfkaggler/ravdess-emotional-speech-audio','crema':'ejlok1/cremad','tess':'ejlok1/toronto-emotional-speech-set-tess','savee':'ejlok1/surrey-audiovisual-expressed-emotion-savee','urban':'chrisfilo/urbansound8k'}
def unpack(archive,dest):
    dest=dest.resolve();dest.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for item in z.infolist():
            if not (dest/item.filename).resolve().is_relative_to(dest):raise ValueError('Unsafe archive path.')
        z.extractall(dest)
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',choices=['all','esc50']+list(SOURCES),default='all');p.add_argument('--accept-licenses',action='store_true');p.add_argument('--root',type=Path,default=ROOT/'data/raw');args=p.parse_args()
    if not args.accept_licenses:p.error('Read data/README.md and source licenses, then pass --accept-licenses.')
    for name in list(SOURCES)+['esc50'] if args.dataset=='all' else [args.dataset]:
        dest=args.root/name;dest.mkdir(parents=True,exist_ok=True)
        if (dest/'.download_complete').exists():print(name,'already downloaded');continue
        if name=='esc50':
            with tempfile.TemporaryDirectory() as td:
                archive=Path(td)/'esc.zip'
                with requests.get('https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip',stream=True,timeout=(30,120)) as r:
                    r.raise_for_status()
                    with archive.open('wb') as f,tqdm(total=int(r.headers.get('content-length',0)),unit='B',unit_scale=True) as bar:
                        for chunk in r.iter_content(1024*1024):f.write(chunk);bar.update(len(chunk))
                unpack(archive,dest)
        else:
            try:subprocess.run([sys.executable,'-m','kaggle','datasets','download','-d',SOURCES[name],'-p',str(dest)],check=True)
            except subprocess.CalledProcessError as e:raise SystemExit('Kaggle download failed. Configure your own API token or download manually using data/README.md.') from e
            archives=list(dest.glob('*.zip'))
            if not archives:raise ValueError('No downloaded archive found.')
            for archive in archives:unpack(archive,dest);archive.unlink()
        (dest/'.download_complete').write_text('Downloaded; original licenses apply.',encoding='utf-8')
    print('Next: python -m scripts.prepare_data')
if __name__=='__main__':main()
