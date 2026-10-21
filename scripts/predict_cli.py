if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import argparse
import json
from pathlib import Path
import sys
from src.pipeline import Pipeline,ModelsUnavailable

def main():
    p=argparse.ArgumentParser();p.add_argument('audio',type=Path);p.add_argument('--output',type=Path);args=p.parse_args()
    try:
        result=Pipeline().analyze_file(args.audio);text=json.dumps(result,indent=2)
        if args.output:args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(text,encoding='utf-8')
        else:print(text)
    except (ModelsUnavailable,ValueError,FileNotFoundError) as exc:print(str(exc),file=sys.stderr);return 2
    return 0
if __name__=='__main__':raise SystemExit(main())
