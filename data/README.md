# Dataset sources and layout

No datasets or credentials are included. Download the complete audio datasets and observe each original source's license. Mirrors do not remove original restrictions.

| Dataset | Original source | Optional downloader mirror |
|---|---|---|
| RAVDESS | https://zenodo.org/records/1188976 — choose Audio_Speech_Actors_01-24.zip | https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio |
| CREMA-D | https://github.com/CheyneyComputerScience/CREMA-D — AudioWAV requires Git LFS; repository ZIPs contain pointers | https://www.kaggle.com/datasets/ejlok1/cremad |
| TESS | https://utoronto.scholaris.ca/collections/036db644-9790-4ed0-90cc-be1dfb8a4b66 | https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess |
| SAVEE | http://kahlan.eps.surrey.ac.uk/savee/Download.html — original registration may be required | https://www.kaggle.com/datasets/ejlok1/surrey-audiovisual-expressed-emotion-savee |
| ESC-50 | https://github.com/karolpiczak/ESC-50 | Original GitHub ZIP |
| UrbanSound8K | https://urbansounddataset.weebly.com/urbansound8k.html | https://www.kaggle.com/datasets/chrisfilo/urbansound8k |

Read the original terms: several datasets restrict commercial usage. Do not redistribute recordings without permission.

## Download

Optional automated download requires your own Kaggle API token at `~/.kaggle/kaggle.json` (Windows: `%USERPROFILE%\.kaggle\kaggle.json`). Create the token in your Kaggle account settings; accept dataset terms in your browser where required. Never put credentials inside this project or a submission ZIP.

```sh
python -m scripts.download_data --accept-licenses
python -m scripts.prepare_data
```

The downloader resumes by skipping dataset directories containing `.download_complete`. Network/account/mirror access can fail; the original sources above provide a manual alternative.

## Manual arrangement

Keep original filenames and metadata. Nested directories are discovered recursively:

```text
data/raw/
  ravdess/Actor_01/03-01-01-01-01-01-01.wav
  crema/AudioWAV/1001_DFA_ANG_XX.wav
  tess/OAF_angry/OAF_back_angry.wav
  savee/DC_a01.wav
  esc50/ESC-50-master/meta/esc50.csv
  esc50/ESC-50-master/audio/1-100032-A-0.wav
  urban/UrbanSound8K/metadata/UrbanSound8K.csv
  urban/UrbanSound8K/audio/fold1/101415-3-0-2.wav
```

These names illustrate the layout; include all files, not only these examples. SAVEE also supports `savee/DC/a01.wav`. RAVDESS processing accepts audio-only speech (`03-01`) and excludes song/video. Metadata and audio for ESC-50/UrbanSound8K must preserve their original relative layout.

Preparation checks file readability, durations, required classes, duplicate bytes and source-group leakage. On failure read `data/processed/preparation_errors.json`; no incomplete manifest is written. Paths in the manifest are absolute; rerun preparation after moving raw data and retrain to preserve provenance.

## Label mapping and balance

`config/label_maps.yaml` gives every mapping. Common emotion labels: angry, disgust, fear, happy, neutral, sad, surprise, calm. Calm is supported by RAVDESS; other datasets are not assigned invented calm labels. TESS pleasant surprise (`ps`) maps to surprise.

The environment model retains all 50 ESC-50 categories and merges UrbanSound8K to a total of 56 labels. `dog` becomes `dog_bark`, `door_wood_knock` becomes `door_knock`, `engine_idling` becomes `engine`, `gun_shot` becomes `gunshot`. Neutral is an emotion; Normal is a separate fusion verdict.

Scream is not an ESC-50/UrbanSound8K training class. It is an explicitly tagged, unvalidated signal from AST's pretrained AudioSet Screaming category.

Split before balancing. RAVDESS/CREMA actors are assigned approximately 60/10/10/10/10 to base_train/base_val/fusion_train/fusion_val/test. SAVEE has only four speakers: two base train, one base validation, one test. TESS has only two: one base train, one test. Thus no false claim of five independent partitions for two speakers.

ESC-50 folds 1–2 train the base model, 3 validates, 4 is reserved for fusion (split by source), 5 tests. UrbanSound8K folds 1–6 train, 7 validates, 8 trains fusion, 9 validates fusion, 10 tests. This is a fixed holdout experiment, not official all-fold cross-validation. Duplicate byte hashes are removed and source groups cannot cross partitions; differently encoded duplicate recordings may require further review.

Each emotion/environment epoch resamples training only for exactly equal exposure per class. Fusion training downsamples the majority for exactly 50% Normal / 50% Distress. Validation/test frequencies remain natural; fusion also reports a balanced held-out subset. See the generated balance reports and printed counts.

## Human-reviewed fusion data

For better labels, provide a CSV with `path,distress,group,split`. Distress is 0/1; split is fusion_train/fusion_val/test; group identifies a speaker/session/source that must stay in one split. Paths are absolute or relative to the CSV. Label all windows in each clip consistently. Include ordinary conversations, laughter, television, fire-drill discussions and loud but benign sounds.

```sh
python -m scripts.build_fusion_dataset --custom-labels data/my_reviewed_labels.csv
python -m scripts.train_fusion
```

Custom files must be new (not byte-identical to base datasets). The builder validates groups and records whether a row is human-labeled or a weak proxy. Real-world accuracy requires a separate, reviewed and source-independent test set.
