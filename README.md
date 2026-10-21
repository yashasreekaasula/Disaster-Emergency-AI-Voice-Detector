# SoundSentinel: Distress Detection with Emotion and Environment Recognition

A local FastAPI application with audio upload, browser microphone streaming, emotion/environment training, timestamped keyword detection and explainable fusion. Plain HTML/CSS/JS frontend; no build step.

**This ZIP contains complete source code, tests and documentation. It does not contain datasets or project-trained weights.** Download data/backbones and train before real recognition. Without checkpoints the server starts and clearly names the training commands. It never fabricates predictions. Fast checkpoints are smoke tests, not accurate recognition models.

**A balanced training set cannot guarantee unbiased or error-free predictions.** This project enforces equal training exposure, measures Normal false positives, and documents label limitations. No honest project can guarantee zero runtime errors on every computer or perfect emergency classification.

## Features

- Upload WAV, MP3, OGG, FLAC, M4A or WEBM; default limits 40 MB and 300 seconds.
- Browser AudioWorklet captures mono audio, low-pass filters/resamples to 16 kHz and sends signed PCM16 over WebSocket.
- **15-second windows with 1-second hops.** First live result after 15 seconds plus inference time.
- Distress/Normal badge with uncalibrated decision confidence; separate No signal state for silence.
- Eight emotion probabilities, top-3 environment classes, keyword chips, highlighted transcript with word timestamps, acoustic trigger spans, level meter and result timeline.
- CPU execution; PyTorch selects CUDA when available. Whisper defaults to portable CPU/int8.
- Temporary upload cleanup, size/duration limits, logging, validation, lazy model loading and bounded live queues.

## Models and preprocessing

1. Decode with ffmpeg, bound decoded duration, convert to 16 kHz mono. Normalize the window by its peak with a 0.01 floor to avoid boosting near-silence. Pad to 15 seconds. Training and inference share the exact preprocessing/feature functions.
2. **Emotion:** a compact CNN trained on 64-band log-mel spectrograms, using RAVDESS + CREMA-D + TESS + SAVEE. FFT=400, hop=160 samples. This is the CNN option in the brief, trained from scratch; it is not falsely presented as a pretrained emotion model. Mild time masking is training-only augmentation.
3. **Environment:** frozen pretrained AudioSet AST encoder plus a trained LayerNorm/linear head on ESC-50 + UrbanSound8K. Up to three five-second subwindows preserve brief-event evidence; padded-only chunks are excluded. Mean embeddings give the window classification, and segment head outputs give coarse event candidates. AST's feature extractor pads each subwindow to its pretrained spectrogram shape. Scream uses a separately tagged pretrained AudioSet output because the two requested datasets lack a dedicated scream class.
4. **Keywords:** faster-whisper tiny.en with VAD, word timestamps, CPU int8 and an English configurable lexicon. Match whole words and multiword phrases; fuzzy matching is limited to phrases. Suppress configured benign contexts such as help desk/fire drill/stop sign. This is not complete language understanding.
5. **Fusion:** standardized logistic regression trained on base-model predictions from reserved speakers/recordings. Equal numbers of Normal and Distress rows. Tune regularization and threshold on fusion validation only. Strong keyword/acoustic evidence has an explicit rule override. A configurable evidence gate prevents anger, sadness or a siren alone from automatically causing Distress. Conservative rules are used if only the fusion model is absent.
6. Upload summary selects the highest distress-score window. The shown transcript belongs to that window; select timeline entries to inspect other times. Overlapping evidence is not treated as independent votes. A longer recording offers more false-alarm opportunities.

Emotion labels: angry, disgust, fear, happy, neutral, sad, surprise, calm. Environment: 56 merged classes. Full mappings are in `config/label_maps.yaml`. Acoustic trigger spans are coarse five-second estimates, not precisely annotated sound onset times. Emotion scores on non-speech audio are not reliable.

## Setup and exact commands

Use **Python 3.11** (3.10 also targeted). Open a terminal in the extracted `soundsentinel` folder containing README.md. A 16 GB RAM laptop is recommended. Allow 20–30 GB or more for datasets, features, dependencies and pretrained backbones. Full feature extraction/training can take hours.

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
$py = ".\.venv\Scripts\python.exe"
& $py -m pip install --upgrade pip
& $py -m pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu
& $py -m pip install -r requirements.txt
winget install --id Gyan.FFmpeg --exact --accept-package-agreements --accept-source-agreements
```

Close/reopen PowerShell after ffmpeg installation, return to the project folder, and continue. The explicit interpreter avoids activation-policy issues.

```powershell
$py = ".\.venv\Scripts\python.exe"
ffmpeg -version
& $py -m scripts.check
# Configure your own Kaggle token as described in data/README.md, or download manually.
& $py -m scripts.download_data --accept-licenses
& $py -m scripts.prepare_data
& $py -m scripts.download_models
# Fast end-to-end verification:
& $py -m scripts.train_emotion --fast
& $py -m scripts.train_environment --fast
& $py -m scripts.build_fusion_dataset --fast
& $py -m scripts.train_fusion --fast
& $py -m scripts.evaluate --fast
# Full training replaces fast checkpoints:
& $py -m scripts.train_emotion
& $py -m scripts.train_environment
& $py -m scripts.build_fusion_dataset
& $py -m scripts.train_fusion
& $py -m scripts.evaluate
& $py -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --ws-max-size 65536
```

Open **http://localhost:8000**. Stop with Ctrl+C before continuing in the same terminal. Put your test recording at `sample.wav`:

```powershell
& $py -m scripts.predict_cli sample.wav --output reports/prediction.json
& $py -m pytest
```

### Linux / macOS

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Linux CPU:
```sh
python -m pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu
```
macOS uses the standard wheels instead:
```sh
python -m pip install torch==2.5.1 torchaudio==2.5.1
```
Then:
```sh
python -m pip install -r requirements.txt
# Ubuntu/Debian:
sudo apt-get update
sudo apt-get install -y ffmpeg libsndfile1
# macOS: use this instead of apt:
# brew install ffmpeg libsndfile
ffmpeg -version
python -m scripts.check
# Set up your Kaggle token, or manually arrange all datasets using data/README.md.
python -m scripts.download_data --accept-licenses
python -m scripts.prepare_data
python -m scripts.download_models
python -m scripts.train_emotion --fast
python -m scripts.train_environment --fast
python -m scripts.build_fusion_dataset --fast
python -m scripts.train_fusion --fast
python -m scripts.evaluate --fast
# Replace smoke-test models with full training:
python -m scripts.train_emotion
python -m scripts.train_environment
python -m scripts.build_fusion_dataset
python -m scripts.train_fusion
python -m scripts.evaluate
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --ws-max-size 65536
# Open http://localhost:8000; Ctrl+C before the following commands.
python -m scripts.predict_cli sample.wav --output reports/prediction.json
python -m pytest
```

For a compatible NVIDIA driver on Windows/Linux, substitute the paired GPU wheels before installing requirements:
```sh
python -m pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

Keep torch/torchaudio builds paired. Auto mode uses CUDA when available, otherwise CPU. Apple Silicon uses CPU in this project. Older Intel macOS combinations may lack these wheels; use Docker/Linux if pip reports no compatible distribution. Whisper GPU needs compatible CTranslate2 CUDA/cuDNN libraries; CPU/int8 is the supported portable default.

## Balanced labels and honest distress targets

There is no single public dataset in this project with verified labels for the whole distress-detection task. Default fusion labels are **weak proxies**: fearful speech, gunshot, glass breaking and crying baby are positive; other source classes are negative. Angry speech and a siren alone are negative by default. These assumptions are configurable and can be wrong in real life.

Split speakers/source recordings **before balancing**. CNN/environment training gives exactly equal class exposure per epoch through seeded resampling. Do not apply inverse-frequency loss weights on top of this (that would double-correct). Fusion downsamples the majority to an exact 50/50 training balance. Test/validation retain natural frequencies; additional balanced fusion reports are also saved.

Base models use base_train/base_val; fusion uses reserved fusion_train/fusion_val; test remains unseen by fitting and threshold selection. TESS/SAVEE cannot supply all five speaker-independent partitions because they contain too few speakers. See `data/README.md` for precise assignments and official-fold usage.

The conservative evidence gate is intentionally stricter than weak fear/crying/glass labels, so weak-proxy recall may be low. This conflict is measured rather than hidden. Strong phrases are handled transparently by a rule override because these datasets contain few authentic distress phrases. No invented transcripts or fabricated keyword training examples are added.

Fusion chooses a threshold seeking high balanced accuracy while meeting a configurable validation Normal false-positive rate (10% default). If no tested threshold meets it, training prints a warning. This is not a deployment guarantee. Balanced training changes the class prior; scores are **uncalibrated decision scores**, not verified emergency probabilities. Normal means the configured decision policy was not met; it does not certify safety.

Improve the model with consented human-reviewed recordings using `--custom-labels` (documented in data/README.md), including ordinary conversations, laughter, television, loud benign scenes and ambiguous phrases. Evaluate on a separate reviewed test collection. Class balancing cannot fix incorrect labels, acting style, domain shift, unseen microphones, languages or demographic coverage.

## Training and outputs

All training scripts support `--fast`. Neural scripts also support `--epochs N --batch-size N`, progress bars, validation early stopping, best checkpoint saving, printed reports and confusion-matrix PNGs. Fusion uses solver convergence and early termination of a non-improving regularization search. Fast reduces examples/epochs; first-time AST extraction/downloads can still take longer than a few minutes on a slow CPU.

- `models/emotion.pt`, `models/environment.pt`: best checkpoint, labels, preprocessing signature, manifest hash and fast flag.
- `models/fusion.json`: readable coefficients, scaler, feature names, threshold, base-model hashes and class counts.
- `models/*_labels.json`: numerical class mapping.
- `data/processed/manifest.csv` and `balance_report.json`: source groups, partitions, hashes and raw label counts.
- `reports/*_history.json`, `*_confusion.png`, `.json`, `.txt`: loss, accuracy, balanced accuracy, macro F1 and per-class precision/recall/support.
- `reports/fusion_balance.json`: before/after class counts; fusion metrics include Normal false-positive and Distress false-negative rates.
- `data/cache`: deterministic features keyed by audio hash and preprocessing; reads are streamed from disk to avoid loading every spectrogram into RAM.

Retrain fusion after changing either base model. Feature building explicitly ignores an old fusion checkpoint so full retraining after a fast run works. Model signatures detect preprocessing/label changes; fusion hashes detect stale base-model weights. Restart the server after replacing weights. Use one Uvicorn worker to avoid duplicate model copies.

### Estimated times, not full-run benchmarks

| Task | Modern laptop CPU | Mid-range NVIDIA GPU |
|---|---|---|
| Downloads | Network-dependent, tens of minutes or longer | Same |
| Fast emotion | A few minutes | Usually under a few minutes |
| Fast environment, uncached | About 5–30+ minutes | About 1–5+ minutes |
| Full emotion | About 1–4+ hours | About 10–40+ minutes |
| Full AST extraction + environment head | About 3–12+ hours | About 20–90+ minutes |
| Fusion feature building with ASR | About 1–8+ hours | About 20–120+ minutes, Whisper CPU by default |
| Fusion classifier fitting | Seconds to minutes | CPU sufficient |

Actual time depends on hardware, disk, threads, lengths and cache. One-second updates are not guaranteed on CPU. The live server keeps only the newest pending window, reports latency/skipped windows, and never queues unlimited audio. Stop discards an incomplete final window.

## Configuration and API

Edit `config/config.yaml`, `config/distress_keywords.yaml`, `config/label_maps.yaml`. Set `SOUNDSENTINEL_CONFIG` to a different config path for isolated experiments. The fixed model contract is 16 kHz/15 seconds; changing these requires code-contract updates and retraining. Hop, subwindow length, sizes, timeouts, thresholds, threads and devices are configurable.

- `GET /api/health`: readiness and commands for missing models.
- `POST /api/analyze`: multipart field `file`; response includes all windows and selected summary.
- `WS /ws/live`: send `{"sample_rate":16000,"encoding":"pcm_s16le"}`, await ready, send binary mono little-endian int16 chunks, receive result packets; send text `stop` to finish. UI sends 100 ms chunks.
- `GET /docs`: interactive API documentation.

Browser microphone access requires **localhost or HTTPS**. A normal HTTP LAN IP will not work. Audio is processed locally, not sent to an external inference service; optional ONNX Runtime and Hugging Face telemetry are disabled before imports/model initialization. Pretrained downloads require internet once; subsequent inference uses local cache. Temporary uploads are deleted in finally blocks, malformed inputs rejected, and decoding duration capped. Missing models produce readable 503 responses.

### Docker

```sh
docker compose build
docker compose run --rm soundsentinel python -m scripts.download_models
docker compose up
```

Mount data/models as configured. For training in Docker, run preparation inside Docker because manifest paths are absolute; use `docker compose run --rm soundsentinel python -m scripts.prepare_data`, then the training modules. Host and container Hugging Face caches differ. Docker defaults to CPU and still requires trained project weights.

## Troubleshooting

| Problem | Fix |
|---|---|
| Wrong Python/import errors | Use Python 3.11 and the same venv for pip and execution; run scripts.check. |
| PowerShell activation blocked | Use the explicit `$py` commands; activation is unnecessary. |
| No module named src | Run `python -m scripts.NAME` from the project root. |
| ffmpeg not found | Install via winget/brew/apt, reopen the terminal and run ffmpeg -version. |
| Kaggle access denied | Use your own token, accept terms, or manually use original source downloads. |
| CREMA WAV unreadable | GitHub ZIP may contain Git LFS pointers. Obtain actual WAV audio using LFS or the mirror. |
| Dataset layout/labels missing | Read preparation_errors.json and data/README.md; include all required datasets and original metadata. |
| Models not trained | Run prepare_data, download_models, train_emotion and train_environment; then build/train fusion. |
| Missing Hugging Face cache | Run download_models with internet in the same user/environment. |
| CUDA out of memory | Reduce --batch-size to 4, close duplicate servers, or set model.device: cpu. |
| Whisper CUDA error | Keep whisper_device: cpu for portable int8 inference. |
| CPU too slow/live skips | Increase hop_seconds to 3–5, use one server worker, or use compatible GPU hardware. |
| Mic denied/not found | Connect a microphone, allow browser permission and use localhost or HTTPS. |
| Mic blocked on HTTP LAN IP | Use HTTPS or localhost on the server machine. |
| Too many Distress results | Replace fast weights, inspect evidence/FPR, review weak labels and collect ordinary/hard-negative audio. Balance alone is insufficient. |
| Too many Normal results | Inspect false negatives and evidence gate; add reviewed positive recordings before relaxing policy. |
| Wrong language transcription | Change Whisper model/language and translate/review the lexicon. Default is English. |
| No signal | Check microphone/input volume; silence is not assigned confident Normal. |
| Moved data | Re-run prepare_data and retrain to keep provenance consistent. |
| Port occupied | Stop the old server or use --port 8001 and localhost:8001. |

## Viva / Report notes

**Problem statement:** jointly analyze speech and environmental audio to identify distress-related evidence with understandable timestamps in uploaded and live recordings.

**Methodology:** merge labels, split independent speakers/recordings, standardize audio, train balanced CNN/AST heads, transcribe/match keywords, build held-out fusion features, train logistic regression, tune the decision threshold on validation, and inspect untouched test errors.

**Model choices:** CNN gives a compact emotion baseline; pretrained AST transfers broad audio knowledge; faster-whisper offers local word timestamps; logistic regression is transparent and CPU-friendly. Explicit fallback/override rules remain identifiable.

**Metrics:** accuracy, balanced accuracy, macro F1, per-class recall/precision, confusion matrix, Normal false-positive rate, Distress false-negative rate, decision distribution and processing latency. Distinguish proxy-label evaluation from human-reviewed real-world evaluation.

**Limitations:** acted speech, few speakers, weak labels, lack of verified emergencies, English lexicon, unfamiliar accents/devices, mixed audio, coarse event times, uncalibrated scores and CPU latency. Fear/crying/glass breaking can be benign; a normal-sounding voice can be in distress. Pretrained scream detection is not separately fine-tuned or validated here.

**Future work:** reviewed emergency and ordinary audio, hard negatives, multilingual support, calibration to deployment prevalence, speaker/location/device-separated evaluation, event annotations, fairness assessment and a distilled low-latency environment model.

## References and verification

Sources: data/README.md. AST: https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593 and https://arxiv.org/abs/2104.01778. faster-whisper: https://github.com/SYSTRAN/faster-whisper. Telemetry controls: https://github.com/microsoft/onnxruntime/blob/main/docs/Privacy.md.

Read TEST_REPORT.md for checks actually run. Unit tests use deterministic injected models for pipeline/API contracts, not recognition accuracy. Synthetic smoke checkpoints are kept outside the delivered models directory. No fake accuracy claims or random trained weights are distributed.
