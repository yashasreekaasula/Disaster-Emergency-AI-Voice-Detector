# Verification report

Verified in a Linux sandbox using Python 3.11.17 and the pinned CPU dependencies. No datasets or synthetic checkpoints are shipped as trained deliverables.

## Passed checks

- `python -m compileall` for src, scripts, app and tests.
- Imported every Python module under src, scripts and app.
- **37 pytest tests passed**, including keyword matching, phrase timing/fuzzy matches/context suppression, fusion rules, exact class balancing, threshold selection, stale-fusion bypass, audio normalization/resampling, silence handling, API validation, missing models and WebSocket windows.
- All six dataset parsers exercised using temporary synthetic files with the original naming/metadata conventions. Required label coverage and speaker/source partition independence checked.
- Installed and checked Python dependencies; `pip check` reported no broken requirements.
- Started a real Uvicorn server. Health endpoint returned HTTP 200. Synthetic WAV upload without checkpoints returned the intended HTTP 503 with training commands; server remained running.
- Unit/API tests with deterministic injected models returned HTTP 200 and valid result fields.
- A real Uvicorn upload using isolated synthetic neural/fusion checkpoints returned HTTP 200, all 8 emotion labels, all 56 environment labels and the expected fast-model warnings.
- Downloaded real AST and faster-whisper pretrained weights and loaded them from local cache.
- Real AST forward pass produced 768-dimensional embeddings and 527 native AudioSet outputs.
- Ran loss/backpropagation/optimizer steps and saved/loaded both neural checkpoint formats on synthetic inputs.
- On isolated synthetic checkpoints/audio: ran build_fusion_dataset, train_fusion, evaluate (fusion task), and predict_cli successfully. These checks exercise software integration, not recognition accuracy.
- JavaScript syntax checks passed for app.js and pcm-worklet.js.
- AudioWorklet resampling checked at 16000, 44100, 48000 and 96000 Hz: each one-second signal produced 16000 output samples with finite, sensible level readings.

## Not established by these checks

- No full downloads of the six real datasets, no full dataset training, no independently verified distress-recognition accuracy, fairness or calibration.
- The complete neural CLI training runs have not been rerun on all real datasets. The parser, neural optimization/checkpoint paths and fusion CLI integration are tested independently.
- No physical microphone/browser permission test or final visual browser review: a required browser socket operation was blocked by the sandbox. WebSocket protocol and AudioWorklet signal processing were tested directly.
- Windows/macOS execution, CUDA/cuDNN execution and Docker build were not run in this Linux CPU sandbox.
- Dataset-host/Kaggle authentication, future download availability and hardware performance remain external conditions.

An earlier smoke process was blocked by automatic approval review after an ONNX Runtime telemetry attempt with an unknown payload. The project now disables optional ONNX/Hugging Face telemetry before imports; the subsequent cached/offline runtime smoke check completed. No authorization for sending audio or project data to a telemetry service was requested or assumed.

Synthetic test outputs and checkpoints are excluded from the ZIP. Reported test success is a software validation result, not an accuracy claim. Full accuracy, bias and false-alarm assessment requires actual training and independent reviewed recordings.
