"""SoundSentinel local audio analysis."""
import os
# Set before ONNX/Hugging Face imports; prevent optional telemetry initialization.
os.environ["ORT_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
