import os
import sys

# Try to disable torchcodec via env var
os.environ["PYANNOTE_AUDIO_IO"] = "false"

print(f"Python version: {sys.version}")
try:
    import torch
    print(f"Torch version: {torch.__version__}")
except ImportError:
    print("Torch not installed")

try:
    import torchaudio
    print(f"Torchaudio version: {torchaudio.__version__}")
except ImportError:
    print("Torchaudio not installed")

try:
    print("Attempting to import torchcodec...")
    import torchcodec
    print(f"Torchcodec version: {torchcodec.__version__}")
except Exception as e:
    print(f"❌ Torchcodec import failed: {e}")

print("\n--- Testing pyannote.audio ---")
try:
    from pyannote.audio import Pipeline
    print("✅ pyannote.audio imported successfully")
except Exception as e:
    print(f"❌ pyannote.audio import failed: {e}")

# Check if FFmpeg is in path
print("\n--- Checking FFmpeg ---")
import subprocess
try:
    res = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
    print(f"✅ FFmpeg found: {res.stdout.splitlines()[0]}")
except FileNotFoundError:
    print("❌ FFmpeg NOT found in PATH")
