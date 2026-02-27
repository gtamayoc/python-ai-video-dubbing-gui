"""
AI Audio Translator - Entry Point
"""
import os
import sys
import warnings
import logging
from dotenv import load_dotenv

# Block torchcodec globally to avoid DLL/FFmpeg errors on Windows
try:
    import torchcodec
except Exception:
    sys.modules["torchcodec"] = None

# Force huggingface_hub to never use symlinks on Windows to prevent WinError 1314
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
try:
    import huggingface_hub.file_download
    huggingface_hub.file_download.are_symlinks_supported = lambda *args, **kwargs: False
except ImportError:
    pass

# Load environment variables from .env file
load_dotenv()

hf_token = os.environ.get("HF_TOKEN")
if hf_token and hf_token.strip() and not hf_token.startswith("YOUR_"):
    try:
        from huggingface_hub import login
        login(token=hf_token.strip(), add_to_git_credential=False)
        logging.info("Hugging Face Hub authenticated.")
    except Exception as e:
        logging.warning(f"Failed to authenticate with Hugging Face Hub: {e}")
else:
    logging.warning("No valid HF_TOKEN found in .env. Some services (Diarization) might fail.")

# Suppress noisy warnings from pyannote/torchcodec and others
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["PYANNOTE_AUDIO_IO"] = "false" # Try to skip torchcodec check if possible
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Set up logging before anything else
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

from src.ui.main_window import MainWindow

if __name__ == "__main__":
    app = MainWindow()
    app.mainloop()
