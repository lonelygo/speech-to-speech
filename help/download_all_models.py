import torch
import nltk
from faster_whisper import WhisperModel
from melo.api import TTS
import ChatTTS
from huggingface_hub import snapshot_download
import os
import sys

# --- Configuration ---
# All models and data will be downloaded into this directory.
CACHE_DIR = "./model_cache"


def set_cache_environment_variables():
    """
    Set environment variables to force all libraries to use our local cache directory.
    """
    # For Hugging Face models (like faster-whisper, ChatTTS)
    os.environ["HF_HOME"] = CACHE_DIR
    os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(CACHE_DIR, "huggingface")
    
    # For PyTorch Hub models (like silero-vad)
    os.environ["TORCH_HOME"] = os.path.join(CACHE_DIR, "torch")
    
    # Add the NLTK data path
    nltk.data.path.append(CACHE_DIR)
    
    # Create directories if they don't exist
    os.makedirs(os.path.join(CACHE_DIR, "huggingface"), exist_ok=True)
    os.makedirs(os.path.join(CACHE_DIR, "torch", "hub"), exist_ok=True)


def download_stt():
    """
    Downloads the Faster Whisper STT model.
    """
    print("\n--- Downloading STT model (faster-whisper large-v3) ---")
    try:
        WhisperModel("large-v3", device="cpu", compute_type="int8")
        print("✅ STT model downloaded successfully.")
    except Exception as e:
        print(f"❌ Error downloading STT model: {e}", file=sys.stderr)


def download_melo_tts():
    """
    Downloads the MeloTTS model for Chinese.
    """
    print("\n--- Downloading TTS model (MeloTTS for Chinese) ---")
    try:
        TTS(language="ZH", device="cpu")
        print("✅ MeloTTS model downloaded successfully.")
    except Exception as e:
        print(f"❌ Error downloading MeloTTS model: {e}", file=sys.stderr)


def download_chat_tts():
    """
    Downloads the ChatTTS model using the huggingface source.
    """
    print("\n--- Downloading TTS model (ChatTTS) ---")
    print("Note: ChatTTS download can be slow and take up significant space (~4GB).")
    try:
        chat = ChatTTS.Chat()
        # By specifying source='huggingface', we ensure all files go to the HF_HOME cache.
        chat.load(source="huggingface", compile=False)
        print("✅ ChatTTS model downloaded successfully.")
    except Exception as e:
        print(f"❌ Error downloading ChatTTS model: {e}", file=sys.stderr)

def download_openvoice_v2():
    """
    Downloads the OpenVoice V2 checkpoints.
    """
    print("\n--- Downloading TTS model (OpenVoice V2 checkpoints) ---")
    try:
        repo_id = "myshell-ai/OpenVoiceV2"
        # We download the checkpoints into a specific subfolder within our main cache
        local_dir = os.path.join(CACHE_DIR, "OpenVoiceV2_checkpoints")
        snapshot_download(repo_id=repo_id, local_dir=local_dir, local_dir_use_symlinks=False)
        print(f"✅ OpenVoice V2 checkpoints downloaded successfully to {local_dir}.")
    except Exception as e:
        print(f"❌ Error downloading OpenVoice V2 checkpoints: {e}", file=sys.stderr)

def download_vad():
    """
    Downloads the Silero VAD model.
    """
    print("\n--- Downloading VAD model (silero-vad) ---")
    try:
        torch.hub.set_dir(os.path.join(os.environ["TORCH_HOME"], "hub"))
        torch.hub.load("snakers4/silero-vad", "silero_vad", force_reload=True)
        print("✅ VAD model downloaded successfully.")
    except Exception as e:
        print(f"❌ Error downloading VAD model: {e}", file=sys.stderr)

def download_nltk_data():
    """
    Downloads the necessary NLTK data.
    """
    print("\n--- Downloading NLTK data (punkt) ---")
    try:
        nltk.download("punkt", download_dir=os.path.join(CACHE_DIR, "nltk_data"))
        print("✅ NLTK data downloaded successfully.")
    except Exception as e:
        print(f"❌ Error downloading NLTK data: {e}", file=sys.stderr)

def main():
    """
    Main function to download all required models for all deployment configurations.
    """
    print("--- Starting model download process ---")
    
    set_cache_environment_variables()
    
    abs_cache_dir = os.path.abspath(CACHE_DIR)
    print(f"All models will be downloaded to: {abs_cache_dir}")
    print("This may take a while and require significant disk space.")

    download_stt()
    download_melo_tts()
    download_chat_tts()
    download_openvoice_assets()
    download_vad()
    download_nltk_data()

    print("\n--- All downloads attempted ---")
    print("Please check the output above for any errors.")
    print(f"\nNext steps:")
    print(f"1. Inspect the contents of the '{abs_cache_dir}' directory.")
    print(f"2. Copy the contents of this directory to your PVC. For example, using a helper pod:")
    print(f"   kubectl cp {abs_cache_dir}/. helper-pod-for-s2s:/data/ -n maas")
    print("   (This assumes your PVC is mounted at /data in the helper pod)")


if __name__ == "__main__":
    main()
