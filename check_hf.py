import os
import logging
from dotenv import load_dotenv
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_hf():
    load_dotenv()
    token = os.environ.get("HF_TOKEN")
    
    if not token:
        print("❌ ERROR: HF_TOKEN no encontrado en el archivo .env")
        return

    print(f"ℹ️ Token detectado: {token[:6]}...{token[-4:]}")
    
    models = [
        "pyannote/speaker-diarization-3.1",
        "pyannote/segmentation-3.0"
    ]
    
    headers = {"Authorization": f"Bearer {token}"}
    
    import httpx
    
    for model in models:
        url = f"https://huggingface.co/{model}/resolve/main/config.yaml"
        print(f"🔍 Verificando acceso HEAD a {url}...")
        try:
            with httpx.Client() as client:
                response = client.head(url, headers=headers, follow_redirects=True)
            
            if response.status_code == 200 or response.status_code == 302:
                print(f"✅ Acceso concedido a {model}")
            elif response.status_code == 403:
                print(f"❌ ERROR 403: Acceso denegado a {model}.")
                print("   Esto confirma que el servidor rechaza la autenticación para este archivo específico.")
            elif response.status_code == 401:
                print(f"❌ ERROR 401: Token inválido.")
            else:
                print(f"❓ Resultado inesperado ({response.status_code})")
        except Exception as e:
            print(f"💥 Error de conexión: {e}")

if __name__ == "__main__":
    check_hf()
