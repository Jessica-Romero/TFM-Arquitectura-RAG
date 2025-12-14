import os
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai  # ← Import directo

# 1. Cargar .env
BASE_DIR = Path(__file__).resolve().parents[1]
env_path = BASE_DIR / '.env'

if not env_path.exists():
    print(f"ERROR: No se encuentra .env en {env_path}")
    print("Crea el archivo .env con: GEMINI_API_KEY=tu_clave")
    exit(1)

load_dotenv(env_path)

# 2. Obtener API key
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    print("ERROR: GEMINI_API_KEY no encontrada en .env")
    exit(1)

# 3. FORZAR configuración de Gemini ANTES que Haystack la use
print(f" Configurando Gemini con key: {API_KEY[:15]}...")
genai.configure(api_key=API_KEY)

# 4. También establecer como variable de entorno para Haystack
os.environ["GOOGLE_API_KEY"] = API_KEY
print(" GOOGLE_API_KEY configurada para Haystack")