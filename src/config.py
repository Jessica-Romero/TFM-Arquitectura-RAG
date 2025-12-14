import os
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai

# Variables globales
TEXT_COLLECTION = ""
IMAGE_COLLECTION = ""
TEXT_EMBED_MODEL = ""
CHROMA_PERSIST_DIR = ""
BASE_DIR = ""
API_KEY = ""

def init_process(TEXT_COLLECTION_param, IMAGE_COLLECTION_param, 
                 TEXT_EMBED_MODEL_param, PATH_CHROMA_DIR_param, BASE_DIR_param):
    """
    Configuración de todo el sistema 
    Llama a esta función desde otro script para configurar las variables globales.
    """
    global TEXT_COLLECTION, IMAGE_COLLECTION, TEXT_EMBED_MODEL, BASE_DIR, CHROMA_PERSIST_DIR, API_KEY
    
    # 1. Guardar los parámetros
    TEXT_COLLECTION = TEXT_COLLECTION_param
    IMAGE_COLLECTION = IMAGE_COLLECTION_param
    TEXT_EMBED_MODEL = TEXT_EMBED_MODEL_param
    CHROMA_PERSIST_DIR = PATH_CHROMA_DIR_param
    BASE_DIR = BASE_DIR_param
    
    # 2. Cargar .env 
    env_path = BASE_DIR / '.env'
    
    if not env_path.exists():
        print(f"ERROR: No se encuentra .env en {env_path}")
        print("Crea el archivo .env con: GEMINI_API_KEY=tu_clave")
        exit(1)
    
    load_dotenv(env_path)
    
    # 3. Obtener API key ()
    API_KEY = os.getenv("GEMINI_API_KEY")
    if not API_KEY:
        print("ERROR: GEMINI_API_KEY no encontrada en .env")
        exit(1)

    # 4. Configurar Gemini 
    genai.configure(api_key=API_KEY)
    os.environ["GOOGLE_API_KEY"] = API_KEY
    print(" GOOGLE_API_KEY configurada para Haystack")
    
    # 6. Mostrar configuración Chroma
    print(f"BD Chroma: {CHROMA_PERSIST_DIR}")
    print(f"Colección TEXTO: {TEXT_COLLECTION}")
    print(f"Colección IMÁGENES: {IMAGE_COLLECTION}")
    print(f"Modelo texto (retrieval): {TEXT_EMBED_MODEL}")
