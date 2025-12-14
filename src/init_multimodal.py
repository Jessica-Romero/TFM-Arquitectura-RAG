from pathlib import Path
from config import *
from rag_multimodal import MiniLMQueryEmbedder, print_config

# 1. CONFIGURACIÓN INICIAL DEL SISTEMA
print("=== INICIANDO CONFIGURACIÓN ===")
init_process(
    TEXT_COLLECTION = "rutas_text_waypoints",
    IMAGE_COLLECTION = "rutas_imagenes",
    TEXT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2",
    BASE_DIR = Path(__file__).resolve().parents[1],
    CHROMA_PERSIST_DIR = BASE_DIR / "src" / "chroma_db"
)
print("\n")

# 2. CREAS EL EMBEDDER 
print("=== CREANDO EMBEDDER ===")
embedder = MiniLMQueryEmbedder() 

print("\n")

# 3. USAS EL EMBEDDER
print("=== PROCESANDO CONSULTA ===")
resultado = embedder.run("¿Qué es machine learning?")
print(f"Resultado: {resultado['model_used']}")
print(f"Longitud embedding: {len(resultado['embedding'])}")

print("\n")

# 4. PUEDES VER TODA LA CONFIGURACIÓN
print_config()