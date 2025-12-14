# list_metadata_keys.py
import chromadb
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_PERSIST_DIR = BASE_DIR / "src" / "chroma_db"

client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
collection = client.get_collection("rutas_text_waypoints")

# Coge los primeros 50 documentos
result = collection.get(include=["metadatas"], limit=50)

all_keys = set()
for meta in result["metadatas"]:
    all_keys.update(meta.keys())

print("🔍 Metadatos presentes en la colección:")
for k in sorted(all_keys):
    print(" -", k)
