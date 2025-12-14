from chromadb import PersistentClient
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]   # raíz del repo
CHROMA_PERSIST_DIR = BASE_DIR / "src" / "chroma_db"

client = PersistentClient(path=CHROMA_PERSIST_DIR)
col = client.get_collection("rutas_text_waypoints")

emb = col.get(limit=1, include=["embeddings"])["embeddings"][0]
print("Embedding length:", len(emb))
