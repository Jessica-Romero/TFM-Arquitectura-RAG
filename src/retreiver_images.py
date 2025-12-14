from pathlib import Path
from typing import List, Optional, Dict, Any

import chromadb
from sentence_transformers import SentenceTransformer


# ============================================================
# 1. CONFIGURACIÓN
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]
CHROMA_PERSIST_DIR = BASE_DIR / "src" / "chroma_db"

COLLECTION_NAME = "rutas_imagenes"
EMBED_MODEL_NAME = "sentence-transformers/clip-ViT-B-32"

print(f"📁 BD Chroma: {CHROMA_PERSIST_DIR}")
print(f"🖼️ Colección imágenes: {COLLECTION_NAME}")
print(f"🔤 Modelo CLIP (query): {EMBED_MODEL_NAME}")

model = SentenceTransformer(EMBED_MODEL_NAME)


# ============================================================
# 2. EMBEDDING QUERY (texto → embedding CLIP)
# ============================================================

def embed_query(query: str) -> List[float]:
    emb = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False
    )[0]
    return emb.tolist()


def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
    try:
        return client.get_collection(name=COLLECTION_NAME)
    except Exception as e:
        raise RuntimeError(
            f"No existe la colección '{COLLECTION_NAME}'. "
            f"Asegúrate de haber ejecutado la ingesta de imágenes."
        ) from e


# ============================================================
# 3. BÚSQUEDA SEMÁNTICA DE IMÁGENES
# ============================================================

def search_images(query: str, top_k=5, where=None):
    print(f"\n🔍 Query: {query}")
    if where:
        print(f"   🔎 Filtros: {where}")

    emb = embed_query(query)
    collection = get_collection()

    result = collection.query(
        query_embeddings=[emb],
        n_results=top_k,
        where=where
    )

    ids = result["ids"][0]
    metas = result["metadatas"][0]
    dists = result["distances"][0]

    print(f"\n📸 Resultados (top {top_k})\n")

    for rank, (doc_id, meta, dist) in enumerate(zip(ids, metas, dists), start=1):
        ruta = meta.get("nombre_ruta")
        poblacion = meta.get("poblacion_cercana")
        image_path = meta.get("image_path")

        print(f"#{rank}  ID: {doc_id}")
        if ruta:
            print(f"    ruta: {ruta}")
        if poblacion:
            print(f"    población cercana: {poblacion}")
        print(f"    distancia vectorial: {dist:.4f}")
        print(f"    imagen: {image_path}")
        print("------------------------------")

    return result


# ============================================================
# 4. MODO INTERACTIVO
# ============================================================

def interactive_loop():
    print("\n💬 Modo interactivo de búsqueda de imágenes.")

    while True:
        q = input("\n❓ Pregunta (o 'salir'): ").strip()
        if q.lower() in {"salir", "exit", "quit"}:
            print("👋 Saliendo.")
            break

        where = {
            "poblacion_cercana": {"$eq": "El Bruc"}
        }

        search_images(q, top_k=5, where=where)


if __name__ == "__main__":
    interactive_loop()
