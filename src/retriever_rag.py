from pathlib import Path
from typing import List, Optional, Dict, Any

import chromadb
from sentence_transformers import SentenceTransformer


# ============================================================
# 1. CONFIGURACIÓN
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_PERSIST_DIR = BASE_DIR / "src" / "chroma_db"

COLLECTION_NAME = "rutas_text_waypoints"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

print(f" Usando BD Chroma en: {CHROMA_PERSIST_DIR}")
print(f" Colección: {COLLECTION_NAME}")
print(f" Modelo embeddings (query): {EMBED_MODEL_NAME}")

# Cargamos el modelo solo una vez
model = SentenceTransformer(EMBED_MODEL_NAME)


# ============================================================
# 2. FUNCIONES DE EMBEDDING Y BÚSQUEDA
# ============================================================

def embed_query(query: str) -> List[float]:
    """
    Calcula el embedding de una query usando MiniLM.
    Mantengo un pequeño prefijo de 'retrieval' para ser consistente
    con la indexación (aunque para MiniLM no es obligatorio).
    """
    query = query.strip()
    text = "Represent the query for retrieval: " + query

    emb = model.encode(
        [text],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )[0]

    return emb.tolist()


def get_collection():
    """
    Abre el cliente de Chroma y devuelve la colección.
    """
    client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except Exception as e:
        raise RuntimeError(
            f"No se pudo abrir la colección '{COLLECTION_NAME}'. "
            f"Asegúrate de haber ejecutado antes la ingesta con MiniLM."
        ) from e
    return collection


def search(
    query: str,
    top_k: int = 5,
    where: Optional[Dict[str, Any]] = None,
):
    """
    Lanza una búsqueda semántica en Chroma.

    Parámetros:
        query: texto de la pregunta
        top_k: número de resultados a devolver
        where: filtros sobre metadatos, por ejemplo:
               {"data_type": "texto"}
               {"data_type": "waypoint"}
    """
    print(f"\n Query: {query}")
    if where:
        print(f" Filtros: {where}")

    emb = embed_query(query)
    collection = get_collection()

    result = collection.query(
        query_embeddings=[emb],
        n_results=top_k,
        where=where,
    )

    ids = result.get("ids", [[]])[0]
    docs = result.get("documents", [[]])[0]
    metas = result.get("metadatas", [[]])[0]
    dists = result.get("distances", [[]])[0]

    print(f"\n📊 Resultados (top_k={top_k}):\n")

    for rank, (doc_id, doc_text, meta, dist) in enumerate(
        zip(ids, docs, metas, dists), start=1
    ):
        data_type = meta.get("data_type", "unknown")
        ruta = meta.get("nombre_ruta")
        pais = meta.get("Pais") or meta.get("pais")
        distancia = meta.get("distancia_total_km")
        poblacion_cercana = meta.get("poblacion_cercana")

        print(f"#{rank}  ID: {doc_id}")
        print(f"    data_type: {data_type}")
        if ruta:
            print(f"    ruta: {ruta}")
        if pais:
            print(f"    país: {pais}")
        if poblacion_cercana:
            print(f"    poblacion_cercana: {poblacion_cercana}")
        if distancia is not None:
            print(f"    distancia_total_km: {distancia}")
        print(f"    distancia (espacio vectorial): {dist:.4f}")

        # Mostrar solo los primeros caracteres del contenido
        print("    --- contenido ---")
        if doc_text is None:
            preview = ""
        else:
            preview = (doc_text[:400] + "…") if len(doc_text) > 400 else doc_text
        print(f"    {preview}")
        print("    -----------------\n")

    return result


# ============================================================
# 3. MODO INTERACTIVO DE PRUEBA
# ============================================================

def interactive_loop():
    """
    Pequeño REPL para probar búsquedas desde consola.
    """
    print("\n Modo interactivo. Escribe una pregunta (o 'salir' para terminar).")
    print("   Puedes cambiar el filtro 'where' en el código si quieres solo texto o solo waypoints.\n")

    while True:
        q = input(" Pregunta: ").strip()
        if not q:
            continue
        if q.lower() in {"salir", "exit", "quit"}:
            print("👋 Saliendo.")
            break

        # Filtros de ejemplo:
        #   where = {"data_type": "texto"}      # solo chunks de texto
        #   where = {"data_type": "waypoint"}   # solo waypoints
        #   where = None                        # ambos tipos mezclados
        where = {
        "poblacion_cercana": {"$eq": "El Bruc"}
        }


        search(q, top_k=5, where=where)


if __name__ == "__main__":
    interactive_loop()
