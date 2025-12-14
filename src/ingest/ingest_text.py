import re
import pandas as pd
from pathlib import Path
from typing import Any, List

from haystack import Document, Pipeline
from haystack.components.writers import DocumentWriter
from haystack_integrations.document_stores.chroma import ChromaDocumentStore

from sentence_transformers import SentenceTransformer
import numpy as np


# =====================================================================
# I. CONFIGURACIÓN Y MODELOS
# =====================================================================

BASE_DIR = Path(__file__).resolve().parent.parent
RAG_DOCS_DIR = BASE_DIR / "rag_docs"
CHROMA_PERSIST_DIR = BASE_DIR / "chroma_db"

# Colección conjunta texto + waypoints
TEXT_WP_COLLECTION = "rutas_text_waypoints"

# Modelo de embeddings (MiniLM)
TEXT_EMBEDDER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TEXT_MODEL = SentenceTransformer(TEXT_EMBEDDER_MODEL, trust_remote_code=True)

# CSV reales del proyecto
DOCS_SECTIONS_CHUNKS = RAG_DOCS_DIR / "docs_sections_chunks_ve.csv"
DOCS_WAYPOINTS = RAG_DOCS_DIR / "docs_waypoints_ve.csv"

# Campos numéricos importantes en rutas/waypoints
NUMERIC_META_KEYS = [
    "distancia_total_km",
    "altitud_minima_m",
    "altitud_maxima_m",
    "desnivel_acumulado_pos",
    "desnivel_acumulado_neg",
    "tiempo_total_min",
]


# =====================================================================
# II. FUNCIONES AUXILIARES
# =====================================================================

def embed_documents(docs: List[Document]) -> None:
    """
    Calcula embeddings con MiniLM y los asigna a cada Document.embedding.
    """
    if not docs:
        return

    texts = []
    for doc in docs:
        txt = doc.content or ""
        prefix = "Represent this document for retrieval: "
        texts.append(prefix + txt)

    embs = TEXT_MODEL.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    for doc, emb in zip(docs, embs):
        doc.embedding = emb.tolist()


def df_to_documents(df: pd.DataFrame) -> List[Document]:
    """
    Convierte un DataFrame de texto o waypoints (tabla) a Document.

    Usa:
      - content: columna 'content'
      - data_type: texto/waypoint
      - metadatos: columnas meta_*
    """
    docs: List[Document] = []

    for _, row in df.iterrows():
        content = str(row.get("content", "")).strip()
        if not content:
            continue

        data_type = str(row.get("data_type", "texto")).strip()

        meta: dict[str, Any] = {"data_type": data_type}

        # doc_id original
        if "doc_id" in row and pd.notna(row["doc_id"]):
            meta["doc_id"] = str(row["doc_id"])

        # original_doc_id en chunks de texto (si la columna existe)
        if "original_doc_id" in row and pd.notna(row.get("original_doc_id")):
            meta["original_doc_id"] = str(row["original_doc_id"])

        # chunk index
        if "chunk_index" in row and pd.notna(row.get("chunk_index")):
            try:
                meta["chunk_index"] = int(row["chunk_index"])
            except Exception:
                meta["chunk_index"] = str(row["chunk_index"])

        # token_count
        if "token_count" in row and pd.notna(row.get("token_count")):
            try:
                meta["token_count"] = int(row["token_count"])
            except Exception:
                meta["token_count"] = str(row["token_count"])

        # metadatos meta_*
        for col, val in row.items():
            if pd.isna(val):
                continue

            if col in [
                "content",
                "data_type",
                "doc_id",
                "original_doc_id",
                "chunk_index",
                "token_count",
            ]:
                continue

            if col.startswith("meta_"):
                key = col[5:]  # quitar "meta_"
                sval = str(val).strip()
                # Campos numéricos importantes
                if key in NUMERIC_META_KEYS:
                    try:
                        meta[key] = float(val)
                    except Exception:
                        meta[key] = sval
                    continue

                # Resto de metadatos: tipado automático genérico
                s_clean = sval.replace(".", "", 1)
                if s_clean.isdigit():
                    meta[key] = float(sval) if "." in sval else int(sval)
                else:
                    meta[key] = sval

        docs.append(Document(content=content, meta=meta))

    return docs


# =====================================================================
# III. MAIN: INGESTA TEXTO + WAYPOINTS
# =====================================================================

def main():
    # 1) Cargar CSVs
    print(f"Leyendo texto chunked: {DOCS_SECTIONS_CHUNKS}")
    df_sections = pd.read_csv(DOCS_SECTIONS_CHUNKS, encoding="utf-8")
    if "data_type" not in df_sections.columns:
        df_sections["data_type"] = "texto"

    if DOCS_WAYPOINTS.exists():
        print(f"Leyendo waypoints (tabla): {DOCS_WAYPOINTS}")
        df_waypoints = pd.read_csv(DOCS_WAYPOINTS, encoding="utf-8")
        if "data_type" not in df_waypoints.columns:
            df_waypoints["data_type"] = "waypoint"
    else:
        print(f"No existe {DOCS_WAYPOINTS}, no se cargarán waypoints.")
        df_waypoints = pd.DataFrame(columns=["content"])

    # 2) Convertir a Document
    docs_text = df_to_documents(df_sections)
    docs_waypoints = df_to_documents(df_waypoints) if not df_waypoints.empty else []

    all_docs = docs_text + docs_waypoints

    print(f"Documentos texto (secciones chunked): {len(docs_text)}")
    print(f"Documentos waypoints (tabla):       {len(docs_waypoints)}")
    print(f"TOTAL documentos a indexar:         {len(all_docs)}")

    if not all_docs:
        print("No hay documentos para indexar. Revisa los CSV.")
        return

    # 3) Calcular embeddings
    print("\n Calculando embeddings con MiniLM...")
    embed_documents(all_docs)

    # 4) Configurar DocumentStore
    doc_store = ChromaDocumentStore(
        persist_path=str(CHROMA_PERSIST_DIR),
        collection_name=TEXT_WP_COLLECTION,
        distance_function="cosine",
    )

    # 5) Vaciar colección antes de reindexar
    try:
        doc_store.delete_all_documents()
        print(f" Colección '{TEXT_WP_COLLECTION}' vaciada.")
    except Exception as e:
        print(f"(Aviso) No se pudo vaciar la colección: {e}")

    # 6) Ingesta con Haystack
    writer = DocumentWriter(document_store=doc_store)
    pipe = Pipeline()
    pipe.add_component("writer", writer)

    print("\n Indexando en Chroma (texto + waypoints)...")
    pipe.run({"writer": {"documents": all_docs}})

    print("\n Ingesta texto + waypoints completada.")
    print(f"   Colección: {TEXT_WP_COLLECTION}")
    print(f"   Total documentos en store: {doc_store.count_documents()}")


if __name__ == "__main__":
    main()
