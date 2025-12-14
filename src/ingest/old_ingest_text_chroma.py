import re
import pandas as pd
from pathlib import Path
from typing import Any, List
from haystack import Document, Pipeline

from haystack.components.writers import DocumentWriter
from haystack_integrations.document_stores.chroma import ChromaDocumentStore

import chromadb  
from sentence_transformers import SentenceTransformer


# =====================================================================
# I. CONFIGURACIÓN Y MODELOS
# =====================================================================

BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_PERSIST_DIR = BASE_DIR / "chroma_db"

TEXT_WP_COLLECTION = "rutas_text_waypoints"

# Modelo de TEXTO para embeddings
TEXT_EMBEDDER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Cargamos el modelo de texto una vez
TEXT_MODEL = SentenceTransformer(TEXT_EMBEDDER_MODEL)


def text_embed_fn(texts: list[str]) -> list[list[float]]:
    """
    Función de embedding para ChromaDocumentStore.
    Recibe una lista de textos y devuelve una lista de vectores.
    """
    # encode(...) ya admite lista de strings
    embs = TEXT_MODEL.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return embs.tolist()


RAG_DOCS_DIR = BASE_DIR / "rag_docs"
DOCS_SECTIONS_CHUNKS = RAG_DOCS_DIR / "docs_sections_chunks_tokens.csv"
DOCS_WAYPOINTS = RAG_DOCS_DIR / "docs_waypoints_v2.csv"


# =====================================================================
# II. FUNCIONES AUXILIARES
# =====================================================================

def df_to_documents(df: pd.DataFrame) -> List[Document]:
    """
    Convierte un DataFrame de docs_* (sections o waypoints) a lista de Document.
    Usa:
      - content: columna 'content'
      - data_type: columna 'data_type'
      - metadatos: columnas que empiezan por 'meta_'
    """
    docs: List[Document] = []

    for _, row in df.iterrows():
        content = str(row["content"])
        data_type = str(row.get("data_type", "texto")).strip()

        meta: dict[str, Any] = {"data_type": data_type}

        # doc_id original (útil para trazabilidad)
        if "doc_id" in row and pd.notna(row["doc_id"]):
            meta["doc_id"] = str(row["doc_id"])

        # metadatos meta_*
        for col, val in row.items():
            if pd.isna(val):
                continue
            if col in ["content", "data_type", "doc_id"]:
                continue
            
            if col.startswith("meta_"):
                key = col[len("meta_"):]  # meta_pais -> pais

                if key in ["distancia_total_km",
                            "altitud_minima_m",
                            "altitud_maxima_m",
                            "desnivel_acumulado_pos",
                            "desnivel_acumulado_neg",
                            "tiempo_total_min"]:
                    meta[key] = float(val)
                else:
                    meta[key] = str(val).strip()
            elif col == "chunk_index":
                meta["chunk_index"] = int(val)

        docs.append(Document(content=content, meta=meta))

    return docs


# =====================================================================
# III. MAIN: INGESTA TEXTO + WAYPOINTS
# =====================================================================

def main():
    # 1) Cargar CSVs
    print(f"Leyendo: {DOCS_SECTIONS_CHUNKS}")
    df_sections = pd.read_csv(DOCS_SECTIONS_CHUNKS, encoding="utf-8")

    print(f"Leyendo: {DOCS_WAYPOINTS}")
    df_waypoints = pd.read_csv(DOCS_WAYPOINTS, encoding="utf-8")

    # asegurar data_type
    if "data_type" not in df_sections.columns:
        df_sections["data_type"] = "texto"
    if "data_type" not in df_waypoints.columns:
        df_waypoints["data_type"] = "waypoint"

    # 2) Convertir a Document
    docs_sections = df_to_documents(df_sections)
    docs_waypoints = df_to_documents(df_waypoints)
    all_docs = docs_sections + docs_waypoints

    print(f"Total documentos texto (secciones chunked): {len(docs_sections)}")
    print(f"Total documentos waypoints: {len(docs_waypoints)}")
    print(f"Total documentos a indexar: {len(all_docs)}")

    # 3) Configurar DocumentStore con embedding_function = callable 🔴
    doc_store = ChromaDocumentStore(
        persist_path=str(CHROMA_PERSIST_DIR),
        collection_name=TEXT_WP_COLLECTION,
        embedding_function=text_embed_fn,   
    )

    # vaciar colección antes de reindexar
    try:
        doc_store.delete_documents()
        print(f"Colección '{TEXT_WP_COLLECTION}' vaciada.")
    except Exception as e:
        print(f"(Aviso) No se pudo vaciar la colección (quizá ya estaba vacía): {e}")

    # 4) Writer de Haystack (Chroma se encarga de llamar a text_embed_fn)
    writer = DocumentWriter(document_store=doc_store)

    pipe = Pipeline()
    pipe.add_component("writer", writer)

    print("\n Indexando en Chroma con Haystack (texto + waypoints)...")
    pipe.run({"writer": {"documents": all_docs}})

    print("\n Ingesta texto + waypoints completada.")
    print(f"   Colección: {TEXT_WP_COLLECTION}")
    print(f"   Total documentos en store: {doc_store.count_documents()}")


if __name__ == "__main__":
    main()
