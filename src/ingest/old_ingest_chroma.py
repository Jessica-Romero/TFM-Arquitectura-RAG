import re
import pandas as pd
from pathlib import Path
from typing import Any, List
from haystack import Document
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from PIL import Image
import numpy as np

# =====================================================================
# I. CONFIGURACIÓN Y MODELOS
# =====================================================================

BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_PERSIST_DIR = BASE_DIR / "chroma_db"

EMBEDDING_MODEL_NAME = "sentence-transformers/clip-ViT-L-14"
COLLECTION_TEXT_MULTIMODAL = "rutas_text_multimodal"
COLLECTION_WAYPOINTS = "rutas_tablas_waypoints"

try:
    CLIP_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)
except Exception as e:
    print(f"Error al cargar el modelo de embedding: {e}")
    raise SystemExit(1)

# =====================================================================
# II. FUNCIONES AUXILIARES
# =====================================================================

def clean_numeric_column(value: Any) -> float | None:
    """
    Extrae el primer valor numérico de una cadena y lo convierte a float.
    """
    if pd.isna(value) or value is None:
        return None

    valor_str = str(value).strip()
    RE_NUMERICO_LIMPIO = re.compile(r"([+\-]?\s*\d+[\.,]?\d*)")
    match = RE_NUMERICO_LIMPIO.search(valor_str)

    if not match:
        return None

    num_str = match.group(1).strip()
    num_str = num_str.replace(".", "")
    num_str = num_str.replace(",", ".")
    num_str = num_str.replace("+", "").replace("-", "").strip()

    try:
        return float(num_str)
    except ValueError:
        return None


def embed_content(content: str, is_image: bool = False, image_path: str | None = None):
    """
    Genera embeddings con CLIP.
    - Texto  -> encode([texto])
    - Imagen -> encode([PIL.Image])
    """
    if is_image:
        try:
            img = Image.open(image_path).convert("RGB")

            emb = CLIP_MODEL.encode(
                [img],                   # o directamente img, ambas valen
                convert_to_numpy=True,
                show_progress_bar=False,
            )[0]

            return emb.tolist()

        except Exception as e:
            print(f"⚠️ Error cargando imagen {image_path}: {e}")
            return None

    # Texto
    emb_text = CLIP_MODEL.encode(
        [content],
        convert_to_numpy=True,
        show_progress_bar=False,
    )[0]

    return emb_text.tolist()


def dataframe_a_documentos_haystack(df: pd.DataFrame, default_type: str
                                     | None = None) -> List[Document]:
    """
    Convierte un DataFrame de docs_* a lista de Document (Haystack).
    Usa la columna 'data_type' si existe; si no, usa default_type.
    Hace conversión final de columnas numéricas a float.
    """
    documents: List[Document] = []

    NUMERIC_COLS_BASE = [
        "distancia_total_km",
        "altitud_minima_m",
        "altitud_maxima_m",
        "desnivel_acumulado_pos",
        "desnivel_acumulado_neg",
        "tiempo_total_min",
    ]

    for _, row in df.iterrows():
        content = str(row["content"])

        # Tipo de dato (texto_seccion, image, waypoint_table, etc.)
        row_type = str(row.get("data_type", default_type) or default_type or "").strip()

        meta: dict[str, Any] = {
            "type": row_type
        }

        # Si viene doc_id en el CSV, lo guardamos como metadato
        doc_id_from_row = row.get("doc_id")
        if pd.notna(doc_id_from_row):
            meta["doc_id"] = str(doc_id_from_row)

        # Metadatos “normales”
        for col_name, value in row.items():
            if col_name in ["content", "data_type", "doc_id"]:
                continue
            if pd.isna(value):
                continue

            # numéricos especiales
            if col_name in NUMERIC_COLS_BASE:
                meta[col_name] = clean_numeric_column(value)
            # IDs
            elif col_name in ["route_id", "img_orden", "table_idx", "section_id", "wp_orden"]:
                try:
                    meta[col_name] = int(value)
                except Exception:
                    meta[col_name] = str(value).strip()
            # path de imagen (para caption)
            elif col_name in ["meta_img_path", "img_path"]:
                meta["img_path"] = str(value).strip()
            else:
                meta[col_name] = str(value).strip()

        documents.append(Document(content=content, meta=meta))

    return documents

def indexar_coleccion(documents: List[Document], collection_name: str, batch_size: int = 50) -> None:
    """
    Indexa una lista de Document en una colección de Chroma usando CLIP.
    Si la colección existe, la borra y la vuelve a crear.
    """
    client = chromadb.PersistentClient(
        path=str(CHROMA_PERSIST_DIR),
        settings=Settings(allow_reset=True),
    )

    # 💣 1) Si existe la colección, la borramos entera
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        # Si no existe todavía, no pasa nada
        pass

    # ✅ 2) Creamos la colección vacía
    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    print(f"\n--- Indexando {len(documents)} documentos en la colección: {collection_name} ---")

    ids_batch, embeddings_batch, metadatas_batch, documents_batch = [], [], [], []

    for doc in documents:
        meta = doc.meta or {}

        tipo = (meta.get("type") or "").lower()
        is_image_data = tipo in ["image", "images", "image_caption"]
        img_path = meta.get("img_path") if is_image_data else None

        emb = embed_content(doc.content, is_image_data, img_path)
        if emb is None:
            continue

        doc_id = (
            meta.get("doc_id")
            or str(meta.get("route_id"))
            or str(len(ids_batch))
        )

        ids_batch.append(str(doc_id))
        embeddings_batch.append(emb)
        metadatas_batch.append(meta)
        documents_batch.append(doc.content)

        if len(ids_batch) >= batch_size:
            collection.upsert(
                ids=ids_batch,
                embeddings=embeddings_batch,
                metadatas=metadatas_batch,
                documents=documents_batch,
            )
            ids_batch, embeddings_batch, metadatas_batch, documents_batch = [], [], [], []

    # Último batch
    if ids_batch:
        collection.upsert(
            ids=ids_batch,
            embeddings=embeddings_batch,
            metadatas=metadatas_batch,
            documents=documents_batch,
        )

    print(f"✅ Indexación en {collection_name} completada. Total final: {collection.count()} documentos.")

# =====================================================================
# III. PUNTO DE ENTRADA
# =====================================================================

if __name__ == "__main__":
    DOCS_SECTIONS_PATH = BASE_DIR / "rag_docs" / "docs_sections_v2.csv"
    DOCS_IMAGES_PATH = BASE_DIR / "rag_docs" / "docs_images_v2.csv"
    DOCS_WAYPOINTS_PATH = BASE_DIR /"rag_docs" / "docs_waypoints_v2.csv"

    try:
        df_sections = pd.read_csv(DOCS_SECTIONS_PATH)
        df_images = pd.read_csv(DOCS_IMAGES_PATH)
        df_waypoints = pd.read_csv(DOCS_WAYPOINTS_PATH)
    except FileNotFoundError as e:
        print(f"❌ FATAL: Archivo preparado no encontrado: {e}")
        print("Asegúrate de ejecutar primero los scripts build_docs_*.py")
        raise SystemExit(1)

    # --- Waypoints / Tablas (colección propia) ---
    docs_waypoints = dataframe_a_documentos_haystack(df_waypoints)
    indexar_coleccion(docs_waypoints, COLLECTION_WAYPOINTS)

    # --- Texto + Imágenes (colección multimodal) ---
    # Aquí confiamos en que docs_sections_v2 y docs_images_v2 ya tienen una columna `data_type`
    # por ejemplo: "texto_seccion" / "image"
    df_multimodal_raw = pd.concat([df_sections, df_images], ignore_index=True)

    docs_multimodal = dataframe_a_documentos_haystack(df_multimodal_raw)
    indexar_coleccion(docs_multimodal, COLLECTION_TEXT_MULTIMODAL)

    print("\n\n✅ ARQUITECTURA RAG MULTIMODAL COMPLETAMENTE INDEXADA.\n")
