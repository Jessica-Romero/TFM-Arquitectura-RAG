import os
from pathlib import Path
import re
from typing import List, Dict, Any, Optional

import pandas as pd
from PIL import Image
from sentence_transformers import SentenceTransformer

from haystack import Document, Pipeline
from haystack.components.writers import DocumentWriter
from haystack_integrations.document_stores.chroma import ChromaDocumentStore


# =====================================================================
# I. CONFIGURACIÓN
# =====================================================================

BASE_DIR = Path(__file__).resolve().parents[2]

# 🟡 IMPORTANTE:
# Usa la MISMA ruta que en tu ingesta de texto.
# Si en texto usas BASE_DIR / "chroma_db", cámbialo aquí igual.
CHROMA_PERSIST_DIR = BASE_DIR / "src" / "chroma_db"

RAG_DOCS_DIR = BASE_DIR / "src" / "rag_docs"
DOCS_IMAGES = RAG_DOCS_DIR / "docs_images_ve.csv"

IMAGES_COLLECTION = "rutas_imagenes"

# Modelo CLIP (versión más ligera para CPU)
IMAGE_EMBEDDER_MODEL = "sentence-transformers/clip-ViT-B-32"
IMAGE_MODEL = SentenceTransformer(IMAGE_EMBEDDER_MODEL)


# =====================================================================
# II. DETECTAR COLUMNA DE RUTA DE IMAGEN
# =====================================================================

def detect_image_path_column(df: pd.DataFrame) -> str:
    if "meta_img_path" in df.columns:
        return "meta_img_path"
    raise ValueError("No se encontró la columna 'meta_img_path' en docs_images_v3.csv")


# =====================================================================
# III. CONVERSIÓN DF -> DOCUMENTS + IMÁGENES
# =====================================================================

def build_docs_and_images(df: pd.DataFrame) -> tuple[list[Document], list[Image.Image]]:
    """
    A partir del DataFrame de docs_images_v3.csv:
      - abre las imágenes
      - crea Document con metadatos
      - devuelve (docs, pil_images) alineados
    """
    img_col = detect_image_path_column(df)

    docs: List[Document] = []
    pil_images: List[Image.Image] = []

    for idx, row in df.iterrows():
        # ruta de imagen desde el CSV
        raw_path = str(row[img_col])

        # Soportar ruta relativa al proyecto o absoluta
        img_path = Path(raw_path)
        if not img_path.is_absolute():
            img_path = BASE_DIR / img_path

        if not img_path.exists():
            print(f"[Fila {idx}] Imagen no encontrada: {img_path}")
            continue

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[Fila {idx}] Error abriendo imagen {img_path}: {e}")
            continue

        # ---------- METADATOS ----------
        data_type = str(row.get("data_type", "image")).strip()

        meta: Dict[str, Any] = {
            "data_type": data_type,
            "image_path": str(img_path.relative_to(BASE_DIR)),  # guardar ruta relativa
        }

        # doc_id si existe
        if "doc_id" in df.columns and pd.notna(row.get("doc_id")):
            meta["doc_id"] = str(row["doc_id"])

        # metadatos meta_*
        for col, val in row.items():
            if pd.isna(val):
                continue

            if col in ["content", "data_type", img_col, "doc_id"]:
                continue

            if col.startswith("meta_"):
                key = col[5:]  # quitar "meta_"
                sval = str(val).strip()
                # intentar tipar nº si procede
                s_clean = sval.replace(".", "", 1)
                if s_clean.isdigit():
                    try:
                        meta[key] = float(sval) if "." in sval else int(sval)
                    except Exception:
                        meta[key] = sval
                else:
                    meta[key] = sval

        # content: puedes usar texto descriptivo si lo has creado,
        # o simplemente dejarlo vacío
        content = str(row.get("content", "") or "").strip()

        doc = Document(content=content, meta=meta)
        docs.append(doc)
        pil_images.append(img)

    return docs, pil_images


# =====================================================================
# IV. MAIN: INGESTA DE IMÁGENES
# =====================================================================

def main():
    print("🖼️ INICIO INGESTA DE IMÁGENES DESDE CSV")
    print("=" * 60)
    print(f" CSV imágenes: {DOCS_IMAGES}")

    if not DOCS_IMAGES.exists():
        print(f" No existe el CSV: {DOCS_IMAGES}")
        return

    df_images = pd.read_csv(DOCS_IMAGES, encoding="utf-8")

    if "data_type" not in df_images.columns:
        df_images["data_type"] = "image"

    print(f" Filas en docs_images_v3.csv: {len(df_images)}")

    # 1) Construir Document + PIL Images
    docs, pil_images = build_docs_and_images(df_images)
    print(f" Imágenes válidas: {len(pil_images)}")

    if not docs:
        print(" No hay imágenes válidas para indexar.")
        return

    # 2) Embeddings con CLIP
    print("\n Calculando embeddings de imágenes con CLIP...")
    embs = IMAGE_MODEL.encode(
        pil_images,
        batch_size=32,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    for doc, emb in zip(docs, embs):
        doc.embedding = emb.tolist()

    # 3) Configurar ChromaDocumentStore
    print("\n Configurando ChromaDB...")
    doc_store = ChromaDocumentStore(
        persist_path=str(CHROMA_PERSIST_DIR),
        collection_name=IMAGES_COLLECTION,
        distance_function="cosine",
    )

    # 4) Vaciar colección de imágenes
    try:
        doc_store.delete_all_documents()
        print(f" Colección '{IMAGES_COLLECTION}' vaciada.")
    except Exception as e:
        print(f"(Aviso) No se pudo vaciar la colección '{IMAGES_COLLECTION}': {e}")

    # 5) Indexar con Haystack
    writer = DocumentWriter(document_store=doc_store)
    pipe = Pipeline()
    pipe.add_component("writer", writer)

    print("\n Indexando imágenes en Chroma...")
    pipe.run({"writer": {"documents": docs}})

    total = doc_store.count_documents()
    print("\n INGESTA DE IMÁGENES COMPLETADA")
    print("=" * 60)
    print(f" Colección: {IMAGES_COLLECTION}")
    print(f" Ruta BD:  {CHROMA_PERSIST_DIR}")
    print(f"Documentos indexados: {total}")


if __name__ == "__main__":
    main()
