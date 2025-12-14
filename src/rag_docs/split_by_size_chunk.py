from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import re
from datetime import datetime

# Usa el mismo modelo que en la ingesta (MiniLM)
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

MAX_TOKENS = 400
OVERLAP_TOKENS = 80

BASE_DIR = Path(__file__).resolve().parent.parent
RAG_DOCS_DIR = BASE_DIR / "rag_docs"

INPUT_CSV = RAG_DOCS_DIR / "docs_sections_ve.csv"
OUTPUT_CSV = RAG_DOCS_DIR / "docs_sections_chunks_ve.csv"


def load_tokenizer():
    """Carga el modelo y tokenizador (solo para tokenizar/contar tokens)."""
    print(f"Cargando modelo para tokenizar: {MODEL_NAME
    }")
    model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)
    # SentenceTransformer expone el tokenizer del modelo base
    tokenizer = model.tokenizer
    return model, tokenizer


def chunk_text_smart(
    text: str,
    tokenizer,
    max_length: int = MAX_TOKENS,
    overlap: int = OVERLAP_TOKENS,
) -> List[Dict[str, Any]]:
    """Divide texto en chunks preservando estructura y evitando cortes raros."""
    if not isinstance(text, str) or not text.strip():
        return []

    # Normalizar espacios
    text = re.sub(r"\s+", " ", text.strip())
    tokens = tokenizer.encode(text, add_special_tokens=False)
    total_tokens = len(tokens)

    if total_tokens <= max_length:
        return [{"text": text, "token_count": total_tokens}]

    chunks = []
    start = 0

    while start < total_tokens:
        end = min(start + max_length, total_tokens)

        # Intentar cortar en un punto "natural"
        if end < total_tokens:
            chunk_tokens = tokens[start:end]
            chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)

            # Buscar un punto, salto de línea o ';' hacia el 90% del chunk
            last_period = max(
                chunk_text.rfind(". ", 0, int(len(chunk_text) * 0.9)),
                chunk_text.rfind("\n", 0, int(len(chunk_text) * 0.9)),
                chunk_text.rfind("; ", 0, int(len(chunk_text) * 0.9)),
            )

            if last_period > len(chunk_text) * 0.3:
                adjusted_text = chunk_text[: last_period + 1]
                adjusted_tokens = tokenizer.encode(
                    adjusted_text, add_special_tokens=False
                )
                end = start + len(adjusted_tokens)

        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True).strip()

        # Evitar chunks demasiado pequeños
        if chunk_text and len(chunk_tokens) > 10:
            chunks.append(
                {
                    "text": chunk_text,
                    "token_count": len(chunk_tokens),
                    "start_token": start,
                    "end_token": end,
                }
            )

        if end == total_tokens:
            break

        # Avanzar con solapamiento
        start = max(start + max_length - overlap, end - min(overlap, 50))

    return chunks


def generate_chunks_csv():
    """Genera el CSV de chunks que tu script de ingesta espera."""
    print(f"Leyendo: {INPUT_CSV}")

    try:
        df = pd.read_csv(INPUT_CSV, encoding="utf-8")
    except FileNotFoundError:
        print(f"Archivo no encontrado: {INPUT_CSV}")
        print(f"Asegúrate de que existe en: {INPUT_CSV.absolute()}")
        return None

    # Validar columnas mínimas
    if "content" not in df.columns:
        print("El CSV debe contener columna 'content'")
        return None

    # Si no tiene doc_id, crear uno
    if "doc_id" not in df.columns:
        df["doc_id"] = [f"doc_{i}" for i in range(len(df))]

    print(f"Documentos a procesar: {len(df)}")

    # Cargar tokenizador
    _, tokenizer = load_tokenizer()

    all_chunks = []

    for idx, row in df.iterrows():
        if idx % 50 == 0:
            print(f" Procesando {idx+1}/{len(df)}...")

        original_doc_id = str(row["doc_id"])
        content = str(row["content"]) if pd.notna(row["content"]) else ""

        chunks = chunk_text_smart(content, tokenizer)

        for chunk_idx, chunk in enumerate(chunks):
            # 🧷 ID único por chunk
            chunk_doc_id = f"{original_doc_id}_chunk{chunk_idx:03d}"

            chunk_row: Dict[str, Any] = {
                "doc_id": chunk_doc_id,          # ID del chunk (lo usará Haystack)
                "content": chunk["text"],
                "data_type": "texto",            # requerido por tu ingesta
                "chunk_index": chunk_idx,
                "token_count": chunk["token_count"],
                "original_doc_id": original_doc_id,  # referencia al doc original
            }

            # Preservar todas las columnas originales
            # OJO: no tocamos 'content' ni 'doc_id', ya los hemos gestionado
            for col in df.columns:
                if col in ["content", "doc_id"]:
                    continue
                chunk_row[col] = row[col]

            all_chunks.append(chunk_row)

    # Crear DataFrame final
    df_chunks = pd.DataFrame(all_chunks)

    # Reordenar columnas para legibilidad
    priority_cols = [
        "doc_id",
        "content",
        "data_type",
        "chunk_index",
        "token_count",
        "original_doc_id",
    ]
    other_cols = [col for col in df_chunks.columns if col not in priority_cols]
    df_chunks = df_chunks[priority_cols + other_cols]

    # Guardar
    print(f"\nGuardando: {OUTPUT_CSV}")
    df_chunks.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    # Reporte
    print("\n" + "=" * 50)
    print(" CHUNKING COMPLETADO")
    print("=" * 50)
    print(f" Documentos originales: {len(df)}")
    print(f" Chunks generados: {len(df_chunks)}")
    print(f" Tokens por chunk (promedio): {df_chunks['token_count'].mean():.1f}")
    print(f" Archivo generado: {OUTPUT_CSV}")

    print("\nVISTA PREVIA:")
    print(df_chunks[["doc_id", "data_type", "token_count"]].head())

    return df_chunks


if __name__ == "__main__":
    chunks_df = generate_chunks_csv()
    if chunks_df is not None:
        print("\n¡Chunks generados exitosamente!")
