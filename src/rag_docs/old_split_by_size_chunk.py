from pathlib import Path
import pandas as pd
from typing import List
from transformers import AutoTokenizer


# Modelo de tokenización
model_name = "BAAI/bge-multilingual-large"

MAX_TOKENS = 400          # tokens por chunk
OVERLAP_TOKENS = 80       # tokens de solape


# Cargamos el tokenizador
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)


# ------------------------------------------------------
# 🔧 Función de chunking por tokens
# ------------------------------------------------------

def chunk_by_tokens(text: str, max_length: int = MAX_TOKENS, overlap: int = OVERLAP_TOKENS) -> List[str]:
    """
    Divide un texto en chunks por número de tokens usando un tokenizador HF.
    Devuelve chunks de texto real, reconstruidos por span_mapping.
    """

    if not isinstance(text, str):
        return []

    # Tokenizamos con mapeo de offsets → permite reconstruir trozos de texto
    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=False
    )

    token_offsets = encoding["offset_mapping"]
    tokens = encoding["input_ids"]
    n_tokens = len(tokens)

    if n_tokens == 0:
        return []

    chunks = []
    start = 0

    while start < n_tokens:
        end = min(start + max_length, n_tokens)

        # Determinar límites reales en caracteres
        char_start = token_offsets[start][0]
        char_end = token_offsets[end - 1][1]

        chunk_text = text[char_start:char_end]
        chunks.append(chunk_text.strip())

        if end == n_tokens:
            break

        # Avanzamos inicio considerando solape
        start = end - overlap

    return chunks


def main():
    base_dir = Path(__file__).resolve().parents[1]
    rag_docs_dir = base_dir / "rag_docs"

    input_csv = rag_docs_dir / "docs_sections_v3.csv"
    output_csv = rag_docs_dir / "docs_sections_chunks_tokens.csv"

    print(f"Leyendo secciones desde: {input_csv}")
    df = pd.read_csv(input_csv, encoding="utf-8")

    if "doc_id" not in df.columns or "content" not in df.columns:
        raise ValueError("El CSV debe contener 'doc_id' y 'content'.")

    new_rows = []

    for _, row in df.iterrows():
        original_doc_id = str(row["doc_id"])
        text = str(row["content"])

        chunks = chunk_by_tokens(text)

        for idx, ch in enumerate(chunks):
            new_row = row.copy()
            new_row["doc_id"] = f"{original_doc_id}_chunk-{idx}"
            new_row["content"] = ch
            new_row["chunk_index"] = idx

            new_rows.append(new_row)

    df_out = pd.DataFrame(new_rows)

    df_out.to_csv(output_csv, index=False, encoding="utf-8")

    print(f"\n Chunking por tokens completado")
    print(f"Secciones originales: {len(df)}")
    print(f"Chunks generados: {len(df_out)}")
    print(f"Guardado en: {output_csv}")


if __name__ == "__main__":
    main()
