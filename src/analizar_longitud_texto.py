import pandas as pd
from pathlib import Path

def analizar_longitud_texto(csv_path: str, limite_chunk: int = 800):
    """
    Analiza la longitud de los textos en un CSV con columna 'content'.
    Muestra:
        - Estadísticas generales
        - Top textos más largos
        - Recomendaciones de chunking
    """

    df = pd.read_csv(csv_path, encoding="utf-8")

    # Longitud en caracteres
    df["len_chars"] = df["content"].astype(str).apply(len)

    # Longitud aproximada en tokens (estimado: 1 token ~ 4 caracteres)
    df["len_tokens"] = (df["len_chars"] / 4).astype(int)

    print("\n Estadísticas generales ===")
    print(df["len_chars"].describe())

    print("\n Estadísticas en tokens (estimado) ===")
    print(df["len_tokens"].describe())

    # Detectar textos por encima del límite elegido
    df_largos = df[df["len_tokens"] > limite_chunk]

    print(f"\n Textos que superan {limite_chunk} tokens ===")
    print(f"Nº textos largos: {len(df_largos)}")

    if len(df_largos) > 0:
        print("\nEjemplos de textos largos:")
        print(df_largos[["doc_id", "len_tokens", "len_chars"]].head(10))

    # Recomendación automática
    print("\n Recomendación ")
    if df["len_tokens"].max() <= limite_chunk:
        print(" No necesitas chunking. Todos los textos están dentro del límite.")
    elif df["len_tokens"].max() <= limite_chunk * 2:
        print(" Algunos textos son largos, pero manejables con cuidad — Podrías usar chunking suave.")
    else:
        print(" Hay textos MUY largos — recomendamos dividirlos en chunks.")
    return df


if __name__ == "__main__":
    CSV_PATH = "./rag_docs/docs_sections_v2.csv"  
    analizar_longitud_texto(CSV_PATH, limite_chunk=800)
