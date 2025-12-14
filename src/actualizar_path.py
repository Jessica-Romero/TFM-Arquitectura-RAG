import pandas as pd
from pathlib import Path

def reemplazar_ruta_pdf(input_csv_path: str,
                        output_csv_path: str | None = None) -> None:
    """
    Reemplaza la ruta base de los archivos PDF en la columna 'file_path_img' de un CSV.
    """
    #nueva_base = Path(r"D:\TFM\TFM-Arquitectura-RAG\TFM-Arquitectura-RAG\imagenes_extraidas\\")
    #nueva_base = Path(r"D:\TFM\TFM-Arquitectura-RAG\TFM-Arquitectura-RAG\data\rutas_img")
    nueva_base = Path(r"D:\TFM\TFM-Arquitectura-RAG\TFM-Arquitectura-RAG\data\rutas_pdf")

    df = pd.read_csv(input_csv_path, encoding="utf-8")

    cols_to_drop = ["prioridad", "pdfExists"]
    df = df.drop(columns=cols_to_drop, errors="ignore")

    df.insert(0, "route_id", df.index.astype(int))

    if "pdf_path" not in df.columns:
        raise ValueError("La columna 'pdf_path' no existe en el CSV")

    # Guardamos solo el nombre del archivo
    df["pdf_path"] = df["pdf_path"].apply(
        lambda ruta: str(nueva_base / Path(ruta).name)
    )

    if output_csv_path is None:
        output_csv_path = input_csv_path

    df.to_csv(output_csv_path, index=False, encoding="utf-8")
    print(f"Rutas actualizadas guardadas en: {output_csv_path}")

if __name__ == "__main__":
    reemplazar_ruta_pdf(".\\rutas_pirineos_actualizado.csv", ".\\rutas.csv")