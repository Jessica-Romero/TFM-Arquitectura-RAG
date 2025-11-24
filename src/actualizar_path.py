import pandas as pd
from pathlib import Path

def reemplazar_ruta_pdf(input_csv_path: str,
                        output_csv_path: str | None = None) -> None:
    """
    Reemplaza la ruta base de los archivos PDF en la columna 'file_path_img' de un CSV.
    """
    nueva_base = Path(r"D:\TFM\TFM-Arquitectura-RAG\TFM-Arquitectura-RAG\imagenes_extraidas\\")

    df = pd.read_csv(input_csv_path, encoding="utf-8")

    if "file_path_img" not in df.columns:
        raise ValueError("La columna 'file_path_img' no existe en el CSV")

    # Guardamos solo el nombre del archivo
    df["file_path_img"] = df["file_path_img"].apply(
        lambda ruta: str(nueva_base / Path(ruta).name)
    )

    if output_csv_path is None:
        output_csv_path = input_csv_path

    df.to_csv(output_csv_path, index=False, encoding="utf-8")
    print(f"Rutas actualizadas guardadas en: {output_csv_path}")

if __name__ == "__main__":
    reemplazar_ruta_pdf(".\\trazabilidad_imagenes.csv", ".\\trazabilidad_imagenes_actualizado.csv")