from pathlib import Path
import pandas as pd


def es_imagen_mapa(img_path: str) -> bool:
    """
    Detecta si una imagen es un mapa en función del nombre de fichero.
    Regla simple: si es .png o el nombre contiene 'es_mapa' / 'mapa'.
    """
    name = Path(img_path).name.lower()
    return (
        name.endswith(".png")
        or "es_mapa" in name
        or "mapa" in name
    )


def build_docs_images() -> None:
    """
    Genera los documentos de imágenes para el RAG.

    Lee:
      - data/staged/rutas.csv            (metadata de rutas)
      - data/staged/rutas_imagenes.csv   (rutas e imágenes asociadas)

    Escribe:
      - data/rag_docs/docs_images.csv
        con columnas: doc_id, content, meta_*, data_type="image"
    """

    # 1) Carga de ficheros
    repo_root = Path(__file__).resolve().parents[2]
    data_dir = repo_root / "src"

    staged_dir = data_dir / "staged"
    rag_docs_dir = data_dir / "rag_docs"
    rag_docs_dir.mkdir(parents=True, exist_ok=True)

    rutas_csv = staged_dir / "rutas_v2.csv"
    imgs_csv = staged_dir / "rutas_imagenes.csv"   
    output_csv = rag_docs_dir / "docs_images_v3.csv"

    # 2) Cargar datasets
    df_rutas = pd.read_csv(rutas_csv, encoding="utf-8", on_bad_lines="skip")
    df_imgs = pd.read_csv(imgs_csv, encoding="utf-8", on_bad_lines="skip")

    # --- Validar columnas ---
    if "route_id" not in df_imgs.columns:
        raise ValueError(f"'route_id' no está en {imgs_csv}. Columnas: {df_imgs.columns.tolist()}")

    # nombre de la columna de path de imagen (intento flexible)
    col_img_path = "file_path_img"


    df_imgs["route_id"] = df_imgs["route_id"].astype("int64")
    df_rutas["route_id"] = df_rutas["route_id"].astype("int64")

    # 3) Crear un índice por imagen dentro de cada ruta (img_orden)
    df_imgs["img_orden"] = df_imgs.groupby("route_id").cumcount() + 1

    # 4) Merge imágenes + metadata de rutas
    df_merged = df_imgs.merge(
        df_rutas,
        on="route_id",
        how="left",
        suffixes=("", "_ruta"),
    )

    # 5) Construir documentos
    registros = []

    meta_cols_candidatas = [
        "pais",
        "region",
        "provincia",
        "comarca",
        "zona",
        "pueblo",
        "nombre_ruta",
        "categoria",
        "dificultad",
        "descripcion_dificultad",
        "distancia_total_km",
        "altitud_minima_m",
        "altitud_maxima_m",
        "desnivel_acumulado_pos",
        "desnivel_acumulado_neg",
        "tiempo_total_min",
        "Punto_salida_llegada",
        "poblacion_cercana",
        "link_archivo",
        "fecha",
    ]
    meta_cols_presentes = [c for c in meta_cols_candidatas if c in df_merged.columns]

    for _, row in df_merged.iterrows():
        route_id = int(row["route_id"])
        img_orden = int(row["img_orden"])
        img_path = str(row[col_img_path])

        nombre_ruta = str(row.get("nombre_ruta", "") or "").strip()
        pueblo = str(row.get("pueblo", "") or "").strip()
        zona = str(row.get("zona", "") or "").strip()
        comarca = str(row.get("comarca", "") or "").strip()
        provincia = str(row.get("provincia", "") or "").strip()
        region = str(row.get("region", "") or "").strip()
        pais = str(row.get("pais", "") or "").strip()

        # doc_id estable: route_id + número de imagen
        doc_id = f"{route_id}_img-{img_orden}"

        # contenido textual para embeddings (descripción sintética)
        partes_desc = []

          # Si es png / es_mapa / mapa → tratamos como mapa
        if es_imagen_mapa(img_path):
            partes_desc = []

            if nombre_ruta:
                partes_desc.append(f"Mapa cartográfico de la ruta '{nombre_ruta}'")
            else:
                partes_desc.append("Mapa cartográfico de la ruta")

            lugar_parts = [p for p in [pueblo, zona, comarca, provincia, region, pais] if p]
            if lugar_parts:
                partes_desc.append("ubicada en " + ", ".join(lugar_parts))

            partes_desc.append("representa el trazado de la ruta, su entorno y puntos de referencia")
            partes_desc.append(f"fichero de imagen: {img_path}")

            content = ". ".join(partes_desc) + "."
            is_map = True
        else:
            # Imagen "normal" (foto de paisaje, entorno, etc.)
            partes_desc = []

            if nombre_ruta:
                partes_desc.append(f"Imagen de la ruta '{nombre_ruta}'")
            else:
                partes_desc.append("Imagen asociada a una ruta de montaña")

            lugar_parts = [p for p in [pueblo, zona, comarca, provincia, region, pais] if p]
            if lugar_parts:
                partes_desc.append("ubicada en " + ", ".join(lugar_parts))

            partes_desc.append(f"fichero de imagen: {img_path}")

            content = ". ".join(partes_desc) + "."
            is_map = False

        doc = {
            "doc_id": doc_id,
            "content": content,
            "data_type": "image",
            "meta_route_id": route_id,
            "meta_img_orden": img_orden,
            "meta_img_path": img_path,
            "meta_is_map": is_map,
        }

        for col in meta_cols_presentes:
            doc[f"meta_{col}"] = row[col]
        registros.append(doc)

    df_out = pd.DataFrame(registros)
    df_out.to_csv(output_csv, index=False, encoding="utf-8")

    print(f"Docs_images generados en: {output_csv}")
    print(f"Nº documentos: {len(df_out)}")


if __name__ == "__main__":
    build_docs_images()
