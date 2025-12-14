from pathlib import Path
import pandas as pd


def build_docs_sections() -> None:
    """
    Genera los documentos de texto por secciones para el RAG.

    Lee:
      - data/staged/rutas.csv           (metadata de rutas)
      - data/staged/rutas_secciones.csv (texto por secciones)

    Descarta:
      - secciones cuyo heading sea
        'PUNTOS IMPORTANTES DE PASO (WAYPOINTS)' -- esta información está en los waypoints 

    Escribe:
      - data/rag_docs/docs_sections.csv
        con columnas: doc_id, content, meta_*, data_type
    """

    # 1) Carga de ficheros
    repo_root = Path(__file__).resolve().parents[2]
    data_dir = repo_root / "src"

    staged_dir = data_dir / "staged"
    rag_docs_dir = data_dir / "rag_docs"
    rag_docs_dir.mkdir(parents=True, exist_ok=True)

    rutas_csv = staged_dir / "rutas_v2.csv"
    secciones_csv = staged_dir / "rutas_secciones.csv"
    output_csv = rag_docs_dir / "docs_sections_v3.csv"

    # 2) Cargar datasets
    df_rutas = pd.read_csv(rutas_csv, encoding="utf-8", on_bad_lines="skip")
    df_secs = pd.read_csv(secciones_csv, encoding="utf-8", on_bad_lines="skip")


    df_secs["route_id"] = df_secs["route_id"].astype(int)

    # 3) Filtrar secciones: excluir WAYPOINTS
  
    df_secs["heading_norm"] = df_secs["heading"].astype(str).str.strip().str.upper()
    df_secs = df_secs[
        df_secs["heading_norm"] != "PUNTOS IMPORTANTES DE PASO (WAYPOINTS)"
    ].copy()
    df_secs = df_secs.drop(columns=["heading_norm"])


    # 4) Juntamos los metadatos de las rutas con las secciones.
    df_merged = df_secs.merge(
        df_rutas,
        on="route_id",
        how="left",
        suffixes=("", "_ruta"),
    )

    # 5) Construir documentos RAG
    
    registros = []

    # columnas de rutas que queremos como metadatos (si existen)
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
        section_id = row["section_id"]
        heading = str(row["heading"])
        texto = str(row.get("text", "") or "").strip()

        if not texto:
            continue  # saltamos secciones vacías

        # doc_id estable: route_id + section_id
        doc_id = str(section_id)

        # contenido para embeddings
        content = texto

        doc = {
            "doc_id": doc_id,
            "content": content,
            "data_type": "texto",
            "meta_route_id": route_id,
            "meta_section_id": section_id,
            "meta_heading": heading,
         
        }

        # añadimos metadatos de la ruta con prefijo meta_
        for col in meta_cols_presentes:
            doc[f"meta_{col}"] = row[col]

        registros.append(doc)

    df_out = pd.DataFrame(registros)

    # 6) Guardar

    df_out.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"docs_sections generados en: {output_csv}")
    print(f"Nº documentos: {len(df_out)}")


if __name__ == "__main__":
    build_docs_sections()
