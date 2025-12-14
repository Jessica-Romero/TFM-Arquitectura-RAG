from pathlib import Path
import pandas as pd


def build_docs_waypoints() -> None:
    """
    Genera los documentos de waypoints (tablas) para el RAG.

    Lee:
      - data/staged/rutas.csv           (metadata de rutas)
      - data/staged/rutas_waypoints.csv (tabla limpia de waypoints)

    Escribe:
      - data/rag_docs/docs_waypoints.csv
        con columnas: doc_id, content, meta_*, data_type="waypoint"
    """

    # 1) Carga de ficheros
    repo_root = Path(__file__).resolve().parents[2]
    data_dir = repo_root / "src"

    staged_dir = data_dir / "staged"
    rag_docs_dir = data_dir / "rag_docs"
    rag_docs_dir.mkdir(parents=True, exist_ok=True)

    rutas_csv = staged_dir / "rutas_v2.csv"
    waypoints_csv = staged_dir / "rutas_waypoints.csv"
    output_csv = rag_docs_dir / "docs_waypoints_v3.csv"

    # 2) Cargar datasets

    df_rutas = pd.read_csv(rutas_csv, encoding="utf-8", on_bad_lines="skip")

    COLS_WAYPOINTS = [
        "route_id", "pdf_path", "table_idx", "punto_de_paso", 
        "tiempo", "altura", "latitud", "longitud"
    ]

    # Cargar df_waypoints usando usecols
    df_wp = pd.read_csv(
        waypoints_csv, 
        encoding="utf-8", 
        on_bad_lines="skip",
        usecols=COLS_WAYPOINTS 
    )

    df_wp["route_id"] = df_wp["route_id"].astype("int64")
    df_rutas["route_id"] = df_rutas["route_id"].astype("int64")

    # filtramos aquellos que no tengan punto de paso ni tiempo 
    df_wp = df_wp[
        df_wp["tiempo"].notna() & (df_wp["tiempo"].str.strip() != "")
    ].copy()

    # 3) Se crea una columna orden

    df_wp["wp_orden"] = df_wp.groupby("route_id").cumcount() + 1

    # 4) Enriquecer con metadatos de rutas

    df_merged = df_wp.merge(
        df_rutas,
        on="route_id",
        how="inner",
        suffixes=("", "_ruta"),
    )


    # 5) Construir documentos RAG para cada waypoint

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
        orden = int(row["wp_orden"])

        punto = str(row.get("punto_de_paso", "") or "").strip()
        tiempo = str(row.get("tiempo", "") or "").strip()
        altura = str(row.get("altura", "") or "").strip()
        lat = row.get("latitud")
        lon = row.get("longitud")
        table_idx = row.get("table_idx", None)

        # Saltar filas totalmente vacías de punto de paso
        if not punto and not tiempo and not altura:
            continue

        # doc_id estable: route_id + número de waypoint
        doc_id = f"{route_id}_wp-{orden}"

        # Construir el content textual que se va a embeder
        parts = []

        if punto:
            parts.append(f"Waypoint '{punto}'")

        if tiempo:
            parts.append(f"tiempo {tiempo}")

        if altura:
            parts.append(f"altura {altura}")

        if pd.notna(lat) and pd.notna(lon):
            parts.append(f"coordenadas {lat}, {lon}")

        # Si por lo que sea no se añadió nada, lo saltamos
        if not parts:
            continue

        content = ", ".join(parts) + "."

        # Metadatos

        doc = {
            "doc_id": doc_id,
            "content": content,
            "data_type": "waypoint",
            "meta_route_id": route_id,
            "meta_wp_orden": orden,
            "meta_punto_de_paso": punto,
            "meta_tiempo": tiempo,
            "meta_altura": altura,
            "meta_latitud": lat,
            "meta_longitud": lon,
            "meta_table_idx": table_idx,
        }

        # metadatos de la ruta
        for col in meta_cols_presentes:
            doc[f"meta_{col}"] = row[col]

        registros.append(doc)

    df_out = pd.DataFrame(registros)

    # 6) Guardar
    df_out.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"Docs_waypoints generados en: {output_csv}")
    print(f"Nº documentos: {len(df_out)}")


if __name__ == "__main__":
    build_docs_waypoints()
