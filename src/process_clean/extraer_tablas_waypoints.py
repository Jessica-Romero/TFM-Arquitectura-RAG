from pathlib import Path
import camelot
import pandas as pd


def es_tabla_waypoints(df: pd.DataFrame) -> bool:
    """
    Devuelve True si la tabla tiene 4 columnas o más (típico de waypoints) 
    y contiene al menos una mención de 'tiempo' en las primeras 5 filas.
    """
    if df.empty or df.shape[1] < 4:
        return False
    
    # 1. Verificar la cantidad de columnas (una tabla de waypoints suele ser ancha)
    # 2. Buscar palabras clave en las primeras 5 filas (para capturar el encabezado o los datos)
    
    df_search = df.head(5).astype(str)
    
    # Concatenar todas las celdas de las primeras 5 filas y buscar palabras clave
    texto_busqueda = " ".join(df_search.values.flatten()).lower()
    
    # Verificación: busca los términos clave para confirmar que es una tabla de waypoints
    es_waypoints = "punto de paso" in texto_busqueda or "desnivel" in texto_busqueda
    es_tiempo = "tiempo" in texto_busqueda or "h" in texto_busqueda
    
    # Buscamos la presencia de las dos características más importantes de la tabla
    return es_waypoints and es_tiempo


def procesar_pdfs_waypoints_simple(
    rutas_csv_path: str | Path,
    col_pdf_path: str,
    output_csv_path: str | Path,
):
    """
    Lee el CSV de rutas y,
    procesa TODOS los PDFs de la columna `col_pdf_path` y guarda SOLO
    las tablas cuya primera fila contiene 'Punto de paso' (waypoints).

    Las tablas se guardan tal cual las devuelve Camelot, añadiendo:
    - route_id
    - pdf_path
    - table_idx (índice de la tabla dentro del PDF)
    """
    rutas_csv_path = Path(rutas_csv_path)
    output_csv_path = Path(output_csv_path)

    df_rutas = pd.read_csv(rutas_csv_path, encoding="utf-8", on_bad_lines="skip")

    tablas_acumuladas = []

    for idx, row in df_rutas.iterrows():
        pdf_path = row.get(col_pdf_path)

        if not isinstance(pdf_path, str) or not pdf_path.strip():
            print(f"[{idx}] Sin pdf_path, se salta la fila.")
            continue

        pdf_path = Path(pdf_path)

        if not pdf_path.is_file():
            print(f"[{idx}] NO se encuentra el PDF: {pdf_path}")
            continue

        route_id = int(idx)
        print(f"\n=== Procesando route_id={route_id} -> {pdf_path.name} ===")

        try:
            # Extraemos tablas de todas las páginas en modo 'stream'
            tablas = camelot.read_pdf(
                str(pdf_path),
                pages="all",
                flavor="stream",
            )
        except Exception as e:
            print(f"Error usando Camelot en {pdf_path.name}: {e}")
            continue

        if tablas.n == 0:
            print("Camelot no encontró ninguna tabla.")
            continue

        for t_idx, tabla in enumerate(tablas):
            df_tabla = tabla.df

            if not es_tabla_waypoints(df_tabla):
                continue

            print(f"Tabla de waypoints detectada (tabla #{t_idx}) en {pdf_path.name}")

            # Añadimos columnas de contexto
            df_out = df_tabla.copy()
            df_out.insert(0, "route_id", route_id)
            df_out.insert(1, "pdf_path", str(pdf_path))
            df_out.insert(2, "table_idx", t_idx)
            print(df_out.head(3))
            tablas_acumuladas.append(df_out)

    if not tablas_acumuladas:
        print("No se ha encontrado ninguna tabla de PUNTOS IMPORTANTES DE PASO.")
        return

    df_final = pd.concat(tablas_acumuladas, ignore_index=True)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(output_csv_path, index=False, encoding="utf-8")

    print(f"\n CSV de tablas de waypoints guardado en: {output_csv_path}")


if __name__ == "__main__":
    rutas_csv = r".\rutas.csv"        # tu CSV de rutas
    col_pdf_path = "pdf_path"                              # nombre de la columna con la ruta al PDF
    output_csv = r".\rutas_waypoints_raw.csv"     # salida con las tablas de waypoints

    procesar_pdfs_waypoints_simple(rutas_csv, col_pdf_path, output_csv)
