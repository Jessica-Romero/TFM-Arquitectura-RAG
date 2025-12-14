from pathlib import Path
import pandas as pd


def limpiar_waypoints_tablas(input_csv: str | Path, output_csv: str | Path):
    """
    Limpia el CSV de tablas de waypoints generado con Camelot.

    Espera columnas:
      - route_id
      - pdf_path
      - table_idx
      - 0,1,2,3,4,5 (contenido de la tabla)

    Hace:
      - Para cada (route_id, pdf_path, table_idx), elimina las 2 primeras filas
        (heading 'PUNTOS IMPORTANTES DE PASO' y cabecera de columnas).
      - Elimina filas que contengan 'Sistema de coordenadas geográficas'
        o 'Datum WGS84' en cualquier columna.
    """
    input_csv = Path(input_csv)
    output_csv = Path(output_csv)

    df = pd.read_csv(input_csv, encoding="utf-8", on_bad_lines="skip")

    # Comprobar columnas requeridas
    required_cols = {"route_id", "pdf_path", "table_idx"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"El CSV debe contener las columnas: {required_cols}. "
                         f"Se han encontrado: {list(df.columns)}")

    # Índice dentro de cada tabla (0,1,2,...) según orden de aparición
    df["row_in_table"] = (
        df.groupby(["route_id", "pdf_path", "table_idx"])
          .cumcount()
    )

    # Nos quedamos SOLO con filas a partir de la tercera (índice >= 2)
    df = df[df["row_in_table"] >= 2].copy()

    # Eliminar filas con 'Sistema de coordenadas geográficas' o 'Datum WGS84'
    def fila_no_es_coordenadas(row):
        for v in row:
            v_str = str(v).lower()
            if "sistema de coordenadas geográficas" in v_str:
                return False
            if "datum wgs84" in v_str:
                return False
        return True

    mask = df.apply(fila_no_es_coordenadas, axis=1)
    df = df[mask].copy()

    # Eliminar la columna auxiliar
    df = df.drop(columns=["row_in_table"])

    # Guardar resultado
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8")

    print(f"CSV limpio guardado en: {output_csv}")
    print("Filas finales:", len(df))


if __name__ == "__main__":
    input_csv = r".\rutas_waypoints_raw.csv"       # tu CSV actual con todo
    output_csv = r".\rutas_waypoints.csv"   # salida limpia

    limpiar_waypoints_tablas(input_csv, output_csv)
