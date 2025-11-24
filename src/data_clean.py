import os, re, unicodedata, glob
import pandas as pd
from datetime import datetime
from pathlib import Path

def load_and_clean_data(file_path: str) -> pd.DataFrame:
    """
    Carga y limpia los datos de un archivo CSV.
    - file_path: ruta al archivo CSV
    Retorna un DataFrame limpio.
    """
    # Cargar datos
    df_rutas = pd.read_csv(file_path)

 
    # Número de datos que hay cargados
    print("Datos cargados:", len(df_rutas))

    # Mostrar columnas y primeras filas para inspección inicial
    print("Columnas del DataFrame:", df_rutas.columns.tolist())

    print("Primeras filas del DataFrame:", df_rutas.head())

   # Comprobar duplicados por enlace 
    duplicated_count = df_rutas.duplicated(subset=["Link archivo"]).sum()
    print("Duplicados por 'Link archivo':", duplicated_count)

    # Mostrar filas duplicadas si existen - Mostrar los campos de Nombre de la ruta , link archivo y Pdf_path
    if duplicated_count > 0:
        print("Filas duplicadas agrupadas por 'Link archivo':")
        dup_links = df_rutas[df_rutas.duplicated(subset=["Link archivo"], keep=False)]
        grouped = dup_links.groupby("Link archivo")[["Nombre de la ruta", "Pdf_path"]].agg(list)
        print(grouped)

    df_rutas["Pdf_path"] = df_rutas["Pdf_path"].astype(str).str.strip()
    pdf_path_lower = df_rutas["Pdf_path"].str.lower()

    # Prioridad: primero las filas que NO tengan Pdf_path tipo "true"/"false"
    df_rutas_sorted = (
        df_rutas.assign(Prioridad=pdf_path_lower.eq("true") | pdf_path_lower.eq("false"))
        .sort_values(by=["Link archivo", "Prioridad"])
    )

    df_rutas_ve = df_rutas_sorted.drop_duplicates(subset=["Link archivo"], keep="first")

    print("Número de filas después de eliminar los duplicados erróneos:", len(df_rutas_ve))

    # Sustituir los valores con pdf_path 'true' o 'false' por rutas correctas
    DOWNLOAD_DIR = 'C:\\Users\\jessi\\OneDrive\\Escritorio\\MÁSTER-UOC\\TFM\\TFM-Arquitectura-RAG\\TFM-Arquitectura-RAG\\download_pdf'


    def corregir_pdf_path(row):
        pdf_path = str(row["Pdf_path"]).strip()
        pdf_lower = pdf_path.lower()
        nombre_ruta = row["Link archivo"].split('/')[-1].replace('.html', '')
        if (
            pdf_lower in ("true", "false", "nan", "", "none")
            or ".pdf" not in pdf_lower
        ):
            # Normalizar el nombre de la ruta
            nombre_normalizado = unicodedata.normalize('NFKD', nombre_ruta)\
                                            .encode('ASCII', 'ignore')\
                                            .decode('utf-8')
            nombre_normalizado = re.sub(r'\s+', '_', nombre_normalizado.strip().lower())
            nueva_ruta = f"{DOWNLOAD_DIR}\\RUTAS-PIRINEOS-{nombre_normalizado}_es.pdf"
            return nueva_ruta

        # Si parece una ruta válida a PDF, se deja como está
        return pdf_path

    df_rutas_ve["Pdf_path"] = df_rutas_ve.apply(corregir_pdf_path, axis=1)
    print("Primeras filas tras corregir 'Pdf_path':", df_rutas_ve.head())
    # Verificar existencia física de los archivos PDF

    # Crear una nueva columna que indique si el archivo existe físicamente
    df_rutas_ve["PdfExists"] = df_rutas_ve["Pdf_path"].apply(lambda p: os.path.exists(p) if isinstance(p, str) else False)

    # Contar resultados
    total = len(df_rutas_ve)
    existen = df_rutas_ve["PdfExists"].sum()
    faltan = total - existen

    print(f"Total de rutas en el dataset: {total}")
    print(f"PDFs encontrados en carpeta: {existen}")
    print(f"PDFs faltantes o no descargados: {faltan}")

    # Mostrar algunos ejemplos de los que faltan
    if faltan > 0:
        print("\nEjemplos de rutas con PDF faltante:")
        print(df_rutas_ve.loc[~df_rutas_ve["PdfExists"], ["Nombre de la ruta", "Pdf_path","Link archivo"]].head(12))

    # Comprobación de qué columnas tienen valor nulos o vacíos agrupados por la columna categorias 
    print("\nComprobación de valores nulos o vacíos por columna:")
    for col in df_rutas_ve.columns: 
        nulos = df_rutas_ve[col].isnull().sum()
        vacios = (df_rutas_ve[col].astype(str).str.strip() == "").sum()
        if nulos > 0 or vacios > 0:
            print(f"Columna '{col}': {nulos} nulos, {vacios} vacíos")

    # Comprobación de qué columnas tienen valor nulos o vacíos agrupados por la columna categorias 
    print("\nComprobación de valores nulos o vacíos por columna y categoría:")
    for col in df_rutas_ve.columns: 
        nulos_por_cat = df_rutas_ve[df_rutas_ve[col].isnull()].groupby("Categoria").size()
        vacios_por_cat = df_rutas_ve[df_rutas_ve[col].astype(str).str.strip() == ""].groupby("Categoria").size()
        if not nulos_por_cat.empty:
            print(f"\nColumna '{col}' - Nulos por categoría:")
            print(nulos_por_cat)
        if not vacios_por_cat.empty:
            print(f"\nColumna '{col}' - Vacíos por categoría:")
            print(vacios_por_cat)

    # Se ha visto que las categorías Movilidad reducida - Rutas por etapas y GRs tienen varios nulos en varias columnas. 
    # Se quiere inspeccionar el resto de filas que no pertenezcan a estas categorías. Se debe mostrar el link archivo y nombre de la ruta y las columnas que tienen nulos
    categorias_a_inspeccionar = ["Movilidad reducida", "Rutas por etapas y GRs"]

    # Se ha visto que las categorías Movilidad reducida - Rutas por etapas y GRs tienen varios nulos en varias columnas. No se quiere inspeccionar estas rutas
    # Se va comprobar si del resto de rutas que no pertenecen a estas categorías tienen nulos en alguna columna.
    # Si es así, se mostrarán los links de archivo y nombre de la ruta junto con las columnas que tienen nulos
    # Se ha visto que algunas de las filas que tienen nulos en algunas columnas son del pais de Andorra y está bien. Se quiere descartar estas filas también
    # Se ha comprobado que las rutas que pertenecen a Francia no tienen específico el pueblo. Es correcto, no lo especifica la ruta.
    print("\nComprobación de nulos en filas fuera de las categorías inspeccionadas y fuera de Andorra y Francia:")
    for col in df_rutas_ve.columns:
        nulos = df_rutas_ve[~df_rutas_ve["Categoria"].isin(categorias_a_inspeccionar) &
                            (~df_rutas_ve["Pais"].isin(["Andorra","Francia"])) 
                ][col].isnull().sum()
        if nulos > 0:
            print(f"!!!Columna '{col}' fuera de las categorías inspeccionadas: {nulos} nulos")     
            print(df_rutas_ve[~df_rutas_ve["Categoria"].isin(categorias_a_inspeccionar) &
                               (~df_rutas_ve["Pais"].isin(["Andorra","Francia"])) &
                                                          df_rutas_ve[col].isnull()]
                                                          
                                                          [["Link archivo","Pais" , "Region","Nombre de la ruta"]].head(10))


    # Carpeta donde guardar el CSV limpio
    OUTPUT_DIR = r"C:\Users\jessi\OneDrive\Escritorio\MÁSTER-UOC\TFM\TFM-Arquitectura-RAG\TFM-Arquitectura-RAG\src"
    OUTPUT_FILE = os.path.join(OUTPUT_DIR, "rutas_pirineos_limpio_ve2.csv")

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # Guardar el DataFrame en CSV
    df_rutas_ve.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    print(f"Archivo guardado correctamente en:\n{OUTPUT_FILE}")
    print(f"Total de filas guardadas: {len(df_rutas_ve)}")

  

if __name__ == "__main__":
    df = load_and_clean_data(file_path="./rutas.csv")