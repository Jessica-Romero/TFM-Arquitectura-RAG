import fitz  # PyMuPDF
import os
import pandas as pd

# Ruta del PDF
# pdf_path = r"c:\users\jessi\onedrive\escritorio\máster-uoc\tfm\tfm-arquitectura-rag\tfm-arquitectura-rag\download_pdf\rutas-pirineos-agulles-frares-encantats-montserrat-can-massana_es.pdf"
#pdf_path = r"c:\users\jessi\onedrive\escritorio\máster-uoc\tfm\tfm-arquitectura-rag\tfm-arquitectura-rag\download_pdf\rutas-pirineos-aiguaneix-alinya-urgell_es.pdf"

def extraer_imagenes_pdf(file_path: str, output_dir: str = None):

    df_img_rutas = pd.read_csv(file_path)
    # Lista para almacenar la trazabilidad
    trazabilidad = []

    # Iterar el dataframe y procesar el PDF correspondiente
    for index, row in df_img_rutas.iterrows():
        pdf_path = row["pdf_path"]
        print(f"\nProcesando PDF ({index + 1}/{len(df_img_rutas)}): {pdf_path}")
        
        imagenes_guardadas = procesar_pdf(pdf_path, output_dir)
        
        # Crear registro de trazabilidad para este PDF
        for img_path in imagenes_guardadas:
            trazabilidad.append({
                "indice": index,
                "file_path_img": img_path
            })
    
    # Guardar CSV de trazabilidad
    df_trazabilidad = pd.DataFrame(trazabilidad)
    trazabilidad_path = os.path.join(output_dir, ".\\trazabilidad_imagenes.csv")
    df_trazabilidad.to_csv(trazabilidad_path, index=False)
    print(f"\nCSV de trazabilidad guardado en: {trazabilidad_path}")
    
    return df_trazabilidad

def procesar_pdf(pdf_path: str, output_dir: str = None):
    """
    Procesa un PDF y devuelve lista de rutas de imágenes guardadas
    """
    # Carpeta de salida para las imágenes
    os.makedirs(output_dir, exist_ok=True)

    imagenes_guardadas = []

    # Abrir el documento
    doc = fitz.open(pdf_path)

    print(f"PDF abierto: {pdf_path}")
    print(f"Número de páginas: {doc.page_count}")

    pdf_base = os.path.splitext(os.path.basename(pdf_path))[0]
    num_paginas = doc.page_count
    ultima_pagina_index = num_paginas - 1

    # 1) DETECTAR PÁGINA DEL MAPA (imagen más grande)
    best_page_index = None
    best_image_area = 0

    for page_index in range(num_paginas):
        page = doc[page_index]
        image_list = page.get_images(full=True)

        print(f"Página {page_index + 1}: {len(image_list)} imagen(es) encontrada(s)")

        for img in image_list:
            # img = (xref, smask, width, height, bpc, colorspace, alt, name, filter)
            width = img[2]
            height = img[3]
            area = width * height

            if area > best_image_area:
                best_image_area = area
                best_page_index = page_index

    print("\n--- DETECCIÓN DEL MAPA ---")
    if best_page_index is not None:
        print(f"Página detectada como mapa: {best_page_index + 1} (imagen más grande: {best_image_area} px²)")
    else:
        print("No se han encontrado imágenes en el PDF.")

    # 2) EXTRAER IMÁGENES, Sin incorporar MAPA y la última página (suele tener logos u otros elementos no deseados).
    imagenes_vistas = set()

    for page_index in range(num_paginas):
        # Saltar la última página
        if page_index == ultima_pagina_index:
            print(f"\n Saltando extracción de imágenes de la última página: {page_index + 1}")
            continue
        # Saltar la página del mapa (para no extraer su imagen base)
        if best_page_index is not None and page_index == best_page_index:
            print(f"\n Saltando extracción de imágenes de la página del mapa: {page_index + 1}")
            continue

        page = doc[page_index]
        image_list = page.get_images(full=True)

        print(f"\n Extrayendo imágenes de la página {page_index + 1}: {len(image_list)} imagen(es)")

        for img_index, img in enumerate(image_list, start=1):
            xref = img[0]

            # Evitar duplicados (misma imagen usada en varias páginas)
            if xref in imagenes_vistas:
                continue
            imagenes_vistas.add(xref)

            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]

            image_name = f"{pdf_base}_p{page_index + 1}_img{img_index}.{image_ext}"
            image_path = os.path.join(output_dir, image_name)
            with open(image_path, "wb") as f:
                f.write(image_bytes)

            print(f" Imagen guardada: {image_path}")
            imagenes_guardadas.append(image_path)  # Agregar a la lista de trazabilidad



    # 3) EXPORTAR SOLO EL MAPA (SIN LA TABLA)
    print("\n--- EXPORTACIÓN DEL MAPA SIN TABLA DE WAYPOINTS ---")
    if best_page_index is not None:
        page = doc[best_page_index]

        # --- Buscar el bloque de texto del título "PUNTOS IMPORTANTES DE PASO..." ---
        blocks = page.get_text("blocks")  # (x0, y0, x1, y1, text, ...)
        cutoff_y = None

        # Patrones que pueden aparecer como título de la sección de waypoints
        patrones_titulo = [
            "PUNTOS IMPORTANTES DE PASO (WAYPOINTS)"
        ]

        for b in blocks:
            x0, y0, x1, y1, text = b[:5]
            texto_norm = text.upper().strip()

            if any(pat in texto_norm for pat in patrones_titulo):
                cutoff_y = y0  # inicio del título
                break

        if cutoff_y is None:
            # Si no encontramos el título, hacemos fallback al recorte completo
            print("\n No se encontró el título de 'PUNTOS IMPORTANTES DE PASO (WAYPOINTS)'; se exportará la página completa del mapa.")
            rect_mapa = page.rect
        else:
            # Rectángulo: desde arriba de la página hasta justo antes del título
            margen = 5  # ajusta si quieres más o menos espacio por abajo
            rect_mapa = fitz.Rect(
                page.rect.x0,
                page.rect.y0,
                page.rect.x1,
                cutoff_y - margen
            )
            print(f"\n Cortando el mapa en y = {cutoff_y:.2f} (por encima del título de los waypoints).")

        # Renderizar solo esa zona
        zoom = 2.0  # más calidad
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, clip=rect_mapa)

        map_output = os.path.join(output_dir, f"{pdf_base}_mapa.png")
        pix.save(map_output)

        print(f"\n Mapa exportado como imagen (sin tabla de waypoints):")
        print(f"   {map_output}")
        imagenes_guardadas.append(map_output)  # Agregar a la lista de trazabilidad

    else:
        print("\n No se pudo exportar el mapa porque no se detectó ninguna página con imágenes.")

    doc.close()
    print("\n Proceso completado con éxito.")
    
    return imagenes_guardadas  # Devolver lista de imágenes guardadas




if __name__ == "__main__":
    df_trazabilidad = extraer_imagenes_pdf(
        file_path=r".\rutas_pirineos_limpio_ve2.csv",
        output_dir=r"..\imagenes_extraidas"
    )
    print("\nTrazabilidad de imágenes:")
    print(df_trazabilidad)