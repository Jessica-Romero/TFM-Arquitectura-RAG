import unicodedata
import fitz  # PyMuPDF
import csv
import re
from pathlib import Path
import pandas as pd


# --- 1) Headings que queremos detectar 

HEADING_PATTERNS = {
    "LO MEJOR DE ESTA RUTA": "LO MEJOR DE ESTA RUTA",
    "INTRODUCCIÓN": "INTRODUCCIÓN",
    "MÁS INFORMACIÓN": "MÁS INFORMACIÓN",
    "PUNTOS IMPORTANTES DE PASO (WAYPOINTS)": "PUNTOS IMPORTANTES DE PASO (WAYPOINTS)",
    "RECORRIDO": "RECORRIDO",
    "¿SABÍAS QUE": "¿SABÍAS QUE...?",
    "CÓMO LLEGAR EN COCHE": "CÓMO LLEGAR EN COCHE",
    "NO TE PIERDAS": "NO TE PIERDAS...",
    "¿QUÉ MÁS PODEMOS VISITAR?": "¿QUÉ MÁS PODEMOS VISITAR?",
}


def detectar_heading_en_linea(linea: str):
    """
    Detecta headings SOLO si la línea ENTERA es un heading.
    No detecta headings incrustados en frases.
    """
    t = linea.strip().upper()

    for pattern, canonical in HEADING_PATTERNS.items():

        # 1) Coincidencia exacta
        if t == pattern:
            return canonical, "", ""

        # 2) Coincidencia exacta con signos (caso ¿SABÍAS QUE...? etc.)
        if t.startswith(pattern) and len(t) <= len(pattern) + 4:  
            # permite "¿SABÍAS QUE...?" pero no frases largas
            return canonical, "", ""

    return None, linea, ""

def es_pie_pagina(linea: str) -> bool:
    t = linea.strip()
    if not t:
        return False

    # Línea es número de página (1, 2, 3, 4...)
    if t.isdigit() and len(t) <= 3:
        return True

    l = t.lower()

    # URLs y emails
    if "rutaspirineos.org" in l:
        return True
    if "info@rutespirineus.cat" in l:
        return True
    if "rutespirineus" in l:
        return True
    if "rutaspirineos" in l:
        return True
    if "rutas pirineos" in l:
        return True
    
    # Copyright
    if "©" in l or "derechos reservados" in l:
        return True

    return False


def clean_text(text: str) -> str:
    """
    Normaliza espacios para evitar una palabra por línea:
    'metros\\nmás\\nadelante' -> 'metros más adelante'
    """
    return " ".join(text.split())

def limpiar_texto(text: str) -> str:
    """
    Limpieza específica de rarezas de PDF:
    - caracteres invisibles
    - área privada Unicode (iconos)
    - ligaduras (ﬁ, ﬂ)
    - soft hyphen
    - comillas y guiones tipográficos
    """
    if not isinstance(text, str):
        return text

    # 1) eliminar zero-width space
    text = text.replace("\u200b", " ")

    # 2) eliminar área privada Unicode (iconos PDF tipo   , etc.)
    text = re.sub(r"[\uf000-\uf0ff]", " ", text)

    # 3) reemplazar ligaduras PDF
    text = text.replace("ﬁ", "fi").replace("ﬂ", "fl")

    # 4) eliminar soft hyphen (cortes invisibles de palabra)
    text = text.replace("\xad", "")

    # 5) normalizar comillas y guiones especiales
    replacements = {
        "’": "'",
        "‘": "'",
        "“": '"',
        "”": '"',
        "–": "-",
        "…": "...",
        "´": "'",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)

    # 6) si aparecen caracteres corruptos tipo â, ô, ê, los quitamos
    text = re.sub(r"[âôê]", "", text)

    # 7) normalización Unicode NFC (acentos bien puestos)
    text = unicodedata.normalize("NFC", text)

    return text

import re

def limpiar_ruido_editorial(text: str) -> str:
    """
    Elimina ruido editorial repetitivo de Rutes Pirineus:
    - Disclaimers legales
    - Frases de sugerencias
    - Sello editorial
    - Frases sobre publicación y variabilidad de rutas/señalización
    - Coordenadas GPS del pie de página

    Respeta el resto del contenido útil.
    """
    if not isinstance(text, str):
        return text
    
    # Normalizar espacios no estándar
    text = text.replace("\u00a0", " ")  # espacio no separable

    patrones = [
        # 1) Sello editorial completo
        r"RUTESPIRINEUS Todas las rutas han sido realizadas sobre el terreno por RUTES PIRINEUS",
        r"Todas las rutas han sido realizadas sobre el terreno por RUTES PIRINEUS",

        r"RUTASPIRINEOS Todas las rutas han sido realizadas sobre el terreno por RUTAS PIRINEOS",
        r"Todas las rutas han sido realizadas sobre el terreno por RUTAS PIRINEOS",

        # 2) Frase de sugerencias (hasta el punto o salto de línea)
        r" Para cualquier sugerencia.*$",
        r"Para cualquier sugerencia.*$",
        # 3) Coordenadas GPS del pie (hasta final de línea o punto)
        r"Coordenadas\s+GPS[^0-9A-Za-záéíóúüñ]*[0-9º.,\s]+",

        # 4) Disclaimer legal
        r"RUTES ?PIRINEUS no se responsabiliza[^.\n]*[.\n]",
        r"RUTES PIRINEUS no se responsabiliza[^.\n]*[.\n]",
        r"RUTESPIRINEUS no se responsabiliza[^.\n]*[.\n]",

        r"RUTAS ?PIRINEOS no se responsabiliza[^.\n]*[.\n]",
        r"RUTAS PIRINEOS no se responsabiliza[^.\n]*[.\n]",
        r"RUTASPIRINEOS no se responsabiliza[^.\n]*[.\n]",

        # 5) Disclaimer sobre publicación y variabilidad
        r"Esta guía web y PDF gratuita ha sido publicada[^.\n]*[.\n]",
    ]

    for pat in patrones:
        text = re.sub(pat, " ", text, flags=re.IGNORECASE)

    # Limpiar espacios múltiples y espacios antes de puntuación
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\s+([.,;:])", r"\1", text)
    text = text.strip()

    return text


def extraer_secciones_pdf(pdf_path: str | Path):
    pdf_path = Path(pdf_path)
    doc = fitz.open(pdf_path)

    secciones = []

    # --- 1ª sección: SOLO página 1 (índice 0) -> FICHA TÉCNICA ---
    ficha_text = ""
    if len(doc) > 0:
        page0 = doc[0]
        data0 = page0.get_text("dict")
        blocks0 = data0.get("blocks", [])

        ficha_lines = []
        for b in blocks0:
            if b.get("type") != 0:  # solo texto
                continue
            for line in b.get("lines", []):
                linea_texto = "".join(
                    span.get("text", "")
                    for span in line.get("spans", [])
                ).strip()

                if not linea_texto:
                    continue

                # aquí sí aplicamos el filtro de pie de página
                if es_pie_pagina(linea_texto):
                    continue

                ficha_lines.append(linea_texto)

        ficha_text = " ".join(ficha_lines)

    secciones.append({
        "heading": "FICHA TÉCNICA",
        "text": ficha_text.strip(),
    })

    # --- Resto de páginas igual que antes excluimos última página ---
    seccion_actual = None

    for page_idx in range(1, len(doc)-1):
        page = doc[page_idx]
        data = page.get_text("dict")
        blocks = data.get("blocks", [])

        for b in blocks:
            if b.get("type") != 0:
                continue

            for line in b.get("lines", []):
                linea_texto = "".join(
                    span.get("text", "")
                    for span in line.get("spans", [])
                ).strip()

                if not linea_texto:
                    continue

                if es_pie_pagina(linea_texto):
                    continue

                heading, before, after = detectar_heading_en_linea(linea_texto)

                if heading is not None:
                    if before:
                        if seccion_actual is None:
                            secciones[-1]["text"] += " " + before
                        else:
                            seccion_actual["text"] += " " + before

                    if seccion_actual and seccion_actual["text"].strip():
                        secciones.append(seccion_actual)

                    seccion_actual = {
                        "heading": heading,
                        "text": "",
                    }

                    if after:
                        seccion_actual["text"] += " " + after

                else:
                    if seccion_actual is None:
                        secciones[-1]["text"] += " " + linea_texto
                    else:
                        seccion_actual["text"] += " " + linea_texto

    if seccion_actual and seccion_actual["text"].strip():
        secciones.append(seccion_actual)

    # limpieza final
    for sec in secciones:
        txt = sec["text"]
        txt = clean_text(txt)     # normaliza espacios
        txt = limpiar_texto(txt)  # limpia rarezas PDF
        txt = limpiar_ruido_editorial(txt)  # elimina ruido editorial específico
        sec["text"] = txt

    return secciones



def procesar_todos_los_pdfs(rutas_csv_path: str | Path,
                            col_pdf_path: str,
                            output_csv_path: str | Path):
    """
    Lee rutas_pirineos_actualizado.csv, procesa TODOS los PDFs de la columna `col_pdf_path`
    y guarda todas las secciones en un único CSV.

    route_id = índice (0, 1, 2, ...) del CSV de rutas.
    """
    rutas_csv_path = Path(rutas_csv_path)
    output_csv_path = Path(output_csv_path)

    df_rutas = pd.read_csv(rutas_csv_path)

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with output_csv_path.open("w", newline="", encoding="utf-8") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(["route_id", "section_id", "heading", "text"])

        for idx, row in df_rutas.iterrows():
            pdf_path = row[col_pdf_path]

            if not isinstance(pdf_path, str) or not pdf_path.strip():
                print(f"[{idx}] Sin pdf_path, se salta la fila.")
                continue

            pdf_path = Path(pdf_path)

            if not pdf_path.is_file():
                print(f"[{idx}] NO se encuentra el PDF: {pdf_path}")
                continue

            # Route_id = índice del CSV
            route_id = int(idx)

            print(f"Procesando route_id={route_id} -> {pdf_path}")

            try:
                secciones = extraer_secciones_pdf(pdf_path)
            except Exception as e:
                print(f"  Error procesando {pdf_path}: {e}")
                continue

            for i, sec in enumerate(secciones):
                section_id = f"{route_id}_section-{i}"
                heading = sec["heading"]
                texto = sec["text"]
                writer.writerow([route_id, section_id, heading, texto])

    print(f"CSV global de secciones guardado en: {output_csv_path}")


if __name__ == "__main__":
    rutas_csv = r".\\rutas.csv"
    col_pdf_path = "pdf_path"  
    output_csv = r".\\rutas_pirineos_secciones_ve4.csv"

    procesar_todos_los_pdfs(rutas_csv, col_pdf_path, output_csv)