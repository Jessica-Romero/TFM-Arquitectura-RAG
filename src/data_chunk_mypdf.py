import fitz  # PyMuPDF
import csv
import re
from pathlib import Path


# --- 1) Headings que queremos detectar (patrones -> nombre "canonico") ---

HEADING_PATTERNS = {
    "LO MEJOR DE ESTA RUTA": "LO MEJOR DE ESTA RUTA",
    "INTRODUCCIÓN": "INTRODUCCIÓN",
    "MÁS INFORMACIÓN": "MÁS INFORMACIÓN",
    "PUNTOS IMPORTANTES DE PASO (WAYPOINTS)": "PUNTOS IMPORTANTES DE PASO (WAYPOINTS)",
    "RECORRIDO": "RECORRIDO",
    "¿SABÍAS QUE": "¿SABÍAS QUE...?",
    "CÓMO LLEGAR EN COCHE": "CÓMO LLEGAR EN COCHE",
    "NO TE PIERDAS": "NO TE PIERDAS...",
    "QUÉ MÁS PODEMOS VISITAR": "¿QUÉ MÁS PODEMOS VISITAR?"
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

                # ⬅️ aquí sí aplicamos el filtro de pie de página
                if es_pie_pagina(linea_texto):
                    continue

                ficha_lines.append(linea_texto)

        ficha_text = " ".join(ficha_lines)

    secciones.append({
        "heading": "FICHA TÉCNICA",
        "text": ficha_text.strip(),
    })

    # --- Resto de páginas igual que antes ---
    seccion_actual = None

    for page_idx in range(1, len(doc)):
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
        sec["text"] = clean_text(sec["text"])

    return secciones


def guardar_secciones_csv(secciones, csv_path: str | Path):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["section_id", "heading", "text"])
        for i, sec in enumerate(secciones):
            section_id = f"section-{i}"
            heading = sec["heading"]
            texto = sec["text"]
            writer.writerow([section_id, heading, texto])

    print(f"CSV guardado en: {csv_path}")


if __name__ == "__main__":
    pdf_path = r"..\download_pdf\RUTAS-PIRINEOS-ruta-tines-valle-del-flequer_es.pdf"
    csv_path = r".\chunks_pdf2.csv"
    secciones = extraer_secciones_pdf(pdf_path)
    print(f"Secciones detectadas: {len(secciones)}")
    for s in secciones:
        print("----")
        print("HEADING:", s["heading"])
        print("PREVIEW:", s["text"][:180], "...")
    guardar_secciones_csv(secciones, csv_path)
