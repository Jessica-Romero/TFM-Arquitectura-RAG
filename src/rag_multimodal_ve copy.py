from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any, Optional
import os
import json
import unicodedata
import re
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from haystack import Pipeline, component
from haystack.dataclasses import Document
from haystack.dataclasses.byte_stream import ByteStream

from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever
from haystack_integrations.components.generators.google_ai import GoogleAIGeminiGenerator

from configuracion_gemini import API_KEY


# ============================================================
# 1. CONFIGURACIÓN BÁSICA
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]   # raíz del repo
CHROMA_PERSIST_DIR = BASE_DIR / "src" / "chroma_db"

TEXT_COLLECTION = "rutas_text_waypoints"
IMAGE_COLLECTION = "rutas_imagenes"

TEXT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# API KEY Google
os.environ["GOOGLE_API_KEY"] = API_KEY
load_dotenv(BASE_DIR / ".env")

print(f"BD Chroma: {CHROMA_PERSIST_DIR}")
print(f"Colección TEXTO:  {TEXT_COLLECTION}")
print(f"Colección IMÁGENES: {IMAGE_COLLECTION}")
print(f"Modelo texto (retrieval): {TEXT_EMBED_MODEL}")
print("GOOGLE_API_KEY leída:", os.getenv("GOOGLE_API_KEY"))


# ============================================================
# 2. EMBEDDER DE QUERY (TEXTO)
# ============================================================

@component
class MiniLMQueryEmbedder:
    """Embedder de queries con MiniLM (para recuperar TEXTO + WAYPOINTS)."""

    def __init__(self, model_name: str = TEXT_EMBED_MODEL):
        self.model = SentenceTransformer(model_name)

    @component.output_types(embedding=List[float])
    def run(self, query: str):
        text = "Represent the query for retrieval: " + query.strip()
        emb = self.model.encode(
            [text],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0]
        return {"embedding": emb.tolist()}


# ============================================================
# 3. FILTROS CON LLM (Gemini → JSON de filtros)
# ============================================================

gemini_filter_extractor = GoogleAIGeminiGenerator(model="gemini-2.5-flash-lite")

FILTER_SCHEMA = [
    "pais", "region", "provincia", "comarca", "zona", "pueblo",
    "Punto_salida_llegada", "poblacion_cercana",
    "categoria", "dificultad", "distancia_total_km",
    "altitud_minima_m", "altitud_maxima_m",
    "desnivel_acumulado_pos", "desnivel_acumulado_neg",
    "tiempo_total_min",
]


# Campos numéricos dentro del esquema de filtros
NUMERIC_FIELDS = {
    "distancia_total_km",
    "altitud_minima_m",
    "altitud_maxima_m",
    "desnivel_acumulado_pos",
    "desnivel_acumulado_neg",
    "tiempo_total_min",
}

# El resto de campos del esquema los consideramos de texto
TEXT_FIELDS = set(FILTER_SCHEMA) - NUMERIC_FIELDS

# Campos de localización: queremos OR entre ellos cuando la query hable de un sitio
LOCATION_FIELDS = {
    "pais", "region", "provincia", "comarca", "zona", "pueblo",
    "Punto_salida_llegada", "poblacion_cercana",
}

# Normalización sencilla de dificultad por si el LLM no clava baja/media/alta
def normalize_difficulty_value(val: str) -> str:
    v = val.lower().strip()
    # ya normalizado sin tildes por normalize_meta_value()
    if any(w in v for w in ["facil", "sencilla", "tranquila", "iniciacion", "baja"]):
        return "baja"
    if any(w in v for w in ["moderada", "notable", "media"]):
        return "media"
    if any(w in v for w in ["dificil", "dura", "exigente", "alta"]):
        return "alta"
    return v  

def normalize_meta_value(text: Any) -> str:
    """
    Normaliza un valor de metadato de TEXTO para que encaje con cómo
    has guardado los metadatos en Chroma:
      - str()
      - minúsculas
      - sin acentos
      - solo letras/números/espacios
    """
    if text is None:
        return ""

    s = str(text).lower().strip()

    # quitar acentos
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("utf-8")
    # mantener solo letras, números y espacios
    s = re.sub(r"[^a-z0-9\s]", "", s)
    # colapsar espacios múltiples
    s = re.sub(r"\s+", " ", s).strip()

    return s

def llm_build_filters(query: str) -> Optional[List[Dict[str, Any]]]:
    prompt = f"""
    # INSTRUCCIONES PRINCIPALES
    Eres un asistente especializado en extraer filtros estructurados para buscar rutas de montaña en una base de datos.

    Tu tarea es analizar la consulta del usuario y generar un JSON con filtros válidos usando SOLO los campos permitidos.

    ## CAMPOS DISPONIBLES
    ### Campos de texto (usar operador "=="):
    pais, region, provincia, comarca, zona, pueblo, categoria, dificultad, Punto_salida_llegada, poblacion_cercana

    ### Campos de localización (para bloques OR):
    pais, region, provincia, comarca, zona, pueblo, poblacion_cercana, Punto_salida_llegada

    ### Campos numéricos (usar operadores comparativos):
    distancia_total_km, tiempo_total_min, altitud_minima_m, altitud_maxima_m, desnivel_acumulado_pos, desnivel_acumulado_neg

    ## REGLAS CLAVE

    ### 1. NORMALIZACIÓN DE VALORES
    - Texto en minúsculas, sin tildes, sin signos especiales
    - Números sin unidades (solo el valor numérico)

    ### 2. BLOQUES OR PARA LOCALIZACIÓN
    Cuando el usuario busque "rutas en <lugar>", crea un bloque OR con TODOS los campos de localización que podrían contener ese valor.

    Ejemplo para "Barcelona":
    {{
    "operator": "OR",
    "conditions": [
        {{"field": "meta.provincia", "operator": "==", "value": "barcelona"}},
        {{"field": "meta.region", "operator": "==", "value": "barcelona"}},
        {{"field": "meta.zona", "operator": "==", "value": "barcelona"}},
        {{"field": "meta.poblacion_cercana", "operator": "==", "value": "barcelona"}}
    ]
    }}

    ### 3. MAPEO DE DIFICULTAD
    - "facil", "sencilla", "tranquila", "de iniciacion" → "baja"
    - "moderada", "notable", "intermedia" → "media"  
    - "dificil", "dura", "exigente", "muy dura" → "alta"

    ### 4. MAPEO DE CATEGORÍAS
    - "circular", "bucle" → "rutas circulares"
    - "familias", "niños", "niñas" → "familias y ninos"
    - "alta montaña", "alta montana" → "ascensiones alta montana"
    - "lagos", "ibones" → "lagos"
    - "raquetas", "nieve" → "raquetas de nieve"
    - "vistas", "panorámica" → "paisajes pintorescos"
    - "patrimonio", "histórico" → "patrimonio historico"
    - "etapas", "GR", "travesía" → "rutas por etapas y grs"

    ### 5. REGLAS NUMÉRICAS
    - "menos de X km" → operador "<="
    - "más de X km" → operador ">="
    - "entre X y Y horas" → dos filtros con ">=" y "<="

    ## RESTRICCIONES ESTRICTAS
    1. **NO INVENTAR** campos fuera de la lista
    2. **MÁXIMO 4 filtros** en total
    3. **PRIORIDAD**: 1) Localización, 2) Dificultad, 3) Distancia/Tiempo
    4. Si no hay filtros claros → devolver []
    5. **Siempre** usar "meta.<campo>" en el field

    ## FORMATO DE RESPUESTA
    Devuelve SOLO un array JSON. Cada elemento puede ser:
    1. Filtro simple: {{"field": "meta.campo", "operator": "==", "value": "valor"}}
    2. Bloque OR: {{"operator": "OR", "conditions": [array de filtros]}}

    ## EJEMPLOS FINALES

    ### Ejemplo 1:
    Consulta: "Rutas en Barcelona de dificultad moderada y menos de 10 km"
    Respuesta: [
    {{
        "operator": "OR",
        "conditions": [
        {{"field": "meta.provincia", "operator": "==", "value": "barcelona"}},
        {{"field": "meta.region", "operator": "==", "value": "barcelona"}},
        {{"field": "meta.poblacion_cercana", "operator": "==", "value": "barcelona"}}
        ]
    }},
    {{"field": "meta.dificultad", "operator": "==", "value": "media"}},
    {{"field": "meta.distancia_total_km", "operator": "<=", "value": 10}}
    ]

    ### Ejemplo 2:
    Consulta: "Ruta fácil en Pirineos"
    Respuesta: [
    {{
        "operator": "OR",
        "conditions": [
        {{"field": "meta.zona", "operator": "==", "value": "pirineos"}},
        {{"field": "meta.region", "operator": "==", "value": "pirineos"}}
        ]
    }},
    {{"field": "meta.dificultad", "operator": "==", "value": "baja"}}
    ]

    ### 4. MAPEO DE CATEGORÍAS
    [... lo que ya tienes ...]

    - IMPORTANTE:
    - Si la consulta menciona varias ideas relacionadas (por ejemplo: circular, niños y lagos),
        elige SOLO la categoría más representativa de la intención principal del usuario.
    - No generes más de UN filtro de categoria.

    Ejemplo:
    Consulta: "Circular para niños con lagos"
    Respuesta correcta:
    [
    {{"field": "meta.categoria", "operator": "==", "value": "familias y ninos"}}
    ]

    ---

    CONSULTA DEL USUARIO: "{query}"

    RESPUESTA (SOLO JSON):
    """
    try:
        out = gemini_filter_extractor.run(parts=[prompt])
        text = out["replies"][0].strip()

        # Limpiar posibles ```...```
        if text.startswith("```"):
            lines = text.splitlines()
            if len(lines) >= 2:
                if lines[0].strip().startswith("```json"):
                    lines = lines[1:]
                if lines[-1].strip().startswith("```"):
                    lines = lines[:-1]
            text = "\n".join(lines).strip()

        data = json.loads(text)

        if not isinstance(data, list):
            print("⚠️ El LLM no devolvió una lista de filtros:", type(data))
            return None

        normalized_filters: List[Dict[str, Any]] = []

        for item in data:
            if not isinstance(item, dict):
                continue

            op = item.get("operator")

            # 1) Bloque OR compuesto (para localización u otros casos)
            if op == "OR" and "conditions" in item:
                raw_conditions = item["conditions"]
                subfilters: List[Dict[str, Any]] = []

                if isinstance(raw_conditions, list):
                    for cond in raw_conditions:
                        if not isinstance(cond, dict):
                            continue

                        field = cond.get("field")
                        c_op = cond.get("operator")
                        value = cond.get("value")

                        if not field or c_op is None or value is None:
                            continue

                        if not field.startswith("meta."):
                            field = f"meta.{field}"

                        raw_field = field[5:]

                        if raw_field not in FILTER_SCHEMA:
                            continue

                        # Numéricos
                        if raw_field in NUMERIC_FIELDS:
                            try:
                                num_val = float(value)
                            except Exception:
                                continue
                            if c_op not in ["<", "<=", ">", ">=", "=="]:
                                continue
                            subfilters.append(
                                {"field": field, "operator": c_op, "value": num_val}
                            )
                            continue

                        # Texto
                        if raw_field in TEXT_FIELDS:
                            norm_val = normalize_meta_value(value)
                            if not norm_val:
                                continue

                            # Dificultad: aseguramos baja/media/alta
                            if raw_field == "dificultad":
                                norm_val = normalize_difficulty_value(norm_val)

                            subfilters.append(
                                {"field": field, "operator": "==", "value": norm_val}
                            )
                            continue

                if subfilters:
                    normalized_filters.append({"operator": "OR", "conditions": subfilters})
                continue

            # 2) Filtro plano normal
            field = item.get("field")
            value = item.get("value")

            if not field or op is None or value is None:
                continue

            if not field.startswith("meta."):
                field = f"meta.{field}"

            raw_field = field[5:]

            if raw_field not in FILTER_SCHEMA:
                continue

            # Numérico plano
            if raw_field in NUMERIC_FIELDS:
                try:
                    num_val = float(value)
                except Exception:
                    continue
                if op not in ["<", "<=", ">", ">=", "=="]:
                    continue
                normalized_filters.append(
                    {"field": field, "operator": op, "value": num_val}
                )
                continue

            # Texto plano
            if raw_field in TEXT_FIELDS:
                norm_val = normalize_meta_value(value)
                if not norm_val:
                    continue

                if raw_field == "dificultad":
                    norm_val = normalize_difficulty_value(norm_val)

                normalized_filters.append(
                    {"field": field, "operator": "==", "value": norm_val}
                )
                continue

        return normalized_filters if normalized_filters else None

    except Exception as e:
        print(f"Error en llm_build_filters: {e}")
        if 'text' in locals():
            print("Respuesta cruda del LLM:", text[:200])
        return None


# ============================================================
# 4. PIPELINE DE RETRIEVAL (solo TEXTO)
# ============================================================

def build_retrieval_pipeline(top_k_text: int = 5) -> Pipeline:
    text_store = ChromaDocumentStore(
        persist_path=str(CHROMA_PERSIST_DIR),
        collection_name=TEXT_COLLECTION,
        distance_function="cosine",
    )

    text_embedder = MiniLMQueryEmbedder()

    text_retriever = ChromaEmbeddingRetriever(
        document_store=text_store,
        top_k=top_k_text,
    )

    pipe = Pipeline()
    pipe.add_component("text_embedder", text_embedder)
    pipe.add_component("text_retriever", text_retriever)

    pipe.connect("text_embedder.embedding", "text_retriever.query_embedding")

    return pipe


# ============================================================
# 5. FUNCIONES AUXILIARES: route_id → imágenes
# ============================================================

def extract_route_ids(docs: List[Document]) -> List[Any]:
    """Extrae route_id únicos de los documentos de texto."""
    ids = []
    for d in docs:
        meta = d.meta or {}
        rid = meta.get("route_id") or meta.get("routeid")
        if rid is not None and rid not in ids:
            ids.append(rid)
    return ids


def fetch_images_for_route_ids(route_ids: List[Any], max_images: int = 8) -> List[Document]:
    """Busca en la colección de imágenes todas las que tengan esos route_id."""
    if not route_ids:
        return []

    image_store = ChromaDocumentStore(
        persist_path=str(CHROMA_PERSIST_DIR),
        collection_name=IMAGE_COLLECTION,
        distance_function="cosine",
    )

    # Caso 1: solo un route_id → filtro simple
    if len(route_ids) == 1:
        filters = {
            "field": "meta.route_id",
            "operator": "==",
            "value": route_ids[0],
        }

    # Caso 2: varios route_id → OR entre ellos
    else:
        filters = {
            "operator": "OR",
            "conditions": [
                {
                    "field": "meta.route_id",
                    "operator": "==",
                    "value": rid,
                }
                for rid in route_ids
            ],
        }

    image_docs = image_store.filter_documents(filters=filters)

    if len(image_docs) > max_images:
        image_docs = image_docs[:max_images]

    print(f"Imágenes recuperadas por route_id: {len(image_docs)}")
    return image_docs


# ============================================================
# 6. PROMPT DE CONTEXTO (texto + metadatos de imágenes + historial)
# ============================================================

def build_prompt_from_context(
    query: str,
    text_docs: List[Document],
    image_docs: List[Document],
) -> str:
    """Construye prompt de texto para Gemini."""

    # --- Contexto textual de rutas ---
    context_lines = []

    for i, doc in enumerate(text_docs[:5], start=1):
        meta = doc.meta or {}
        score = getattr(doc, "score", None)
        ruta = meta.get("nombre_ruta_org") # Indicar que use el nombre original de la ruta
        # Localización de la ruta (completa)
        pais = meta.get("pais_org")
        region = meta.get("region_org")
        provincia = meta.get("provincia_org")
        comarca = meta.get("comarca_org")
        zona = meta.get("zona_org")
        pueblo = meta.get("pueblo_org")
        poblacion = meta.get("poblacion_cercana_org")
        punto_salida_llegada = meta.get("Punto_salida_llegada_org")
        # Metadatos relevantes a tener en cuenta al valorar una ruta
        dificultad = meta.get("dificultad_org")
        descripcion_dificultad = meta.get("descripcion_dificultad")  # Indicar que use esta descripción para justificar la dificultad categorizada
        distancia = meta.get("distancia_total_km")
        tiempo = meta.get("tiempo_total_min") # Indicar que convierta el dato de minutos a horas
        desnivel_pos = meta.get("desnivel_acumulado_pos") # Indicar que si este dato existe, lo use para justificar la dificultad categorizada
        desnivel_neg = meta.get("desnivel_acumulado_neg") # Indicar que si este dato existe, lo use para justificar la dificultad categorizada
        altitud_min = meta.get("altitud_minima_m")  # Metadato adicional para contextualizar la ruta
        altitud_max = meta.get("altitud_maxima_m")  # Metadato adicional para contextualizar la ruta
        # Metadato adicional para indicar el tipo de ruta
          # Tipo de ruta / categoría
        categoria = meta.get("categoria_org") or meta.get("categoria")
        if isinstance(categoria, str):
            # Si viene como "senderismo;circular;familiar", lo hacemos más legible
            categorias_limpias = [
                c.strip() for c in categoria.split(";") if c.strip()
            ]
            if categorias_limpias:
                categoria = ", ".join(categorias_limpias)
            else:
                categoria = None
        # Metadato para buscar más información sobre la ruta
        link_archivo = meta.get("link_archivo")
        # Metadato de fecha para contextualizar de cuándo es la información
        fecha = meta.get("fecha")

        # -------------------------------
        # 2) Construcción del header
        # -------------------------------
        header_parts = []

        if ruta:
            header_parts.append(f"Ruta: {ruta}")
        # relevancia según el retriever
        if score is not None:
            # asumiendo que es similitud (0–1). Si vieras valores raros, puedes quitar el formato.
            header_parts.append(f"Relevancia (score del retriever): {score:.3f}")
        # Localización (orden más concreto → más general)
        if pueblo:
            header_parts.append(f"Pueblo: {pueblo}")
        if poblacion:
            header_parts.append(f"Población cercana: {poblacion}")
        if punto_salida_llegada:
            header_parts.append(f"Punto de salida/llegada: {punto_salida_llegada}")
        if comarca:
            header_parts.append(f"Comarca: {comarca}")
        if zona:
            header_parts.append(f"Zona: {zona}")
        if provincia:
            header_parts.append(f"Provincia: {provincia}")
        if region:
            header_parts.append(f"Región: {region}")
        if pais:
            header_parts.append(f"País: {pais}")

        # Tipo de ruta + dificultad
        if categoria:
            header_parts.append(f"Categoría / tipo de ruta: {categoria}")
        if dificultad:
            header_parts.append(f"Dificultad (texto original): {dificultad}")
        if descripcion_dificultad:
            header_parts.append(
                f"Descripción de la dificultad (útil para justificar la categoría): {descripcion_dificultad}"
            )

        # Datos cuantitativos
        if distancia is not None and distancia != "":
            header_parts.append(f"Distancia total: {distancia} km")
        if tiempo is not None:
            header_parts.append(f"Tiempo total (minutos): {tiempo}")

        if desnivel_pos is not None and desnivel_pos != "":
            header_parts.append(
                f"Desnivel positivo acumulado (+): {desnivel_pos} m (puede ayudar a justificar la dificultad)"
            )
        if desnivel_neg is not None and desnivel_neg != "":
            header_parts.append(
                f"Desnivel negativo acumulado (-): {desnivel_neg} m (puede ayudar a justificar la dificultad)"
            )
        if altitud_min is not None and altitud_min != "":
            header_parts.append(f"Altitud mínima en la que empieza la ruta: {altitud_min} m")
        if altitud_max is not None and altitud_max != "":
            header_parts.append(f"Altitud máxima a la que se alcanza en la ruta: {altitud_max} m")

        # Enlace / fecha
        if fecha:
            header_parts.append(f"Fecha de la información extraida de la ruta: {fecha}")
        if link_archivo:
            header_parts.append(f"Más información en el enlace de origen: {link_archivo}")

        header = " | ".join(header_parts) if header_parts else f"Documento {i}"
        texto = (doc.content or "").replace("\n", " ")

        context_lines.append(f"[{i}] {header}\n{texto[:1000]}")

    context_text = "\n\n".join(context_lines) if context_lines else "No hay contexto textual disponible."

    # --- Información sobre imágenes (solo como texto, adicional) ---
    image_lines = []
    for j, img_doc in enumerate(image_docs[:4], start=1):
        meta = img_doc.meta or {}
        img_rel = meta.get("image_path")
        ruta = meta.get("nombre_ruta")
        img_path = meta.get("image_path")
        poblacion = meta.get("poblacion_cercana")
        line = f"Imagen {j}: path={img_rel}"
        if ruta:
            line += f" | ruta={ruta}"
        if img_path:
            line += f" | Ruta relativa de la imagen={img_path}"
        image_lines.append(line)

    images_text = "\n".join(image_lines) if image_lines else "No hay imágenes asociadas disponibles."
 
    system_instruction_optimizado = """# GUÍA EXPERTA PARA RUTAS DE MONTAÑA

        ## IDENTIFICACIÓN DE TIPO DE PREGUNTA
        Analiza la pregunta y clasifícala en EXACTAMENTE UNA categoría:

        **CATEGORÍA A: FACTUAL SIMPLE** - Pregunta por UN dato numérico específico
        - Patrones: "¿Cuál es [distancia|tiempo|desnivel|altitud]...?", "¿Qué [valor]...?"
        - Ejemplo: "¿Cuál es la distancia total?"

        **CATEGORÍA B: FACTUAL MÚLTIPLE** - Pide 2-3 datos específicos
        - Patrones: "¿Cuál es X e Y...?", "Indica A y B..."
        - Ejemplo: "¿Qué distancia y desnivel tiene?"

        **CATEGORÍA C: ANÁLISIS/CONTEXTO** - Pide interpretación, recomendación o descripción
        - Patrones: "¿Qué ruta recomiendas...?", "¿Cómo es el paisaje...?", "Analiza la dificultad..."
        - Ejemplo: "¿Es apta para familias?"

        ## REGLAS DE RESPUESTA (ESTRICTO)

        ### PARA CATEGORÍA A (FACTUAL SIMPLE):
        1. Respuesta en UNA frase, máximo 15 palabras
        2. Solo el dato solicitado, sin contexto adicional
        3. Si no hay dato: "Según los datos, no se especifica [dato solicitado]"
        4. NO uses encabezados, listas ni formato

        ### PARA CATEGORÍA B (FACTUAL MÚLTIPLE):
        1. Respuesta en 2-3 frases
        2. Lista clara de cada dato: "• Distancia: X km • Desnivel: Y m"
        3. Sin análisis ni justificaciones

        ### PARA CATEGORÍA C (ANÁLISIS/CONTEXTO):
        1. Usa la estructura completa con secciones
        2. Desarrolla explicaciones basadas en datos
        3. Mantén tono técnico pero accesible

        ## REGLAS ABSOLUTAS
        1. NUNCA inventes valores numéricos
        2. Si falta información, dilo claramente
        3. Prioriza siempre los datos del contexto sobre tu conocimiento general
        4. Responde SIEMPRE en español

        ## FORMATO DE SECCIONES (solo para Categoría C)
        1. Identificación y ubicación
        2. Características técnicas  
        3. Evaluación de dificultad
        4. Contexto adicional
        5. Recomendaciones prácticas

        **NO uses estas secciones para Categorías A o B**"""
    # Template del prompt final
    prompt = f"""{system_instruction_optimizado}

    ## PREGUNTA DEL USUARIO
    {query}

    ## DATOS DE LA RUTA (CONTEXTO)
    {context_text}

    ## INFORMACIÓN VISUAL DISPONIBLE
    {images_text}

    ## TAREA FINAL
    Analiza los datos anteriores y proporciona una explicación completa y útil para un senderista, siguiendo la estructura recomendada y todas las reglas establecidas.

    RESPUESTA:
    """

    return prompt


# ============================================================
# 7. PARTES MULTIMODALES: prompt + ByteStream de imágenes
# ============================================================

def build_multimodal_parts(
    query: str,
    text_docs: List[Document],
    image_docs: List[Document],
    base_dir: Path,
    max_images: int = 4,
) -> List[Any]:
    """
    Construye la lista de 'parts' para Gemini:
      - Primera parte: prompt textual
      - Siguientes partes: imágenes como ByteStream
    """
    context_prompt = build_prompt_from_context(query, text_docs, image_docs)
    parts: List[Any] = [context_prompt]

    for img_doc in image_docs[:max_images]:
        meta = img_doc.meta or {}
        img_rel_path = meta.get("image_path")

        if not img_rel_path:
            continue

        img_abs = base_dir / img_rel_path

        if not img_abs.exists():
            print(f"⚠️ Imagen no encontrada en disco: {img_abs}")
            continue

        try:
            with open(img_abs, "rb") as f:
                img_data = f.read()

            ext = img_abs.suffix.lower()
            mime = "image/png" if ext == ".png" else "image/jpeg"

            parts.append(ByteStream(data=img_data, mime_type=mime))

        except Exception as e:
            print(f"⚠️ Error leyendo imagen {img_abs}: {e}")
            continue

    return parts


# ============================================================
# 8. MODO INTERACTIVO: RAG MULTIMODAL COMPLETO
# ============================================================

def interactive_multimodal_chat():
    pipe = build_retrieval_pipeline(top_k_text=5)

    gemini = GoogleAIGeminiGenerator(
        model="gemini-2.5-flash-lite",
        # api_key se coge de GOOGLE_API_KEY
    )

    last_text_docs: List[Document] = []
    last_route_ids: List[Any] = []

    print("\n💬 Chat RAG MULTIMODAL (texto + imágenes con Gemini).")
    print("   Escribe 'salir' para terminar.\n")

    while True:
        query = input("❓ Pregunta: ").strip()
        if not query:
            continue
        if query.lower() in {"salir", "exit", "quit"}:
            print("👋 Saliendo.")
            break

        # Detectar si es una pregunta de seguimiento sobre imágenes
        wants_images = any(word in query.lower() for word in ["imagen", "imágenes", "foto", "fotos"])

        # 1) Filtros sólo para queries "normales" (no puramente sobre imágenes)
        filters_dict = None
        if not wants_images:
            filters_list = llm_build_filters(query)
            print("DEBUG filtros LLM (lista):", filters_list)

            if filters_list:
                if len(filters_list) == 1:
                    filters_dict = filters_list[0]
                else:
                    filters_dict = {"operator": "AND", "conditions": filters_list}

        # 2) Recuperación de contexto
        if wants_images and last_route_ids:
            # Reutilizamos la última ruta para buscar imágenes asociadas
            print("Usando la última ruta encontrada para buscar imágenes asociadas.")
            text_docs = last_text_docs
            image_docs = fetch_images_for_route_ids(last_route_ids, max_images=8)

        else:
            # Consulta normal: retrieval texto + imágenes por route_id
            inputs: Dict[str, Any] = {
                "text_embedder": {"query": query},
            }
            if filters_dict is not None:
                inputs["text_retriever"] = {"filters": filters_dict}
            result = pipe.run(inputs)
            text_docs = result["text_retriever"]["documents"]

            # Fallback: si los filtros han dejado el retrieval a cero, repetir sin filtros
            if not text_docs and filters_dict is not None:
                print("Sin resultados con filtros; repito retrieval SIN filtros.")
                result = pipe.run({"text_embedder": {"query": query}})
                text_docs = result["text_retriever"]["documents"]

            route_ids = extract_route_ids(text_docs)
            print(f"Buscando imágenes para route_id: {route_ids}")
            image_docs = fetch_images_for_route_ids(route_ids, max_images=8)


            last_text_docs = text_docs
            last_route_ids = route_ids

        # 3) Construir parts multimodales (prompt + imágenes)
        parts = build_multimodal_parts(
            query=query,
            text_docs=text_docs,
            image_docs=image_docs,
            base_dir=BASE_DIR,
        )

        # 4) Llamar a Gemini
        gen_out = gemini.run(parts=parts)
        answer = gen_out["replies"][0]

        print("\n🧠 Respuesta del modelo:\n")
        print(answer)
        print("\n" + "=" * 80)

        print("\n📚 (DEBUG) Algunos documentos de texto recuperados:\n")
        for i, doc in enumerate(text_docs[:3], start=1):
            meta = doc.meta or {}
            ruta = meta.get("nombre_ruta_org") or meta.get("nombre_ruta")
            #poblacion = meta.get("poblacion_cercana_org") or meta.get("poblacion_cercana")
            score = getattr(doc, "score", None)
            print(f"[{i}] score={score:.3f} | ruta={ruta}" if score is not None
                else f"[{i}] score=None | ruta={ruta}")
        print("\n" + "=" * 80)

        print("\nDEBUG — valores reales en metadatos:\n")
        for d in text_docs[:10]:
            print(
                "ruta:", d.meta.get("nombre_ruta"),
                "| zona:", d.meta.get("zona"),
                "| dificultad:", d.meta.get("dificultad"),
                "| score:", getattr(d, "score", None),
            )


if __name__ == "__main__":
    interactive_multimodal_chat()
