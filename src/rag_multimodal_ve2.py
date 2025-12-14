from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any, Optional
import os
import json

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

gemini_filter_extractor = GoogleAIGeminiGenerator(model="gemini-2.5-flash")

FILTER_SCHEMA = [
    "pais", "region", "provincia", "comarca", "zona", "pueblo",
    "nombre_ruta", "categoria", "dificultad", "distancia_total_km",
    "altitud_minima_m", "altitud_maxima_m",
    "desnivel_acumulado_pos", "desnivel_acumulado_neg",
    "tiempo_total_min", "Punto_salida_llegada", "poblacion_cercana", "fecha",
]


def llm_build_filters(query: str) -> Optional[List[Dict[str, Any]]]:
    """
    Usa Gemini para extraer filtros estructurados de la pregunta.

    Devuelve una LISTA de filtros del tipo:
    [
        {"field": "meta.dificultad", "operator": "==", "value": "Fácil"},
        {"field": "meta.distancia_total_km", "operator": "<=", "value": 10},
        ...
    ]

    Si no hay filtros → None
    """

    prompt = f"""
Extrae filtros de esta pregunta sobre rutas de montaña.

Campos disponibles (sin el prefijo meta.): {', '.join(FILTER_SCHEMA)}

Devuelve SOLO un JSON array con filtros en este formato:
[
  {{"field": "meta.<campo>", "operator": "==", "value": "valor"}},
  ...
]

Reglas:
- Para campos de texto usa '=='.
- Para comparaciones numéricas usa '<=', '>=', '>', '<' cuando tenga sentido.
- Si la pregunta no tiene filtros claros, devuelve [].

Ejemplos:
- "Rutas en Barcelona" →
  [{{"field": "meta.provincia", "operator": "==", "value": "Barcelona"}}]

- "Sendero de menos de 10 km" →
  [{{"field": "meta.distancia_total_km", "operator": "<=", "value": 10}}]

- "Rutas cerca de El Bruc" →
  [{{"field": "meta.poblacion_cercana", "operator": "==", "value": "El Bruc"}}]

Pregunta del usuario: "{query}"

Responde SOLO con el JSON array (sin texto extra).
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

        # Asegurar prefijo meta.
        for item in data:
            if "field" in item and not item["field"].startswith("meta."):
                item["field"] = f"meta.{item['field']}"

        return data if data else None

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

    # 👉 Caso 1: solo un route_id → filtro simple
    if len(route_ids) == 1:
        filters = {
            "field": "meta.route_id",
            "operator": "==",
            "value": route_ids[0],
        }

    # 👉 Caso 2: varios route_id → OR entre ellos
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

    print(f"🖼️ Imágenes recuperadas por route_id: {len(image_docs)}")
    return image_docs


# ============================================================
# 6. PROMPT DE CONTEXTO (texto + metadatos de imágenes + historial)
# ============================================================

def build_prompt_from_context(
    query: str,
    text_docs: List[Document],
    image_docs: List[Document],
    chat_history: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Construye prompt de texto para Gemini."""

    # --- Historial ---
    history_lines = []
    if chat_history:
        for i, turn in enumerate(chat_history[-5:], start=1):
            h_user = turn.get("user", "")
            h_assistant = turn.get("assistant", "")
            history_lines.append(
                f"Turno {i}:\n"
                f"  Usuario: {h_user}\n"
                f"  Asistente: {h_assistant}\n"
            )
    history_text = "\n".join(history_lines) if history_lines else "No hay historial previo relevante."

    # --- Contexto textual de rutas ---
    context_lines = []

    for i, doc in enumerate(text_docs[:5], start=1):
        meta = doc.meta or {}
        ruta = meta.get("nombre_ruta")
        pais = meta.get("pais")
        zona = meta.get("zona")
        comarca = meta.get("comarca")
        poblacion = meta.get("poblacion_cercana")
        dificultad = meta.get("dificultad")
        distancia = meta.get("distancia_total_km")
        tiempo = meta.get("tiempo_total_min")
        desnivel_pos = meta.get("desnivel_acumulado_pos")
        desnivel_neg = meta.get("desnivel_acumulado_neg")
        descripcion_dificultad = meta.get("descripcion_dificultad")
        categoria = meta.get("categoria")

        header_parts = []
        if ruta:
            header_parts.append(f"Ruta: {ruta}")
        if poblacion:
            header_parts.append(f"Población cercana: {poblacion}")
        if comarca:
            header_parts.append(f"Comarca: {comarca}")
        if zona:
            header_parts.append(f"Zona: {zona}")
        if pais:
            header_parts.append(f"País: {pais}")
        if categoria:
            header_parts.append(f"Categoría: {categoria}")
        if dificultad:
            header_parts.append(f"Dificultad: {dificultad}")
        if distancia is not None:
            header_parts.append(f"Distancia: {distancia} km")
        if tiempo is not None:
            header_parts.append(f"Tiempo total: {tiempo} min")
        if desnivel_pos is not None:
            header_parts.append(f"Desnivel +: {desnivel_pos} m")
        if desnivel_neg is not None:
            header_parts.append(f"Desnivel -: {desnivel_neg} m")
        if descripcion_dificultad:
            header_parts.append(f"Descripción dificultad: {descripcion_dificultad}")

        header = " | ".join(header_parts) if header_parts else f"Documento {i}"
        texto = (doc.content or "").replace("\n", " ")

        context_lines.append(f"[{i}] {header}\n{texto[:800]}")

    context_text = "\n\n".join(context_lines) if context_lines else "No hay contexto textual disponible."

    # --- Información sobre imágenes (solo como texto, adicional) ---
    image_lines = []
    for j, img_doc in enumerate(image_docs[:4], start=1):
        meta = img_doc.meta or {}
        img_rel = meta.get("image_path")
        ruta = meta.get("nombre_ruta")
        poblacion = meta.get("poblacion_cercana")
        line = f"Imagen {j}: path={img_rel}"
        if ruta:
            line += f" | ruta={ruta}"
        if poblacion:
            line += f" | población cercana={poblacion}"
        image_lines.append(line)

    images_text = "\n".join(image_lines) if image_lines else "No hay imágenes asociadas disponibles."

    system_instruction = (
        "Eres un guía experto en rutas de montaña y barrancos. "
        "Responde SIEMPRE en español, de forma clara y estructurada. "
        "Puedes usar el historial de conversación para entender preguntas de seguimiento. "
        "Si en la sección 'Información sobre imágenes recuperadas' aparece 'Imagen 1:', "
        "significa que SÍ hay imágenes disponibles de la ruta y debes mencionarlo en la respuesta. "
        "Utiliza el contexto textual y la información sobre las imágenes "
        "para razonar sobre la ruta, el terreno, el entorno, la dificultad y el desnivel positivo y negativo. " \
        "Si no sabes algo, no lo inventes; di que no tienes esa información."
    )

    prompt = (
        f"{system_instruction}\n\n"
        f"Historial reciente de la conversación:\n{history_text}\n\n"
        f"Pregunta actual del usuario: {query}\n\n"
        f"Contexto textual relevante de la base de datos:\n{context_text}\n\n"
        f"Información sobre imágenes recuperadas:\n{images_text}\n\n"
        f"Responde de forma útil para un senderista."
    )

    return prompt


# ============================================================
# 7. PARTES MULTIMODALES: prompt + ByteStream de imágenes
# ============================================================

def build_multimodal_parts(
    query: str,
    text_docs: List[Document],
    image_docs: List[Document],
    chat_history: Optional[List[Dict[str, Any]]],
    base_dir: Path,
    max_images: int = 4,
) -> List[Any]:
    """
    Construye la lista de 'parts' para Gemini:
      - Primera parte: prompt textual
      - Siguientes partes: imágenes como ByteStream
    """
    context_prompt = build_prompt_from_context(query, text_docs, image_docs, chat_history)
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
        model="gemini-2.5-flash",
        # api_key se coge de GOOGLE_API_KEY
    )

    chat_history: List[Dict[str, Any]] = []
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
            print("🔁 Usando la última ruta encontrada para buscar imágenes asociadas.")
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

            route_ids = extract_route_ids(text_docs)
            print(f"🔗 Buscando imágenes para route_id: {route_ids}")
            image_docs = fetch_images_for_route_ids(route_ids, max_images=8)

            last_text_docs = text_docs
            last_route_ids = route_ids

        # 3) Construir parts multimodales (prompt + imágenes)
        parts = build_multimodal_parts(
            query=query,
            text_docs=text_docs,
            image_docs=image_docs,
            chat_history=chat_history,
            base_dir=BASE_DIR,
        )

        # 4) Llamar a Gemini
        gen_out = gemini.run(parts=parts)
        answer = gen_out["replies"][0]

        print("\n🧠 Respuesta del modelo:\n")
        print(answer)
        print("\n" + "=" * 80)

        # 5) DEBUG: ver qué docs ha usado
        print("\n📚 (DEBUG) Algunos documentos de texto recuperados:\n")
        for i, doc in enumerate(text_docs[:3], start=1):
            ruta = doc.meta.get("nombre_ruta")
            poblacion = doc.meta.get("poblacion_cercana")
            print(f"[{i}] ruta={ruta} | poblacion_cercana={poblacion}")
        print("\n" + "=" * 80)

        # 6) Guardar en historial
        chat_history.append(
            {
                "user": query,
                "assistant": answer,
                "route_ids": last_route_ids,
            }
        )
        if len(chat_history) > 10:
            chat_history = chat_history[-10:]


if __name__ == "__main__":
    interactive_multimodal_chat()
