from pathlib import Path
from typing import List

import os
from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer

from haystack import Pipeline, component
from haystack.dataclasses import Document
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
IMAGE_EMBED_MODEL = "sentence-transformers/clip-ViT-B-32"

# Configurar globalmente
os.environ["GOOGLE_API_KEY"] = API_KEY

print(f"BD Chroma: {CHROMA_PERSIST_DIR}")
print(f"Colección TEXTO:  {TEXT_COLLECTION}")
print(f"Colección IMÁGENES: {IMAGE_COLLECTION}")
print(f"Modelo texto (retrieval): {TEXT_EMBED_MODEL}")
print(f"Modelo imágenes (retrieval): {IMAGE_EMBED_MODEL}")

# Cargar variables de entorno (.env en la raíz)
load_dotenv(BASE_DIR / ".env")
print("GOOGLE_API_KEY leída:", os.getenv("GOOGLE_API_KEY"))


# ============================================================
# 2. EMBEDDERS DE QUERY (COMPONENTES)
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


@component
class CLIPQueryEmbedder:
    """
    Embedder de queries con CLIP.
    Usamos el encoder de TEXTO de CLIP para recuperar IMÁGENES semánticamente relacionadas.
    """

    def __init__(self, model_name: str = IMAGE_EMBED_MODEL):
        self.model = SentenceTransformer(model_name)

    @component.output_types(embedding=List[float])
    def run(self, query: str):
        emb = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0]
        return {"embedding": emb.tolist()}


# ============================================================
# 3. PIPELINE SOLO DE RETRIEVAL (SIN GENERADOR)
# ============================================================

def build_retrieval_pipeline(
    top_k_text: int = 5,
    top_k_images: int = 4,
) -> Pipeline:
    # Stores de Chroma (los embeddings ya están guardados)
    text_store = ChromaDocumentStore(
        persist_path=str(CHROMA_PERSIST_DIR),
        collection_name=TEXT_COLLECTION,
        distance_function="cosine",
    )

    image_store = ChromaDocumentStore(
        persist_path=str(CHROMA_PERSIST_DIR),
        collection_name=IMAGE_COLLECTION,
        distance_function="cosine",
    )

    text_embedder = MiniLMQueryEmbedder()
    image_embedder = CLIPQueryEmbedder()

    text_retriever = ChromaEmbeddingRetriever(
        document_store=text_store,
        top_k=top_k_text,
    )
    image_retriever = ChromaEmbeddingRetriever(
        document_store=image_store,
        top_k=top_k_images,
    )

    pipe = Pipeline()
    pipe.add_component("text_embedder", text_embedder)
    pipe.add_component("image_embedder", image_embedder)
    pipe.add_component("text_retriever", text_retriever)
    pipe.add_component("image_retriever", image_retriever)

    pipe.connect("text_embedder.embedding", "text_retriever.query_embedding")
    pipe.connect("image_embedder.embedding", "image_retriever.query_embedding")

    return pipe


# ============================================================
# 4. CONSTRUCCIÓN DEL PROMPT PARA GEMINI (FUERA DEL PIPELINE)
# ============================================================

def build_prompt_from_context(query: str,
                              text_docs: List[Document],
                              image_docs: List[Document]) -> str:
    """Construye un prompt de texto usando contexto de rutas + info de imágenes."""

    context_lines = []

    for i, doc in enumerate(text_docs[:5], start=1):
        meta = doc.meta or {}
        ruta = meta.get("nombre_ruta") or meta.get("ruta")
        poblacion = meta.get("poblacion_cercana")
        dificultad = meta.get("dificultad")
        distancia = meta.get("distancia_total_km")


        header_parts = []
        if ruta:
            header_parts.append(f"Ruta: {ruta}")
        if poblacion:
            header_parts.append(f"Población cercana: {poblacion}")
        if dificultad:
            header_parts.append(f"Dificultad: {dificultad}")
        if distancia is not None:
            header_parts.append(f"Distancia total: {distancia} km")

        header = " | ".join(header_parts) if header_parts else f"Documento {i}"
        texto = (doc.content or "").replace("\n", " ")

        context_lines.append(f"[{i}] {header}\n{texto[:600]}")

    context_text = "\n\n".join(context_lines) if context_lines else "No hay contexto textual disponible."

    # Info sobre imágenes (como texto, no como píxeles)
    image_lines = []
    for j, img_doc in enumerate(image_docs[:4], start=1):
        meta = img_doc.meta or {}
        img_rel = meta.get("image_path")
        ruta = meta.get("nombre_ruta") or meta.get("ruta")
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
        "Utiliza el contexto textual y la información sobre las imágenes "
        "para razonar sobre el mapa de la ruta, el terreno, el entorno y la dificultad. " \
        "Además menciona siempre el país, provincia, zona o población cercana si está disponible. "
    )

    prompt = (
        f"{system_instruction}\n\n"
        f"Pregunta del usuario: {query}\n\n"
        f"Contexto textual relevante:\n{context_text}\n\n"
        f"Información sobre imágenes recuperadas:\n{images_text}\n\n"
        f"Responde de forma útil para un senderista."
    )

    return prompt


# ============================================================
# 5. MODO INTERACTIVO: RETRIEVAL + GEMINI MANUAL
# ============================================================

def interactive_multimodal_chat():
    pipe = build_retrieval_pipeline(top_k_text=5, top_k_images=4)

    # Generador de Gemini (fuera del pipeline)
    gemini = GoogleAIGeminiGenerator(
        model="gemini-2.0-flash",
        # api_key se coge de GOOGLE_API_KEY por defecto
    )

    print("\n💬 Chat RAG (texto + info de imágenes con Gemini).")
    print("   Escribe 'salir' para terminar.\n")

    while True:
        query = input("❓ Pregunta: ").strip()
        if not query:
            continue
        if query.lower() in {"salir", "exit", "quit"}:
            print("👋 Saliendo.")
            break

        # 🔹 FILTRO MANUAL: solo rutas con poblacion_cercana = "El Bruc"
        filters_el_bruc = {
            "field": "meta.poblacion_cercana",
            "operator": "==",
            "value": "El Bruc",
        }

        result = pipe.run(
            {
                "text_embedder": {"query": query},
                "image_embedder": {"query": query},
                "text_retriever": {"filters": filters_el_bruc},
                "image_retriever": {"filters": filters_el_bruc},
            }
        )

        text_docs = result["text_retriever"]["documents"]
        image_docs = result["image_retriever"]["documents"]

        # Construir prompt a partir del contexto
        prompt = build_prompt_from_context(query, text_docs, image_docs)

        # Llamar a Gemini directamente
        gen_out = gemini.run(parts=prompt)
        answer = gen_out["replies"][0]

        print("\n🧠 Respuesta del modelo:\n")
        print(answer)
        print("\n" + "=" * 80)

        # DEBUG: ver qué docs se han usado
        print("\n📚 (DEBUG) Algunos documentos de texto recuperados:\n")
        for i, doc in enumerate(text_docs[:3], start=1):
            ruta = doc.meta.get("nombre_ruta")
            poblacion = doc.meta.get("poblacion_cercana")
            print(f"[{i}] ruta={ruta} | poblacion_cercana={poblacion}")
        print("\n" + "=" * 80)


if __name__ == "__main__":
    interactive_multimodal_chat()
