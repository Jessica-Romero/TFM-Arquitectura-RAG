from pathlib import Path
from typing import List

from sentence_transformers import SentenceTransformer
from PIL import Image

from haystack import Pipeline, component
from haystack.dataclasses import Document
from haystack.dataclasses.byte_stream import ByteStream
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever
from haystack_integrations.components.generators.google_ai import GoogleAIGeminiGenerator
# Línea 1: Importar configuración
from configuracion_gemini import API_KEY
from typing import Union, List 
import os


# ============================================================
# 1. CONFIGURACIÓN
# ============================================================

# Este fichero está en: .../src/rag_multimodal_gemini.py
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


print("🔄 Cargando modelo de texto MiniLM...")
TEXT_MODEL = SentenceTransformer(TEXT_EMBED_MODEL)

print("🔄 Cargando modelo de imágenes CLIP...")
IMAGE_MODEL = SentenceTransformer(IMAGE_EMBED_MODEL)



# ============================================================
# 2. EMBEDDERS DE QUERY
# ============================================================

@component
class MiniLMQueryEmbedder:
    """Embedder de queries con MiniLM (para recuperar TEXTO + WAYPOINTS)."""

    def __init__(self, model_name: str = TEXT_EMBED_MODEL):
        self.model = TEXT_MODEL

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
        self.model = IMAGE_MODEL

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
# 3. PREPARADOR MULTIMODAL → PARTS PARA GEMINI
# ============================================================

@component
class GeminiMultimodalPreparer:
    """
    Versión CORREGIDA que SÍ funciona con GoogleAIGeminiGenerator.
    Devuelve tipos EXACTOS que espera el generador.
    """
    
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir

    @component.output_types(parts=List[Union[str, ByteStream]])  # ← TIPO EXACTO
    def run(
        self,
        query: str,
        text_docs: List[Document],
        image_docs: List[Document],
    ):
        from typing import Union  # Añade esto si no está importado
        
        # 1) Construir contexto textual
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
                header_parts.append(f"Población: {poblacion}")
            if dificultad:
                header_parts.append(f"Dificultad: {dificultad}")
            if distancia is not None:
                header_parts.append(f"Distancia: {distancia} km")

            header = " | ".join(header_parts) if header_parts else f"Documento {i}"
            texto = (doc.content or "").replace("\n", " ")
            context_lines.append(f"{header}\n{texto[:400]}")

        context_text = "\n\n".join(context_lines) if context_lines else "Sin contexto."

        # 2) Crear prompt FINAL (texto plano)
        prompt_text = f"""Eres un guía experto en rutas de montaña.

CONTEXTO DISPONIBLE:
{context_text}

PREGUNTA DEL USUARIO: {query}

INSTRUCCIONES:
- Responde en español de forma clara
- Usa la información del contexto
- Si hay imágenes, descríbelas brevemente

RESPUESTA:"""

        # 3) Crear lista de partes (IMPORTANTE: tipos exactos)
        parts: List[Union[str, ByteStream]] = []  # ← Tipo explícito
        parts.append(prompt_text)  # ← str
        
        # 4) Añadir imágenes como ByteStream
        images_added = 0
        for img_doc in image_docs[:4]:  # máximo 4 imágenes
            if images_added >= 4:
                break
                
            meta = img_doc.meta or {}
            img_rel = meta.get("image_path")
            if not img_rel:
                continue

            img_path = (self.base_dir / img_rel).resolve()
            if not img_path.exists():
                print(f"⚠️ Imagen no encontrada: {img_path}")
                continue

            try:
                data = img_path.read_bytes()
                # Determinar MIME type
                suffix = img_path.suffix.lower()
                mime = "image/png" if suffix == ".png" else "image/jpeg"
                
                # Crear ByteStream (tipo EXACTO que espera Gemini)
                byte_stream = ByteStream(data=data, mime_type=mime)
                parts.append(byte_stream)  # ← ByteStream
                images_added += 1
                
            except Exception as e:
                print(f"⚠️ Error cargando {img_path}: {e}")
                continue

        print(f"📦 Preparador: 1 texto + {images_added} imágenes")
        return {"parts": parts}

# ============================================================
# 4. CONSTRUCCIÓN DEL PIPELINE MULTIMODAL
# ============================================================

def build_multimodal_rag_pipeline(
    top_k_text: int = 5,
    top_k_images: int = 4,
) -> Pipeline:
    # Document stores: misma BD, distintas colecciones
    text_store = ChromaDocumentStore(
        persist_path=str(CHROMA_PERSIST_DIR),
        collection_name=TEXT_COLLECTION,
        distance_function="cosine",
        embedding_function="default", 
    )

    image_store = ChromaDocumentStore(
        persist_path=str(CHROMA_PERSIST_DIR),
        collection_name=IMAGE_COLLECTION,
        distance_function="cosine",
        embedding_function="default",  
    )

    text_embedder = MiniLMQueryEmbedder()
    image_embedder = CLIPQueryEmbedder()


    # Embedding Retriever compatible with the Chroma Document Store.
    text_retriever = ChromaEmbeddingRetriever(document_store=text_store, top_k=top_k_text)
    image_retriever = ChromaEmbeddingRetriever(document_store=image_store, top_k=top_k_images)


    preparer = GeminiMultimodalPreparer(base_dir=BASE_DIR)

    gemini = GoogleAIGeminiGenerator(
        model="gemini-2.0-flash",   # o gemini-2.5-flash / gemini-1.5-pro, etc.
        # api_key se toma de la env var GOOGLE_API_KEY por defecto
    )

    pipe = Pipeline()
    pipe.add_component("text_embedder", text_embedder)
    pipe.add_component("image_embedder", image_embedder)
    pipe.add_component("text_retriever", text_retriever)
    pipe.add_component("image_retriever", image_retriever)
    pipe.add_component("preparer", preparer)
    pipe.add_component("generator", gemini)

    # conexiones de embeddings → retrievers
    pipe.connect("text_embedder.embedding", "text_retriever.query_embedding")
    pipe.connect("image_embedder.embedding", "image_retriever.query_embedding")

    # documentos → preparer
    pipe.connect("text_retriever.documents", "preparer.text_docs")
    pipe.connect("image_retriever.documents", "preparer.image_docs")

    # parts multimodales → Gemini
    pipe.connect("preparer.prompt", "generator.prompt")

    return pipe


# ============================================================
# 5. MODO INTERACTIVO
# ============================================================

def interactive_multimodal_chat():
    pipe = build_multimodal_rag_pipeline(top_k_text=5, top_k_images=4)

    print("\n💬 Chat RAG MULTIMODAL (texto + imágenes con Gemini).")
    print("   Escribe 'salir' para terminar.\n")

    while True:
        query = input("❓ Pregunta: ").strip()
        if not query:
            continue
        if query.lower() in {"salir", "exit", "quit"}:
            print("👋 Saliendo.")
            break

        filters_el_bruc = {
            "field": "meta.poblacion_cercana",
            "operator": "==",
            "value": "El Bruc",
        }

        result = pipe.run(
            {
                "text_embedder": {"query": query},
                "image_embedder": {"query": query},
                # 👇 aquí aplicamos el filtro al retriever de texto e imágenes
                "text_retriever": {"filters": filters_el_bruc},
                "image_retriever": {"filters": filters_el_bruc},
                "preparer": {"query": query},
            }
        )

        replies = result["generator"]["replies"]
        answer = replies[0]

        print("\n🧠 Respuesta del modelo:\n")
        print(answer)
        print("\n" + "=" * 80)

        # Opcional: inspeccionar qué docs ha usado
        text_docs = result["text_retriever"]["documents"]
        print("\n📚 (DEBUG) Algunos documentos de texto recuperados:\n")
        for i, doc in enumerate(text_docs[:3], start=1):
            ruta = doc.meta.get("nombre_ruta")
            poblacion = doc.meta.get("poblacion_cercana")
            print(f"[{i}] ruta={ruta} | poblacion_cercana={poblacion}")
        print("\n" + "=" * 80)


if __name__ == "__main__":
    interactive_multimodal_chat()