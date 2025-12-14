from pathlib import Path
from collections import Counter
from haystack_integrations.document_stores.chroma import ChromaDocumentStore

CHROMA_PERSIST_DIR = Path(__file__).resolve().parents[1] / "src" / "chroma_db"

image_store = ChromaDocumentStore(
    persist_path=str(CHROMA_PERSIST_DIR),
    collection_name="rutas_imagenes",
    distance_function="cosine",
)

# 🔥 Truco: filtro vacío que devuelve TODOS los documentos
docs = image_store.filter_documents(filters={})

print("Total image docs:", len(docs))
print("-" * 60)

route_ids = []
doc_ids = []

for d in docs:
    meta = d.meta or {}
    route_ids.append(meta.get("route_id"))
    doc_ids.append(meta.get("doc_id"))

from collections import Counter
print("📊 route_id más frecuentes:")
print(Counter(route_ids).most_common(10))

print("\n📊 doc_id más frecuentes:")
print(Counter(doc_ids).most_common(10))

print("\n🧪 Muestra de 10 documentos:")
for d in docs[:10]:
    print(
        "route_id:", d.meta.get("route_id"),
        "| doc_id:", d.meta.get("doc_id"),
        "| nombre_ruta:", d.meta.get("nombre_ruta_org"),
        "| image_path:", d.meta.get("image_path"),
    )
