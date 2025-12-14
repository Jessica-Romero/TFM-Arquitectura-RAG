from haystack import Pipeline
from haystack.components.converters import PyPDFToDocument
from haystack.components.converters.image import ImageFileToDocument
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.embedders.image import SentenceTransformersDocumentImageEmbedder
from haystack.components.joiners import DocumentJoiner
from haystack.components.preprocessors.document_splitter import DocumentSplitter
from haystack.components.routers.file_type_router import FileTypeRouter
from haystack.components.writers.document_writer import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.embedders import SentenceTransformersTextEmbedder


# Initialize the embedders
text_embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/clip-ViT-L-14", progress_bar=False)

# Warm up the models to load them
text_embedder.warm_up()


# Create our document store
doc_store = InMemoryDocumentStore(embedding_similarity_function="cosine")

# Define our components
file_type_router = FileTypeRouter(mime_types=["application/pdf", "image/png"])
# Define componente para unir el resto de docs procesados
final_doc_joiner = DocumentJoiner(sort_by_score=False)
image_converter = ImageFileToDocument()
pdf_converter = PyPDFToDocument()
pdf_splitter = DocumentSplitter(split_by="page", split_length=1)  
text_doc_embedder = SentenceTransformersDocumentEmbedder(
    model="sentence-transformers/clip-ViT-L-14", progress_bar=False
)
image_embedder = SentenceTransformersDocumentImageEmbedder(
    model="sentence-transformers/clip-ViT-L-14",
    progress_bar=False,
    root_path=r"C:\Users\jessi\OneDrive\Escritorio\MÁSTER-UOC\TFM\TFM-Arquitectura-RAG\TFM-Arquitectura-RAG\imagenes_extraidas"
)
document_writer = DocumentWriter(doc_store)


# Create the Indexing Pipeline
indexing_pipe = Pipeline()
indexing_pipe.add_component("file_type_router", file_type_router)
indexing_pipe.add_component("pdf_converter", pdf_converter)
indexing_pipe.add_component("pdf_splitter", pdf_splitter)
indexing_pipe.add_component("image_converter", image_converter)
indexing_pipe.add_component("text_doc_embedder", text_doc_embedder)
indexing_pipe.add_component("image_doc_embedder", image_embedder)
indexing_pipe.add_component("final_doc_joiner", final_doc_joiner)
indexing_pipe.add_component("document_writer", document_writer)

indexing_pipe.connect("file_type_router.application/pdf", "pdf_converter.sources")
indexing_pipe.connect("pdf_converter.documents", "pdf_splitter.documents")
indexing_pipe.connect("pdf_splitter.documents", "text_doc_embedder.documents")
indexing_pipe.connect("file_type_router.image/png", "image_converter.sources")
indexing_pipe.connect("image_converter.documents", "image_doc_embedder.documents")
indexing_pipe.connect("text_doc_embedder.documents", "final_doc_joiner.documents")
indexing_pipe.connect("image_doc_embedder.documents", "final_doc_joiner.documents")
indexing_pipe.connect("final_doc_joiner.documents", "document_writer.documents")

#indexing_pipe.draw()

indexing_result = indexing_pipe.run(
    data={"file_type_router": {"sources": [r"C:\Users\jessi\OneDrive\Escritorio\MÁSTER-UOC\TFM\TFM-Arquitectura-RAG\TFM-Arquitectura-RAG\download_pdf\RUTAS-PIRINEOS-aiguaneix-alinya-urgell_es.pdf",
                                          r"C:\Users\jessi\OneDrive\Escritorio\MÁSTER-UOC\TFM\TFM-Arquitectura-RAG\TFM-Arquitectura-RAG\imagenes_extraidas\RUTAS-PIRINEOS-aiguaneix-alinya-urgell_es_mapa.png"]}}
)

indexed_documents = doc_store.filter_documents()
print(f"Indexed {len(indexed_documents)} documents")


retriever = InMemoryEmbeddingRetriever(document_store=doc_store)
text_embedding = text_embedder.run(text="How long is the route Aiguaneix de Alinyà?")["embedding"]
results = retriever.run(query_embedding=text_embedding)["documents"]

for idx, doc in enumerate(results[:5]):
    print(f"Document {idx+1}:")
    print(f"Score: {doc.score}")
    print(f"File Path: {doc.meta['file_path']}")
    print("")

