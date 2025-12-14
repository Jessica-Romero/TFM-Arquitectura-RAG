import csv
from pathlib import Path
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer
from haystack.dataclasses import Document as HDocument


EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"  

def chunk_pdf_by_subtitle(pdf_path: str | Path) -> list[HDocument]:
    try:
        pdf_path = Path(pdf_path)
        
        # Verificar que el archivo existe
        if not pdf_path.exists():
            raise FileNotFoundError(f"El archivo {pdf_path} no existe")
        
        print(f"Procesando: {pdf_path.name}")

        # 1) Convertir el PDF a DoclingDocument
        print("Convirtiendo PDF...")
        conversion_result = DocumentConverter().convert(source=str(pdf_path))
        dl_doc = conversion_result.document
        print("PDF convertido exitosamente")

        # 2) Configurar tokenizer 
        print("Configurando tokenizer...")
        tokenizer = HuggingFaceTokenizer(
            tokenizer=AutoTokenizer.from_pretrained(EMBED_MODEL_ID),
            max_tokens=2048,  # límite grande para NO trocear demasiado
        )

        # 3) HybridChunker: respeta la estructura del documento  
        chunker = HybridChunker(
            tokenizer=tokenizer,
            merge_peers=True,   
        )

        print("Dividiendo en chunks...")
        chunks = list(chunker.chunk(dl_doc=dl_doc))
        print(f"Se generaron {len(chunks)} chunks")

        docs: list[HDocument] = []

        for i, ch in enumerate(chunks):
            # ch.text = texto del chunk
            # ch.headings = lista de headings desde el nivel superior hasta el actual
            heading = None
            heading_level = None
            if hasattr(ch, "headings") and ch.headings:
                last = ch.headings[-1]
                heading = last.text
                heading_level = last.level

            meta = {
                "source": str(pdf_path),
                "chunk_id": i,
                "heading": heading,
                "heading_level": heading_level,
                "page_start": getattr(ch, "page_start", None),
                "page_end": getattr(ch, "page_end", None),
            }

            docs.append(
                HDocument(
                    id=f"{pdf_path.stem}-chunk-{i}",
                    content=ch.text,
                    meta=meta,
                )
            )

        return docs

    except Exception as e:
        print(f"Error en chunk_pdf_by_subtitle: {e}")
        return []
    
def save_to_csv(docs: list[HDocument], csv_path: str | Path):
    """
    Guarda los chuncks en un CSV con columnas:
      - section_id
      - heading
      - pages
      - text
    """
    csv_path = Path(csv_path)
    print(f"Guardando CSV en: {csv_path}")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["section_id", "heading", "pages","text"])
        for doc in docs:
            section_id = doc.id
            heading = doc.meta.get("heading")
            page_start = doc.meta.get("page_start")
            page_end = doc.meta.get("page_end")
            pages = f"{page_start} - {page_end}"
            
            text = (doc.content or "").replace("\n", " ").strip()

            writer.writerow([section_id, heading, pages, text])



    print("CSV generado correctamente.")



if __name__ == "__main__":
    pdf_path = r"C:\Users\jessi\OneDrive\Escritorio\MÁSTER-UOC\TFM\TFM-Arquitectura-RAG\TFM-Arquitectura-RAG\download_pdf\RUTAS-PIRINEOS-aiguaneix-alinya-urgell_es.pdf"
    
    doc_list = chunk_pdf_by_subtitle(pdf_path)
    
    if doc_list:
        print(f"\n Chunking completado. Se generaron {len(doc_list)} documentos.")
        print("\nPrimeros 3 chunks:")
        for i, doc in enumerate(doc_list):
            print(f"\n--- Chunk {i+1} ---")
            print(f"ID: {doc.id}")
            print(f"Heading: {doc.meta.get('heading')} (Level: {doc.meta.get('heading_level')})")
            print(f"Pages: {doc.meta.get('page_start')} - {doc.meta.get('page_end')}")
            print(f"Content Preview: {doc.content[:150]}...")
            print("-" * 40)
    else:
        print("No se generaron chunks. Revisa el error anterior.")
    
    save_to_csv(doc_list,r".\chunks_rutas_aiguaneix_por_subtitulos.csv")
