from docling.document_converter import DocumentConverter

source =r"C:\Users\jessi\OneDrive\Escritorio\MÁSTER-UOC\TFM\TFM-Arquitectura-RAG\TFM-Arquitectura-RAG\download_pdf\RUTAS-PIRINEOS-aiguaneix-alinya-urgell_es.pdf"
    
converter = DocumentConverter()
doc = converter.convert(source).document

print(doc.export_to_markdown())  # output: "### Docling Technical Report[...]"