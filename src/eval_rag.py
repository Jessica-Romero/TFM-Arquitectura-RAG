from pathlib import Path
import pandas as pd

from haystack import Pipeline
from haystack.dataclasses import Document
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever
from haystack_integrations.components.generators.google_ai import GoogleAIGeminiGenerator

from rag_multimodal_ve import (
    build_retrieval_pipeline,
    fetch_images_for_route_ids,
    build_multimodal_parts,
    llm_build_filters,
    BASE_DIR,
    CHROMA_PERSIST_DIR,
    TEXT_COLLECTION,
    IMAGE_COLLECTION,
)

def run_single_query(query: str, top_k_text: int = 5):
    """Ejecuta tu pipeline RAG para UNA pregunta y devuelve:
       - answer (str)
       - lista de documentos recuperados (Document) con scores
    """
    pipe = build_retrieval_pipeline(top_k_text=top_k_text)

    # Filtros con LLM (igual que en tu chat)
    filters_list = llm_build_filters(query)
    filters_dict = None
    if filters_list:
        if len(filters_list) == 1:
            filters_dict = filters_list[0]
        else:
            filters_dict = {"operator": "AND", "conditions": filters_list}

    inputs = {
        "text_embedder": {"query": query},
    }
    if filters_dict is not None:
        inputs["text_retriever"] = {"filters": filters_dict}

    result = pipe.run(inputs)
    text_docs = result["text_retriever"]["documents"]

    # Agrupar por route_id (si quieres quedarte con el mejor chunk de cada ruta)
    # Aquí asumo que ya tienes la lógica de agrupación implementada; si no, podrías usar:
    # route_groups = {}
    # for d in text_docs:
    #     rid = d.meta.get("route_id")
    #     if rid is None:
    #         continue
    #     score = d.meta.get("score", 0.0)
    #     if rid not in route_groups or score > route_groups[rid].meta.get("score", 0.0):
    #         route_groups[rid] = d
    # top_docs = list(route_groups.values())
    top_docs = text_docs  # si no agrupas todavía

    # Buscar imágenes de estos route_id (opcional, como en tu chat)
    from rag_multimodal_ve import extract_route_ids
    route_ids = extract_route_ids(top_docs)
    image_docs = fetch_images_for_route_ids(route_ids, max_images=8)

    # Construir parts para Gemini
    from rag_multimodal_ve import build_multimodal_parts
    gemini = GoogleAIGeminiGenerator(model="gemini-2.5-flash")

    parts = build_multimodal_parts(
        query=query,
        text_docs=top_docs,
        image_docs=image_docs,
        chat_history=None,
        base_dir=BASE_DIR,
    )

    gen_out = gemini.run(parts=parts)
    answer = gen_out["replies"][0]

    return answer, top_docs


def evaluar_lote_preguntas(
    questions_csv: Path,
    output_csv: Path,
    top_k_text: int = 5,
):
    df_q = pd.read_csv(questions_csv, encoding="utf-8")
    # Evaluar SOLO las dos primeras preguntas
    df_q = df_q.iloc[2:13].copy()
    rows = []

    for _, row in df_q.iterrows():
        qid = row["question_id"]
        question = row["question_text"]

        print(f"\n=== Evaluando {qid}: {question} ===")
        try:
            answer, docs = run_single_query(question, top_k_text=top_k_text)

            # Extraer doc_ids y scores
            doc_ids = []
            doc_scores = []
            for d in docs:
                doc_ids.append(str(d.meta.get("doc_id", d.id)))
                score = getattr(d, "score", None)
                doc_scores.append("" if score is None else f"{score:.6f}")


            rows.append({
                "question_id": qid,
                "question_text": question,
                "answer": answer,
                "used_doc_ids": "|".join(doc_ids),
                "used_doc_scores": "|".join(doc_scores),
            })

        except Exception as e:
            print(f"Error en {qid}: {e}")
            rows.append({
                "question_id": qid,
                "question_text": question,
                "answer": f"ERROR: {e}",
                "used_doc_ids": "",
                "used_doc_scores": "",
            })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"\n✅ Evaluación guardada en: {output_csv}")


if __name__ == "__main__":
    base = Path(__file__).resolve().parents[0]  # ajusta si hace falta
    questions_path = base / "eval_questions_rutas.csv"
    output_path = base / "eval_results_rutas.csv"

    evaluar_lote_preguntas(
        questions_csv=questions_path,
        output_csv=output_path,
        top_k_text=5,
    )
