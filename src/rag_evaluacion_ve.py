from __future__ import annotations
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Any

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import config as cfg
import math
from config import init_process
from rag_multimodal import crear_chatbot  

# -----------------------------
# Embeddings: similitud Q-A
# -----------------------------
# ---- Perplejidad ----
MODEL_NAME = "dccuchile/bert-base-spanish-wwm-cased"
tokenizer_ppl = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=False,
    force_download=True
)
model_ppl = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
model_ppl.eval()

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b))

def embed_texts_minilm(texts: List[str], model_name: str) -> np.ndarray:
    model = SentenceTransformer(model_name)
    emb = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False
    )
    return emb

def recall_at_k(retrieved_docs, relevant_doc, k=7) -> int:
    retrieved_topk = retrieved_docs[:k]
    return int(relevant_doc in retrieved_topk)

def compute_bertscore(candidate: str, reference: str) -> Optional[float]:
    try:
        from bert_score import score as bertscore
        P, R, F1 = bertscore([candidate], [reference], lang="es", verbose=False)
        return float(F1[0].item())
    except Exception:
        return None


def compute_pseudo_perplexity(text: str) -> float:
    text = text[:800]  # limitar longitud para eficiencia
    enc = tokenizer_ppl(text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = enc["input_ids"]

    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    losses = []

    with torch.no_grad():
        for i in range(1, input_ids.size(1) - 1):
            masked_ids = input_ids.clone()
            masked_ids[0, i] = tokenizer_ppl.mask_token_id

            outputs = model_ppl(masked_ids)
            logits = outputs.logits

            loss = loss_fct(
                logits[0, i],
                input_ids[0, i]
            )
            losses.append(loss.item())

    return math.exp(sum(losses) / len(losses))


def eval_one_row(
    bot,
    question: str,
    reference_answer: str,
    k: int,
    relevance_threshold: float,
    embed_model_name: str,
    do_bertscore: bool,
    route_id: str,
    document_id: str,
    type_q: str,
) -> Dict[str, Any]:
    out = bot.preguntar(question, mostrar_info=False)
    answer = out.get("respuesta", "")
    docs_ids = out.get("doc_ids", []) or []
    route_ids_docs = out.get("route_ids_docs", []) or []
    image_ids = out.get("image_ids", []) or []
    avg_doc_score = out.get("avg_score_topk_sim", None)
    print(f" Documentos de texto recuperados (section_id): {docs_ids}")
    print(f" Route IDs de documentos de texto recuperados: {route_ids_docs}")
    print(f" Documentos de imagen recuperados (doc_id): {image_ids}")
    scores_k = out.get("scores_topk_sim", []) or []
    print(f" avg_doc_score: {avg_doc_score}")
    print(f" scores_k: {scores_k}")

    # Relevancia por query: 1 si hay >=1 doc relevante en top-k
    relevant_hits = sum(1 for s in scores_k if s >= relevance_threshold)
    relevance_binary = 1 if relevant_hits >= 1 else 0

    # Precision@k: fracción de docs relevantes en top-k
    precision_k = (relevant_hits / len(scores_k)) if scores_k else 0.0

    # Similaridad (Respuesta vs A) usando el mismo embedder
    emb = embed_texts_minilm([answer, reference_answer], embed_model_name)
    sim_q_a = cosine_sim(emb[0], emb[1])
    print(f"route_id: {route_id}, document_id: {document_id}, type_q: {type_q}")

    # Extras opcionales
    bert_f1 = compute_bertscore(answer, reference_answer) if do_bertscore else None
    if type_q == "Multimodal (basada en imagen)":
        recall = recall_at_k(image_ids, relevant_doc=document_id, k=k)   
        precision_k = (sum(1 for im in image_ids if im == document_id) / k) if (k and document_id) else 0.0
    else:
        recall = recall_at_k(docs_ids, relevant_doc=document_id, k=k)
        precision_k = (sum(1 for d in docs_ids if d == document_id) / k) if k else 0.0
        
    recall_route = recall_at_k(route_ids_docs, relevant_doc=route_id, k=k)
    precision_route_k = (sum(1 for r in route_ids_docs if r == route_id) / k) if k else 0.0

    print(f" Recall@{k}: {recall}, Recall@{k} (route_id): {recall_route}")
    print(f" Precision@{k}: {precision_k}, Precision@{k} (route_id): {precision_route_k}")
    return {
        "Respuesta_generada": answer,
        "Score de similitud (Q-A) promedio": sim_q_a,  # respuesta vs A (embeddings)
        f"Score medio docs (retriever)": avg_doc_score,
        f"Precision": precision_k,
        f"Precision (route_id)": precision_route_k,
        f"scores_top": str(scores_k),
        f"Relevancia": relevance_binary,
        "Perplejidad": None,
        "BertScore": bert_f1,
        "Recall@k": recall,
        "Recall@k (route_id)": recall_route,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="CSV con columnas: Nro Q, Q, A (y opcional Tipo de pregunta (*))")
    ap.add_argument("--output_csv", required=True, help="CSV de salida con métricas")
    ap.add_argument("--output_xlsx", default=None, help="(Opcional) XLSX de salida")
    ap.add_argument("--k", type=int, default=7)
    ap.add_argument("--thr", type=float, default=0.7)
    ap.add_argument("--compute_bertscore", action="store_true")
    ap.add_argument("--compute_ppl", action="store_true")
    args = ap.parse_args()

    in_path = Path(args.input)
    if in_path.suffix.lower() != ".csv":
        raise ValueError("El input debe ser un CSV")

    df = pd.read_csv(in_path, sep=";", encoding="utf-8-sig")
    for col in ["route_id", "document_id", "tipo_pregunta", "pregunta", "respuesta"]:
        if col not in df.columns:
            raise ValueError(f"Falta columna obligatoria: {col}. Columnas: {list(df.columns)}")
        
    # Identificador automático
    df.insert(0, "id", range(1, len(df) + 1))

    BASE_DIR = Path(__file__).resolve().parents[1]

    BASE_DIR = Path(__file__).resolve().parents[1]

    init_process(
        TEXT_COLLECTION_param="rutas_text_waypoints",
        IMAGE_COLLECTION_param="rutas_imagenes",
        TEXT_EMBED_MODEL_param="sentence-transformers/all-MiniLM-L6-v2",
        PATH_CHROMA_DIR_param=BASE_DIR / "src" / "chroma_db",
        BASE_DIR_param=BASE_DIR
    )

    bot = crear_chatbot(verbose=False,top_k_text=args.k)

    rows = []
    for _, r in df.iterrows():
        q = str(r["pregunta"])
        a = str(r["respuesta"])
        type = str(r["tipo_pregunta"])
        route_id = int(float(r["route_id"]))
        document_id = str(r["document_id"])
        rows.append(
            eval_one_row(
                bot=bot,
                question=q,
                reference_answer=a,
                type_q = type,
                k=args.k,
                relevance_threshold=args.thr,
                embed_model_name=cfg.TEXT_EMBED_MODEL,
                do_bertscore=args.compute_bertscore,
                route_id = route_id,
                document_id = document_id,
            )
        )

    met = pd.DataFrame(rows)

    # Recall@k global: fracción de queries con >=1 doc relevante en top5
    #recall_at_k = float(met["Relevancia"].mean()) if len(met) else 0.0
    #met[f"Recall@{args.k}"] = recall_at_k

    out_df = pd.concat([df.reset_index(drop=True), met.reset_index(drop=True)], axis=1)

    # Columnas manuales (si no existen)
    manual_cols = [
        "Exactitud factual / Fidelidad (0–3)",
        "Utilidad / Relevancia (0–3)",
        "Fluidez / Calidad del lenguaje (0–2)",
    ]
    for c in manual_cols:
        if c not in out_df.columns:
            out_df[c] = np.nan

    out_csv = Path(args.output_csv)
    out_df.to_csv(out_csv, index=False, encoding="utf-8-sig", sep=";")

    if args.output_xlsx:
        out_df.to_excel(Path(args.output_xlsx), index=False)

    print(f"OK -> {out_csv}")


if __name__ == "__main__":
    main()
