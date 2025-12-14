import csv
import math
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from bert_score import score as bert_score

# Tu pipeline multimodal y filtros
from rag_multimodal_ve import (
    build_retrieval_pipeline,
    llm_build_filters,
    GoogleAIGeminiGenerator,
    build_prompt_from_context,
    extract_route_ids,
    fetch_images_for_route_ids,
    build_multimodal_parts,
)

# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
QA_CSV_PATH = BASE_DIR / "eval_questions_rutas.csv"
OUTPUT_CSV_PATH = BASE_DIR / "evaluacion_rag_resultados_multimodal.csv"

TOP_K = 5  # como pides
MAX_IMAGES = 4  # cuántas imágenes adjuntar realmente al LLM


# ============================================================
# MODELOS PARA MÉTRICAS EXTERNAS
# ============================================================

# ---- GPT-2 Perplexity ----
print("🔧 Cargando modelo de Perplejidad GPT-2...")
GPT2_NAME = "gpt2"
tokenizer_ppl = GPT2TokenizerFast.from_pretrained(GPT2_NAME)
model_ppl = GPT2LMHeadModel.from_pretrained(GPT2_NAME)
model_ppl.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_ppl.to(device)


def perplexity(text: str, max_length: int = 512) -> float:
    """Calcula perplejidad con GPT-2 truncando el texto."""
    if not text:
        return float("nan")

    encodings = tokenizer_ppl(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    input_ids = encodings["input_ids"].to(device)

    with torch.no_grad():
        outputs = model_ppl(input_ids, labels=input_ids)
        loss = outputs.loss

    return torch.exp(loss).item()


# ---- BERTScore ----
def compute_bertscore(pred: str, gold: str) -> float:
    """Calcula BERTScore (F1) usando XLM-R Large para español."""
    if not pred.strip() or not gold.strip():
        return float("nan")

    P, R, F1 = bert_score(
        [pred],
        [gold],
        lang="es",
        model_type="xlm-roberta-large",
        verbose=False,
    )
    return float(F1[0])


# ============================================================
# AUXILIARES
# ============================================================

def clean_text(x):
    """Evita NaN o None en textos."""
    if x is None:
        return ""
    if isinstance(x, float) and math.isnan(x):
        return ""
    return str(x).strip()


def is_doc_relevant(doc: Any, gold_answer: str) -> bool:
    """Heurística simple (ojo: muy estricta si gold es larga)."""
    if not gold_answer:
        return False
    gold = gold_answer.lower()
    content = (getattr(doc, "content", "") or "").lower()
    meta_text = " ".join(str(v) for v in (getattr(doc, "meta", None) or {}).values()).lower()
    return gold in (content + " " + meta_text)


def build_filters_for_query(query: str) -> Dict[str, Any] | None:
    """Construye filtros usando el LLM."""
    filters_list = llm_build_filters(query)
    if not filters_list:
        return None
    if len(filters_list) == 1:
        return filters_list[0]
    return {"operator": "AND", "conditions": filters_list}


# ============================================================
# EVALUACIÓN MULTIMODAL
# ============================================================

def evaluar_rag_multimodal_sobre_csv():

    # ---- Cargar CSV ----
    #df = pd.read_csv(QA_CSV_PATH, encoding="utf-8", encoding_errors="ignore")
    df = pd.read_csv( QA_CSV_PATH,
    sep=";",              # ✅ clave: tu CSV es con punto y coma
    encoding="utf-8",
    quotechar='"',
    engine="python",      # ✅ más robusto con textos largos y comillas
)

    expected_cols = {"tipo_pregunta", "pregunta", "respuesta"}
    if not expected_cols.issubset(df.columns):
        raise ValueError(f"El CSV debe contener: {expected_cols}")

    # ---- Retrieval (texto) ----
    pipe = build_retrieval_pipeline(top_k_text=TOP_K)

    # ---- Modelo generador Gemini ----
    gemini = GoogleAIGeminiGenerator(model="gemini-2.5-flash-lite")

    resultados = []

    for idx, row in df.iterrows():
        nro_q = idx + 1
        tipo = clean_text(row["tipo_pregunta"])
        pregunta = clean_text(row["pregunta"])
        respuesta_gold = clean_text(row["respuesta"])

        if not pregunta:
            print(f"⏭️ Saltando pregunta vacía en fila {idx}")
            continue

        print(f"\n=== Evaluando Q{nro_q}: {pregunta} ===")

        # 1) Filtros del LLM
        filters_dict = build_filters_for_query(pregunta)

        # 2) Retrieval de TEXTO
        inputs = {"text_embedder": {"query": pregunta}}
        if filters_dict:
            inputs["text_retriever"] = {"filters": filters_dict}

        result = pipe.run(inputs)
        docs = result["text_retriever"]["documents"] or []

        # 3) Multimodal: route_id -> imágenes
        route_ids = extract_route_ids(docs)
        image_docs = fetch_images_for_route_ids(route_ids, max_images=8)  # recupera candidates
        # (el recorte final a MAX_IMAGES lo hace build_multimodal_parts)

        # 4) Métricas de retrieval @5
        docs_k = docs[:TOP_K]
        scores = []
        for d in docs_k:
            s = getattr(d, "score", 0.0)
            scores.append(float(s) if s is not None else 0.0)

        avg_score = sum(scores) / len(scores) if scores else 0.0
        relevancia = 1 if avg_score >= 0.7 else 0  # (umbral a calibrar)

        relevancias_docs = [is_doc_relevant(d, respuesta_gold) for d in docs_k]
        recall_hit = 1 if any(relevancias_docs) else 0
        precision_at5 = (sum(relevancias_docs) / len(relevancias_docs)) if relevancias_docs else 0.0

        # 5) Generación MULTIMODAL (prompt + ByteStreams)
        #    Esto sigue tu flujo: build_prompt_from_context + adjuntar imágenes reales
        parts = build_multimodal_parts(
            query=pregunta,
            text_docs=docs,
            image_docs=image_docs,
            base_dir=BASE_DIR,     # ✅ base para resolver image_path relativo
            max_images=MAX_IMAGES, # ✅ adjunta hasta 4 imágenes al LLM
        )

        llm_out = gemini.run(parts=parts)
        respuesta_generada = llm_out["replies"][0] if llm_out and llm_out.get("replies") else ""

        # 6) Métricas “complejas”
        ppl = perplexity(respuesta_generada)
        bscore = compute_bertscore(respuesta_generada, respuesta_gold)

        # 7) Guardar resultados
        resultados.append(
            {
                "Nro Q": nro_q,
                "Tipo de pregunta": tipo,
                "Q": pregunta,
                "A_gold": respuesta_gold,
                "A_generada": respuesta_generada,
                "Score_sim_QA_top5": avg_score,
                "Relevancia": relevancia,
                "Recall@5_hit": recall_hit,
                "Precision@5": precision_at5,
                "Num_text_docs": len(docs),
                "Route_ids": ";".join(str(r) for r in route_ids),
                "Num_image_docs": len(image_docs),
                "Perplejidad": ppl,
                "BertScore": bscore,
                "Exactitud_factual": "",
                "Utilidad_relevancia": "",
                "Fluidez": "",
            }
        )

    # ---- Guardar CSV ----
    out_df = pd.DataFrame(resultados)
    out_df.to_csv(OUTPUT_CSV_PATH, index=False, quoting=csv.QUOTE_NONNUMERIC)

    print(f"\n✅ Evaluación multimodal completada y guardada en: {OUTPUT_CSV_PATH}")

    # ---- Métricas globales ----
    if len(out_df) > 0:
        print("\n===== MÉTRICAS GLOBALES (@5) =====")
        print(f"Score promedio: {out_df['Score_sim_QA_top5'].mean():.3f}")
        print(f"Recall@5 global: {out_df['Recall@5_hit'].mean():.3f}")
        print(f"Precision@5 global: {out_df['Precision@5'].mean():.3f}")
        print(f"Imágenes recuperadas (media): {out_df['Num_image_docs'].mean():.2f}")


if __name__ == "__main__":
    evaluar_rag_multimodal_sobre_csv()
