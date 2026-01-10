import pandas as pd
import os
import re
from typing import List, Dict
from transformers import pipeline
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")




def load_hypotheses(csv_path: str) -> pd.DataFrame:
    """
    Load test(1).csv or train(3).csv containing hypothetical backstories.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    return df


def extract_claims_from_df(df: pd.DataFrame) -> List[Dict]:
    """
    Convert dataframe rows into structured claim/hypothesis objects.
    """
    claims = []

    for idx, row in df.iterrows():
        claim = {
            "id": row.get("id", idx),
            "character": row.get("char"),
            "book": row.get("book_name"),
            "caption": row.get("caption"),
            "content": row.get("content")
        }
        claims.append(claim)

    return claims


import re

def extract_claims_from_text(backstory_text):
    """
    Split backstory into small factual claims.
    """
    sentences = re.split(r'[.!?]', backstory_text)
    claims = []

    for s in sentences:
        s = s.strip()
        if len(s) < 15:
            continue

        sub_parts = re.split(r'\band\b|\bbut\b|\bwhile\b|,', s)

        for part in sub_parts:
            part = part.strip()
            if len(part) > 15:
                claims.append(part)

    return claims

def retrieve_candidate_evidence(
    claim_text,
    novel_chunks,
    chunk_embeddings,
    top_k=5
):
    claim_embedding = embedding_model.encode(
        [claim_text],
        convert_to_numpy=True
    )

    similarities = cosine_similarity(
        claim_embedding,
        chunk_embeddings
    )[0]

    top_indices = similarities.argsort()[-top_k:][::-1]
    return [novel_chunks[i] for i in top_indices]




def check_claim_vs_evidence(claim_text: str, evidence_text: str):
    """
    Runs NLI between evidence (premise) and claim (hypothesis).

    Returns:
    {
        "label": "ENTAILMENT" | "CONTRADICTION" | "NEUTRAL",
        "score": float
    }
    """

    labels = ["ENTAILMENT", "CONTRADICTION", "NEUTRAL"]

    output = nli_pipeline(
        sequences=evidence_text,
        candidate_labels=labels,
        hypothesis_template="This statement implies that {}."
    )

    top_label = output["labels"][0]
    top_score = output["scores"][0]

    return {
        "label": top_label,
        "score": float(top_score)
    }

def aggregate_nli_results(nli_results):
    """
    Aggregate multiple NLI results into a single final decision.

    Strategy:
    - If any strong CONTRADICTION exists → CONTRADICTION
    - Else if any strong ENTAILMENT exists → ENTAILMENT
    - Else → NEUTRAL

    Returns:
    {
        "final_label": str,
        "confidence": float
    }
    """

    if not nli_results:
        return {
            "final_label": "NEUTRAL",
            "confidence": 0.0
        }

    # Track best scores per label
    best_scores = {
        "ENTAILMENT": 0.0,
        "CONTRADICTION": 0.0,
        "NEUTRAL": 0.0
    }

    for res in nli_results:
        label = res["label"]
        score = res["score"]
        if label in best_scores:
            best_scores[label] = max(best_scores[label], score)

    # Decision logic (very important)
    if best_scores["CONTRADICTION"] >= 0.5:
        return {
            "final_label": "CONTRADICTION",
            "confidence": best_scores["CONTRADICTION"]
        }

    if best_scores["ENTAILMENT"] >= 0.5:
        return {
            "final_label": "ENTAILMENT",
            "confidence": best_scores["ENTAILMENT"]
        }

    return {
        "final_label": "NEUTRAL",
        "confidence": max(best_scores.values())
    }

# ================================
# Embedding utilities (STEP 1.7)
# ================================

def build_chunk_embeddings(chunks, model_name="all-MiniLM-L6-v2"):
    embeddings = embedding_model.encode(
        chunks,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    return embeddings



def load_or_create_embeddings(chunks, cache_path):
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    embeddings = build_chunk_embeddings(chunks)

    with open(cache_path, "wb") as f:
        pickle.dump(embeddings, f)

    return embeddings


