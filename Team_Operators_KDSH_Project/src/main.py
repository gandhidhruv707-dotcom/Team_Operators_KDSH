# src/main.py
import csv
import os

from material import load_all_novels
from processing import (
    load_hypotheses,
    extract_claims_from_df,
    retrieve_candidate_evidence,
    check_claim_vs_evidence,
    aggregate_nli_results,
    load_or_create_embeddings
)




if __name__ == "__main__":

    # STEP 1: Resolve base paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    BOOKS_DIR = os.path.join(BASE_DIR, "..", "Data", "books")
    CSV_PATH = os.path.join(BASE_DIR, "..", "Data", "csv", "test (1).csv")

    # STEP 2: Load novels + chunks
    print("Loading novels...")
    novels = load_all_novels(BOOKS_DIR, chunk_size=5)

    all_chunks = []
    for chunks in novels.values():
        all_chunks.extend(chunks)

    print(f"Total evidence chunks loaded: {len(all_chunks)}")

    # ----------------------------
    # STEP 2b: Load/create embeddings
    # ----------------------------
    CACHE_PATH = os.path.join(BASE_DIR, "..", "output", "chunk_embeddings.pkl")

    print("Loading / creating chunk embeddings...")
    
   chunk_embeddings = load_or_create_embeddings(
    all_chunks,
    CACHE_PATH,
)


    print("Embeddings ready.")

    # ----------------------------
    # STEP 3: Load hypotheses CSV
    # ----------------------------
    df = load_hypotheses(CSV_PATH)
    hypotheses = extract_claims_from_df(df)

    print(f"Total hypotheses loaded: {len(hypotheses)}")

    # ----------------------------
    # STEP 4: Setup output directory
    # ----------------------------
    OUTPUT_DIR = os.path.join(BASE_DIR, ".", "output")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    OUTPUT_PATH = os.path.join(OUTPUT_DIR, "results_test.csv")

    results = []

    print("\nRunning full hypothesis evaluation...\n")

    for idx, hyp in enumerate(hypotheses, start=1):
        claim_text = hyp["content"]

        print(f"[{idx}/{len(hypotheses)}] Processing hypothesis ID {hyp['id']}")

       candidate_evidence = retrieve_candidate_evidence(
    claim_text,
    all_chunks,
    chunk_embeddings,
    top_k=3
)


        # 2. Run NLI
            nli_results = []

            for ev in candidate_evidence:
        res = check_claim_vs_evidence(claim_text, ev)
        nli_results.append(res)


        # 3. Aggregate decision
        final_decision = aggregate_nli_results(nli_results)

        # 4. Store result
        results.append({
            "hypothesis_id": hyp["id"],
            "book": hyp["book"],
            "character": hyp["character"],
            "caption": hyp["caption"],
            "claim": claim_text,
            "final_label": final_decision["final_label"],
            "confidence": final_decision["confidence"],
            "evidence_used": " ||| ".join(candidate_evidence)
        })

    # ----------------------------
    # Save results to CSV
    # ----------------------------
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=results[0].keys()
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\nDone. Results saved to: {OUTPUT_PATH}")
