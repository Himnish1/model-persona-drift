"""
embed_and_score.py

Embeds model responses to base and variant prompts using a sentence-transformer
model and computes pairwise cosine similarities.

Expected input (variants.json produced by generate_variants.py):
  [
    {
      "id": "<base_id>",
      "base": "<original prompt>",
      "variants": {
        "noise": "...",
        "adversarial": "...",
        "emotional": "...",
        "jailbreak": "...",
        "irrelevant": "..."
      },
      "responses": {               # optional – see --responses-file
        "base": "...",
        "noise": "...",
        ...
      }
    }
  ]

If responses are not yet embedded in variants.json you can supply them via a
separate JSON file (--responses-file) that follows the same structure.

Output (results/similarity_scores.json):
  [
    {
      "id": "p001",
      "scores": {
        "noise":        0.94,
        "adversarial":  0.81,
        "emotional":    0.88,
        "jailbreak":    0.72,
        "irrelevant":   0.91
      }
    },
    ...
  ]

A summary CSV is also written to results/similarity_scores.csv.

Usage:
    python scripts/embed_and_score.py \
        [--variants    data/variants.json] \
        [--responses-file  data/responses.json] \
        [--output-dir  results/] \
        [--model       all-MiniLM-L6-v2]
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Return the cosine similarity between two 1-D vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def embed_texts(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    """Return a 2-D numpy array of embeddings for *texts*."""
    return model.encode(texts, convert_to_numpy=True, show_progress_bar=False)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def load_responses(variants: list[dict], responses_file: Path | None) -> list[dict]:
    """
    Merge an optional external responses file into *variants*.

    If *responses_file* is provided its ``responses`` field takes precedence
    over anything already in *variants*.
    """
    if responses_file is None:
        return variants

    with open(responses_file, encoding="utf-8") as f:
        extra = {r["id"]: r.get("responses", {}) for r in json.load(f)}

    for record in variants:
        if record["id"] in extra:
            record.setdefault("responses", {}).update(extra[record["id"]])

    return variants


def score_variants(
    variants: list[dict],
    model: SentenceTransformer,
) -> list[dict]:
    """
    Compute cosine similarity between base and variant *responses*.

    Falls back to comparing the *prompts* themselves when responses are absent,
    so the script can be exercised without a live LLM.

    All texts for a single prompt record are embedded in one batched call to
    take advantage of the model's batch processing.
    """
    results = []
    for record in variants:
        responses = record.get("responses", {})
        prompt_variants = record.get("variants", {})

        base_text = responses.get("base") or record["base"]
        methods = list(prompt_variants.keys())
        variant_texts = [
            responses.get(m) or prompt_variants[m] for m in methods
        ]

        # Embed base + all variants in a single batched call
        all_texts = [base_text] + variant_texts
        all_embeddings = embed_texts(model, all_texts)
        base_embedding = all_embeddings[0]

        scores = {
            method: round(cosine_similarity(base_embedding, all_embeddings[i + 1]), 6)
            for i, method in enumerate(methods)
        }
        results.append({"id": record["id"], "scores": scores})

    return results


def write_csv(scores: list[dict], path: Path) -> None:
    """Write *scores* to a CSV file at *path*."""
    if not scores:
        return

    methods = list(scores[0]["scores"].keys())
    fieldnames = ["id"] + methods

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in scores:
            row = {"id": record["id"]}
            row.update(record["scores"])
            writer.writerow(row)


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(
        description="Embed responses and compute cosine similarity scores."
    )
    parser.add_argument(
        "--variants",
        type=Path,
        default=repo_root / "data" / "variants.json",
        help="Path to variants JSON (default: data/variants.json)",
    )
    parser.add_argument(
        "--responses-file",
        type=Path,
        default=None,
        help="Optional separate responses JSON file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "results",
        help="Directory for output files (default: results/)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Sentence-transformer model name (default: all-MiniLM-L6-v2)",
    )
    args = parser.parse_args()

    print(f"Loading sentence-transformer model: {args.model}")
    model = SentenceTransformer(args.model)

    with open(args.variants, encoding="utf-8") as f:
        variants = json.load(f)

    variants = load_responses(variants, args.responses_file)

    print(f"Scoring {len(variants)} prompt(s)…")
    scores = score_variants(variants, model)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    json_path = args.output_dir / "similarity_scores.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2, ensure_ascii=False)
    print(f"JSON results → {json_path}")

    csv_path = args.output_dir / "similarity_scores.csv"
    write_csv(scores, csv_path)
    print(f"CSV results  → {csv_path}")

    # Print a quick summary table
    if scores:
        methods = list(scores[0]["scores"].keys())
        header = f"{'ID':8s} " + " ".join(f"{m:12s}" for m in methods)
        print("\n" + header)
        print("-" * len(header))
        for record in scores:
            row = f"{record['id']:8s} "
            row += " ".join(f"{record['scores'][m]:12.4f}" for m in methods)
            print(row)


if __name__ == "__main__":
    main()
