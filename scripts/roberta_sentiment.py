#!/usr/bin/env python3
"""Generic RoBERTa-large financial sentiment runner.

Loads soleimanian/financial-roberta-large-sentiment via a transformers
pipeline on CPU and scores an arbitrary list of English financial texts
(any market, any source).

Usage:
    python scripts/roberta_sentiment.py --input texts.txt [--label my-run]
    python scripts/roberta_sentiment.py --input texts.json [--label my-run]

Input formats:
  - .txt: one text per line (blank lines skipped)
  - .json: a JSON list of strings (or objects with a "text" field)

Outputs (into results/):
  - sentiment.json: per-item label + probabilities, label counts, mean probs
  - report.md: human-readable table of the same

Labels are mapped by the model's own id2label. The canonical keys are
positive / negative / neutral; the raw model label names are preserved
per item as well, in case they differ.
"""
import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json
from collections import Counter
from pathlib import Path

MODEL_ID = "soleimanian/financial-roberta-large-sentiment"
CANONICAL_LABELS = ("positive", "negative", "neutral")


def canonical_label(raw_label: str) -> str:
    low = str(raw_label).strip().lower()
    if "pos" in low:
        return "positive"
    if "neg" in low:
        return "negative"
    if "neu" in low:
        return "neutral"
    return low


def load_texts(path: Path):
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        texts = []
        for item in data:
            if isinstance(item, str):
                texts.append(item)
            elif isinstance(item, dict) and "text" in item:
                texts.append(str(item["text"]))
            else:
                raise ValueError(f"Unsupported JSON item: {item!r}")
        return [t for t in texts if t.strip()]
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description="Run RoBERTa-large financial sentiment over a list of texts.")
    parser.add_argument("--input", required=True, help="Path to a .txt (one text per line) or .json (list) file")
    parser.add_argument("--label", default="run", help="Label for this run (recorded in the outputs)")
    args = parser.parse_args()

    from transformers import AutoConfig, pipeline

    texts = load_texts(Path(args.input))
    if not texts:
        raise SystemExit("No texts found in input.")

    # The model's own id2label is the source of truth for label names.
    config = AutoConfig.from_pretrained(MODEL_ID)
    id2label = {int(k): str(v) for k, v in (getattr(config, "id2label", {}) or {}).items()}
    print(f"Model: {MODEL_ID}")
    print(f"id2label: {id2label}")

    clf = pipeline("sentiment-analysis", model=MODEL_ID, top_k=None, device=-1)
    raw_outputs = clf(texts)
    # pipeline with top_k=None returns a list of lists; normalise a single dict.
    if isinstance(raw_outputs, dict):
        raw_outputs = [[raw_outputs]]
    elif raw_outputs and isinstance(raw_outputs[0], dict):
        raw_outputs = [[o] for o in raw_outputs]

    items = []
    for text, scores in zip(texts, raw_outputs):
        raw_probs = {}
        probs = {label: 0.0 for label in CANONICAL_LABELS}
        for entry in scores:
            raw_label = str(entry["label"])
            score = float(entry["score"])
            raw_probs[raw_label] = score
            canon = canonical_label(raw_label)
            if canon in probs:
                probs[canon] = score
            else:
                # Unknown label name: keep it in the canonical dict too.
                probs[canon] = score
        # Predicted = highest-probability canonical label.
        predicted = max(probs, key=probs.get)
        # Raw label corresponding to the prediction, if it differs.
        raw_predicted = predicted
        for raw_label, score in raw_probs.items():
            if score == probs[predicted] and canonical_label(raw_label) == predicted:
                raw_predicted = raw_label
                break
        items.append({
            "text": text,
            "predicted": predicted,
            "raw_predicted_label": raw_predicted,
            "probabilities": probs,
            "raw_probabilities": raw_probs,
        })

    # Aggregate over the canonical three labels (plus any extras seen).
    all_labels = list(CANONICAL_LABELS)
    for item in items:
        for key in item["probabilities"]:
            if key not in all_labels:
                all_labels.append(key)
    counts = Counter(item["predicted"] for item in items)
    means = {label: sum(item["probabilities"].get(label, 0.0) for item in items) / len(items) for label in all_labels}

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "label": args.label,
        "model": MODEL_ID,
        "id2label": {str(k): v for k, v in id2label.items()},
        "count": len(items),
        "label_counts": {label: counts.get(label, 0) for label in all_labels},
        "mean_probabilities": means,
        "items": items,
    }
    (results_dir / "sentiment.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        f"# RoBERTa financial sentiment — {args.label}",
        "",
        f"- Model: `{MODEL_ID}` (CPU)",
        f"- id2label: `{id2label}`",
        f"- Texts scored: {len(items)}",
        "- Label counts: " + ", ".join(f"{label}={counts.get(label, 0)}" for label in all_labels),
        "- Mean probabilities: " + ", ".join(f"{label}={means[label]:.4f}" for label in all_labels),
        "",
        "| # | Predicted | Raw label | P(pos) | P(neg) | P(neu) | Text |",
        "|---|-----------|-----------|--------|--------|--------|------|",
    ]
    for i, item in enumerate(items, 1):
        p = item["probabilities"]
        text = item["text"].replace("|", "\\|")
        lines.append(
            f"| {i} | {item['predicted']} | {item['raw_predicted_label']} | "
            f"{p.get('positive', 0.0):.4f} | {p.get('negative', 0.0):.4f} | {p.get('neutral', 0.0):.4f} | {text} |"
        )
    lines.append("")
    (results_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"Scored {len(items)} texts | counts={dict(counts)} | means=" +
          ", ".join(f"{label}={means[label]:.4f}" for label in all_labels))


if __name__ == "__main__":
    main()
