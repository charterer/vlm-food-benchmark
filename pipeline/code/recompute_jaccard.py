#!/usr/bin/env python3
"""Recompute Jaccard scores in an evaluation CSV using updated stopwords."""
import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple, Optional


_STOPWORDS = {
    "and", "with", "of", "a", "an", "the", "on", "in", "to", "for",
    "some", "type", "kind", "few", "several", "variety", "assortment",
    "piece", "pieces", "slice", "slices", "stick", "sticks", "semi", "circles",
    "light", "small", "big", "little", "lot", "seems", "appears",
    "salad", "dressing", "vinaigrette", "sauce", "coating",
    "cooked", "raw", "fried", "grilled", "roasted", "baked", "mashed", "sauteed",
    "dark", "pale", "fresh",
    "white", "black", "brown", "red", "green", "yellow", "orange", "purple", "blue", "gold", "golden",
    "creamy", "buttery", "zesty", "savory", "sweet", "tangy", "spicy", "mild",
    "wild", "mixed", "assorted", "varied",
    "sliced", "diced", "chopped", "cubed", "minced", "shredded", "grated", "julienned",
    "whole", "halved", "quartered", "peeled", "roasted", "toasted",
    "ripe", "fresh", "dry", "dried", "seasoned", "plain",
    # User-specified exclusions
    "dish", "asparagu", "strip", "herb", "breast", "or", "stir", "breaded", "steamed", "scrambled",
    "couscou", "cube", "leafy", "chunk", "floret", "meat", "cob", "mustard", "seed", "country",
    "ground", "steak", "greek", "sweetpotato", "berry", "olive", "thigh", "slice", "roasted",
    "vegetables", "salad", "wild", "wheat", "leaves", "sauce", "dressing", "raw", "mixed", "greens",
    # Non-food placeholders / metadata-ish tokens
    "other", "others", "ingredient", "ingredients", "etc", "etcetera",
    "misc", "miscellaneous", "unknown", "none", "null", "nil",
    "na", "nan", "n", "food", "foods", "item", "items", "stuff",
    "anything", "something",
}

_SYNONYM_TOKEN = {
    "aubergine": "eggplant",
    "yams": "sweetpotato",
    "yam": "sweetpotato",
    "strawberry": "berry",
    "blueberry": "berry",
    "raspberry": "berry",
    "blackberry": "berry",
}

NONFOOD_PHRASE_RE = re.compile(
    r"^(?:n/?a|nan|none|null|nil|unknown|not\s+available|no\s+ingredients?)$",
    re.IGNORECASE,
)


def _normalize_token(token: str) -> str:
    token = token.lower()
    if token == "greens":
        return "greens"
    if token.endswith("ies") and len(token) > 3:
        token = token[:-3] + "y"
    elif token.endswith("oes") and len(token) > 3:
        token = token[:-2]
    elif token.endswith("es") and len(token) > 3:
        suffix = token[-3:]
        if suffix in {"xes", "ches", "shes", "ses", "zes"}:
            token = token[:-2]
        else:
            token = token[:-1]
    elif token.endswith("s") and len(token) > 3:
        token = token[:-1]
    if token.endswith("berry"):
        token = "berry"
    return token


def _phrase_to_tokens(phrase: str) -> List[str]:
    if NONFOOD_PHRASE_RE.match((phrase or "").strip()):
        return []
    tokens = re.findall(r"[a-zA-Z]+", phrase.lower())
    canonical_tokens: List[str] = []
    for t in tokens:
        t = _normalize_token(t)
        if not t or t in _STOPWORDS:
            continue
        t = _SYNONYM_TOKEN.get(t, t)
        canonical_tokens.append(t)
    if ("sweet" in canonical_tokens and "potato" in canonical_tokens) and "sweetpotato" not in canonical_tokens:
        canonical_tokens.append("sweetpotato")
    if ("grape" in canonical_tokens and "tomato" in canonical_tokens) and "grapetomato" not in canonical_tokens:
        canonical_tokens.append("grapetomato")
    seen = set()
    deduped = []
    for t in canonical_tokens:
        if t not in seen:
            deduped.append(t)
            seen.add(t)
    return deduped


def _canonical_phrase_key(phrase: str) -> str:
    toks = _phrase_to_tokens(phrase)
    return " ".join(sorted(toks))


def compute_jaccard(true_items: Iterable[str], predicted_items: Iterable[str]) -> float:
    true_list = []
    seen_true = set()
    for x in true_items:
        k = _canonical_phrase_key(x)
        if not k:
            continue
        if k not in seen_true:
            true_list.append((x, set(_phrase_to_tokens(x))))
            seen_true.add(k)

    pred_list = []
    seen_pred = set()
    for y in predicted_items:
        k = _canonical_phrase_key(y)
        if not k:
            continue
        if k not in seen_pred:
            pred_list.append((y, set(_phrase_to_tokens(y))))
            seen_pred.add(k)

    n_true = len(true_list)
    n_pred = len(pred_list)
    if n_true == 0 and n_pred == 0:
        return float("nan")

    matched_true = set()
    match_count = 0
    for _, pred_tokens in pred_list:
        best_true_idx = -1
        best_overlap = 0
        for idx, (_, true_tokens) in enumerate(true_list):
            if idx in matched_true:
                continue
            overlap = len(pred_tokens & true_tokens)
            if overlap > best_overlap:
                best_overlap = overlap
                best_true_idx = idx
        if best_true_idx >= 0 and best_overlap > 0:
            matched_true.add(best_true_idx)
            match_count += 1

    union = n_true + n_pred - match_count
    if union == 0:
        return float("nan")
    return match_count / union


def main() -> int:
    parser = argparse.ArgumentParser(description="Recompute Jaccard scores using updated stopwords.")
    parser.add_argument("--input-csv", type=Path, required=True, help="Original evaluation CSV.")
    parser.add_argument("--cache-dir", type=Path, required=True, help="Gemini cache directory.")
    parser.add_argument("--output-csv", type=Path, required=True, help="Output CSV with updated Jaccard.")
    args = parser.parse_args()

    csv_path = args.input_csv.expanduser().resolve()
    cache_dir = args.cache_dir.expanduser().resolve()
    output_path = args.output_csv.expanduser().resolve()

    if not csv_path.exists():
        raise SystemExit(f"Input CSV not found: {csv_path}")
    if not cache_dir.exists():
        raise SystemExit(f"Cache directory not found: {cache_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows_out = []
    with csv_path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames
        for row in reader:
            dish_id = row.get("dish_id", "").strip()
            true_ing_str = row.get("true_ingredients", "")
            true_ingredients = [s.strip() for s in true_ing_str.split("|") if s.strip()]

            # Load AI ingredients from cache
            cache_file = cache_dir / f"{dish_id}.json"
            ai_ingredients = []
            try:
                if cache_file.exists():
                    with cache_file.open("r") as cf:
                        data = json.load(cf)
                    analysis = data.get("analysis", data) if isinstance(data, dict) else data
                    if isinstance(analysis, dict):
                        items = analysis.get("items") or []
                        for item in items:
                            if isinstance(item, dict) and item.get("name"):
                                ai_ingredients.append(str(item["name"]))
            except Exception:
                pass

            # Recompute Jaccard
            new_jaccard = compute_jaccard(true_ingredients, ai_ingredients)
            row["ingredient_jaccard"] = f"{new_jaccard:.3f}" if not math.isnan(new_jaccard) else ""
            rows_out.append(row)

    with output_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"Recomputed Jaccard for {len(rows_out)} rows.")
    print(f"Updated CSV written to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


