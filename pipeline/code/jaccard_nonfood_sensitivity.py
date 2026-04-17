#!/usr/bin/env python3
"""
Recompute ingredient Jaccard with expanded non-food filters.

This script leaves existing evaluation code untouched and runs a
side-by-side comparison against the current Jaccard implementation.
"""

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


_SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(_SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_ROOT))

import evaluate_nutrition5k as ev  # noqa: E402


EXTRA_STOPWORDS = {
    "other",
    "others",
    "ingredient",
    "ingredients",
    "etc",
    "etcetera",
    "misc",
    "miscellaneous",
    "unknown",
    "none",
    "null",
    "nil",
    "na",
    "nan",
    "n",
    "food",
    "foods",
    "item",
    "items",
    "stuff",
    "anything",
    "something",
}

NONFOOD_RAW_RE = re.compile(
    r"^(?:n/?a|nan|none|null|nil|unknown|not\s+available|no\s+ingredients?|ingredients?)$",
    re.IGNORECASE,
)


def _parse_true_ingredients(value: str) -> List[str]:
    value = (value or "").strip()
    if not value:
        return []
    return [p.strip() for p in value.split("|") if p.strip()]


def _parse_pred_ingredients_csv(value: str) -> List[str]:
    value = (value or "").strip()
    if not value:
        return []
    if "|" in value:
        return [p.strip() for p in value.split("|") if p.strip()]
    return [p.strip() for p in value.split(",") if p.strip()]


def _load_ai_ingredients_from_cache(dish_id: str, cache_dir: Path) -> Optional[List[str]]:
    cache_file = cache_dir / f"{dish_id}.json"
    if not cache_file.exists():
        return None
    try:
        data = json.loads(cache_file.read_text())
    except Exception:
        return None
    analysis = data.get("analysis", data) if isinstance(data, dict) else data
    if not isinstance(analysis, dict):
        return None
    items = analysis.get("items") or []
    out: List[str] = []
    for item in items:
        if isinstance(item, dict) and item.get("name"):
            out.append(str(item["name"]))
    return ev._filter_predicted_ingredients(out)  # type: ignore[attr-defined]


def _normalize_token_ext(token: str) -> str:
    return ev._normalize_token(token)  # type: ignore[attr-defined]


def _phrase_to_tokens_ext(phrase: str) -> List[str]:
    if NONFOOD_RAW_RE.match((phrase or "").strip()):
        return []
    tokens = re.findall(r"[a-zA-Z]+", phrase.lower())
    canonical_tokens: List[str] = []
    stopwords = ev._STOPWORDS | EXTRA_STOPWORDS  # type: ignore[attr-defined]
    for t in tokens:
        t = _normalize_token_ext(t)
        if not t or t in stopwords:
            continue
        t = ev._SYNONYM_TOKEN.get(t, t)  # type: ignore[attr-defined]
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


def _canonical_phrase_key_ext(phrase: str) -> str:
    toks = _phrase_to_tokens_ext(phrase)
    return " ".join(sorted(toks))


def _canonical_dedup_phrases(
    phrases: Sequence[str],
    drop_counter: Optional[Counter] = None,
) -> Tuple[List[Tuple[str, set]], int]:
    out: List[Tuple[str, set]] = []
    seen = set()
    dropped = 0
    for x in phrases:
        k = _canonical_phrase_key_ext(x)
        if not k:
            dropped += 1
            if drop_counter is not None:
                drop_counter[str(x)] += 1
            continue
        if k in seen:
            continue
        seen.add(k)
        out.append((x, set(_phrase_to_tokens_ext(x))))
    return out, dropped


def compute_jaccard_ext(
    true_items: Sequence[str],
    predicted_items: Sequence[str],
    drop_true: Optional[Counter] = None,
    drop_pred: Optional[Counter] = None,
) -> Tuple[float, int, int, int, int]:
    true_list, dropped_true = _canonical_dedup_phrases(true_items, drop_true)
    pred_list, dropped_pred = _canonical_dedup_phrases(predicted_items, drop_pred)

    n_true = len(true_list)
    n_pred = len(pred_list)
    if n_true == 0 and n_pred == 0:
        return math.nan, n_true, n_pred, dropped_true, dropped_pred

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
        return math.nan, n_true, n_pred, dropped_true, dropped_pred
    return match_count / union, n_true, n_pred, dropped_true, dropped_pred


def _mean(values: List[float]) -> float:
    vals = [v for v in values if isinstance(v, (int, float)) and not math.isnan(v)]
    return float(sum(vals) / len(vals)) if vals else math.nan


def _median(values: List[float]) -> float:
    vals = sorted(v for v in values if isinstance(v, (int, float)) and not math.isnan(v))
    if not vals:
        return math.nan
    mid = len(vals) // 2
    if len(vals) % 2:
        return float(vals[mid])
    return float((vals[mid - 1] + vals[mid]) / 2)


def main() -> int:
    parser = argparse.ArgumentParser(description="Sensitivity test for non-food Jaccard filtering.")
    parser.add_argument("--csv", required=True, type=Path, help="Evaluation CSV to read.")
    parser.add_argument("--cache-dir", type=Path, help="Cache dir with dish_id.json files.")
    parser.add_argument("--pred-ingredients-column", help="CSV column with predicted ingredients.")
    parser.add_argument("--output-csv", type=Path, help="Optional CSV to write per-row deltas.")
    parser.add_argument("--limit", type=int, help="Optional row limit for quick testing.")
    args = parser.parse_args()

    if not args.cache_dir and not args.pred_ingredients_column:
        raise SystemExit("Provide --cache-dir or --pred-ingredients-column.")

    drop_true = Counter()
    drop_pred = Counter()
    baseline_vals: List[float] = []
    new_vals: List[float] = []
    deltas: List[float] = []
    changed = 0
    processed = 0
    missing_cache = 0
    out_rows: List[Dict[str, str]] = []

    with args.csv.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row_idx, row in enumerate(reader):
            if args.limit and row_idx >= args.limit:
                break
            dish_id = (row.get("dish_id") or "").strip()
            if not dish_id:
                continue
            processed += 1
            true_items = _parse_true_ingredients(row.get("true_ingredients", ""))

            pred_items: Optional[List[str]] = None
            if args.pred_ingredients_column:
                pred_items = _parse_pred_ingredients_csv(row.get(args.pred_ingredients_column, ""))
            if pred_items is None and args.cache_dir:
                pred_items = _load_ai_ingredients_from_cache(dish_id, args.cache_dir)
            if pred_items is None:
                missing_cache += 1
                continue

            pred_items = ev._filter_predicted_ingredients(pred_items)  # type: ignore[attr-defined]

            base_j = ev.compute_jaccard(true_items, pred_items)
            new_j, n_true, n_pred, drop_t, drop_p = compute_jaccard_ext(
                true_items,
                pred_items,
                drop_true=drop_true,
                drop_pred=drop_pred,
            )

            if isinstance(base_j, (int, float)) and not math.isnan(base_j):
                baseline_vals.append(float(base_j))
            if isinstance(new_j, (int, float)) and not math.isnan(new_j):
                new_vals.append(float(new_j))

            if isinstance(base_j, (int, float)) and isinstance(new_j, (int, float)):
                if not (math.isnan(base_j) or math.isnan(new_j)):
                    delta = float(new_j - base_j)
                    deltas.append(delta)
                    if abs(delta) > 1e-9:
                        changed += 1
                else:
                    delta = math.nan
            else:
                delta = math.nan

            if args.output_csv:
                out_rows.append(
                    {
                        "dish_id": dish_id,
                        "jaccard_base": f"{base_j:.6f}" if isinstance(base_j, (int, float)) and not math.isnan(base_j) else "",
                        "jaccard_new": f"{new_j:.6f}" if isinstance(new_j, (int, float)) and not math.isnan(new_j) else "",
                        "delta": f"{delta:.6f}" if isinstance(delta, (int, float)) and not math.isnan(delta) else "",
                        "n_true": str(n_true),
                        "n_pred": str(n_pred),
                        "dropped_true_phrases": str(drop_t),
                        "dropped_pred_phrases": str(drop_p),
                    }
                )

    if args.output_csv and out_rows:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=[
                    "dish_id",
                    "jaccard_base",
                    "jaccard_new",
                    "delta",
                    "n_true",
                    "n_pred",
                    "dropped_true_phrases",
                    "dropped_pred_phrases",
                ],
            )
            writer.writeheader()
            writer.writerows(out_rows)

    valid = len(deltas)
    base_mean = _mean(baseline_vals)
    new_mean = _mean(new_vals)
    mean_delta = _mean(deltas)
    median_delta = _median(deltas)
    max_delta = max(deltas) if deltas else math.nan
    min_delta = min(deltas) if deltas else math.nan

    print("Jaccard non-food sensitivity test")
    print(f"CSV             : {args.csv}")
    if args.cache_dir:
        print(f"Cache dir       : {args.cache_dir}")
    if args.pred_ingredients_column:
        print(f"Pred column     : {args.pred_ingredients_column}")
    print(f"Rows processed  : {processed}")
    print(f"Missing cache   : {missing_cache}")
    print(f"Valid rows      : {valid}")
    print(f"Baseline mean   : {base_mean:.6f}" if not math.isnan(base_mean) else "Baseline mean   : n/a")
    print(f"New mean        : {new_mean:.6f}" if not math.isnan(new_mean) else "New mean        : n/a")
    if not math.isnan(mean_delta):
        print(f"Mean delta      : {mean_delta:+.6f}")
        print(f"Median delta    : {median_delta:+.6f}")
        print(f"Min/Max delta   : {min_delta:+.6f} / {max_delta:+.6f}")
    print(f"Rows changed    : {changed}/{valid}")

    if drop_pred:
        print("Top dropped predicted phrases:")
        for phrase, count in drop_pred.most_common(10):
            print(f"  {phrase} ({count})")
    if drop_true:
        print("Top dropped true phrases:")
        for phrase, count in drop_true.most_common(10):
            print(f"  {phrase} ({count})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
