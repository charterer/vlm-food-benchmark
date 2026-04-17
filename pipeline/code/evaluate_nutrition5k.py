#!/usr/bin/env python3
"""Run a VLM benchmark on Nutrition5k overhead meal photos.

Writes a CSV of per-dish predictions (calories, weight, ingredients) alongside
the Nutrition5k ground-truth. The provider is chosen with --provider; API keys
are read from the environment. See README for usage.
"""

import argparse
import contextlib
import csv
import io
import json
import math
import os
import re
import string
import sys
import textwrap
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Optional dotenv-style keys file: if present, loads into os.environ.
env_file = SCRIPT_DIR / "api_keys.env"
if env_file.exists():
    with env_file.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                key, value = line.split("=", 1)
                os.environ[key] = value.strip('"').strip("'")
            except ValueError:
                pass

from analyzers.factory import get_analyzer
from analyzers.base import FoodAnalyzer

MAX_PHOTO_DIMENSION = 1200


def _get_gemini_api_key_fallback() -> Optional[str]:
    return os.environ.get("GEMINI_API_KEY")

INVISIBLE_TOKENS = {
    "vinegar",
    "oil",
    "salt",
    "pepper",
    "sauce",
    "dressing",
    "seasoning",
    "spice",
    "herb",
    "parsley",
}

VISIBLE_GRAM_THRESHOLD = 10.0
VISIBLE_KCAL_SHARE = 0.05


def _standardize_image_bytes(image_bytes: bytes, max_dim: int) -> bytes:
    """
    Ensure all providers see the *same* downsampled JPEG bytes.

    - Nutrition5k overhead images are often PNGs; some providers assume JPEG.
    - We downsample to max_dim (e.g., 1200px) and encode as JPEG for consistency.
    """
    try:
        from PIL import Image as PILImage
        buf = io.BytesIO(image_bytes)
        img = PILImage.open(buf).convert("RGB")
        img.thumbnail((max_dim, max_dim))
        out = io.BytesIO()
        img.save(out, format="JPEG", quality=90, optimize=True)
        return out.getvalue()
    except Exception:
        return image_bytes


def load_dish_metadata(metadata_dir: Path) -> Dict[str, Dict[str, object]]:
    """Parse Nutrition5k per-dish metadata files into a dictionary."""

    dish_map: Dict[str, Dict[str, object]] = {}
    filenames = [
        metadata_dir / "dish_metadata_cafe1.csv",
        metadata_dir / "dish_metadata_cafe2.csv",
    ]

    for path in filenames:
        if not path.exists():
            continue
        with path.open("r", newline="") as csv_file:
            reader = csv.reader(csv_file)
            for row in reader:
                if not row:
                    continue
                dish_id = row[0].strip()
                if not dish_id:
                    continue
                try:
                    total_calories = float(row[1]) if row[1] else math.nan
                    total_weight = float(row[2]) if row[2] else math.nan
                except (IndexError, ValueError):
                    continue

                ingredient_names: List[str] = []
                ingredient_entries: List[Dict[str, float | str]] = []
                # Ingredient chunks start at index 6 and repeat every 7 columns:
                # [ingredient_id, name, grams, kcal, fat, carbs, protein]
                for idx in range(6, len(row), 7):
                    chunk = row[idx : idx + 7]
                    if len(chunk) < 7:
                        continue
                    name = chunk[1].strip()
                    try:
                        grams = float(chunk[2]) if chunk[2] else 0.0
                    except (ValueError, IndexError):
                        grams = 0.0
                    try:
                        kcal = float(chunk[3]) if len(chunk) > 3 and chunk[3] else 0.0
                    except (ValueError, IndexError):
                        kcal = 0.0
                    if name:
                        ingredient_names.append(name)
                        ingredient_entries.append({
                            "name": name,
                            "grams": grams,
                            "kcal": kcal,
                        })

                dish_map[dish_id] = {
                    "true_calories": total_calories,
                    "true_weight": total_weight,
                    "true_ingredients": ingredient_names,
                    "ingredients": ingredient_entries,
                }

    return dish_map


def _normalize_token(token: str) -> str:
    token = token.lower()
    if token == "greens":
        return "greens"
    # Basic plural handling with minimal stem damage
    if token.endswith("ies") and len(token) > 3:
        token = token[:-3] + "y"  # strawberries -> strawberry
    elif token.endswith("oes") and len(token) > 3:
        token = token[:-2]  # tomatoes -> tomato
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


_STOPWORDS = {
    "and", "with", "of", "a", "an", "the", "on", "in", "to", "for",
    "some", "type", "kind", "few", "several", "variety", "assortment",
    "piece", "pieces", "slice", "slices", "stick", "sticks", "semi", "circles",
    "light", "small", "big", "little", "lot", "seems", "appears",
    "salad", "dressing", "vinaigrette", "sauce", "coating",
    "cooked", "raw", "fried", "grilled", "roasted", "baked", "mashed", "sauteed",
    "dark", "pale", "fresh",
    # Adjectives / descriptors to ignore in token overlap (colors, textures, cuts)
    "white", "black", "brown", "red", "green", "yellow", "orange", "purple", "blue", "gold", "golden",
    "creamy", "buttery", "zesty", "savory", "sweet", "tangy", "spicy", "mild",
    "wild", "mixed", "assorted", "varied",
    "sliced", "diced", "chopped", "cubed", "minced", "shredded", "grated", "julienned",
    "whole", "halved", "quartered", "peeled", "roasted", "toasted",
    "ripe", "fresh", "dry", "dried", "seasoned", "plain",
    # User-specified exclusions (generic descriptors, preparation methods, parts)
    "dish", "asparagu", "strip", "herb", "breast", "or", "stir", "breaded", "steamed", "scrambled",
    "couscou", "cube", "leafy", "chunk", "floret", "meat", "cob", "mustard", "seed", "country",
    "ground", "steak", "greek", "sweetpotato", "berry", "thigh", "slice", "roasted",
    "vegetables", "salad", "wild", "wheat", "leaves", "sauce", "dressing", "raw", "mixed", "greens",
    # Hedging / qualifier words (verbose model outputs)
    "likely", "similar", "possibly", "perhaps", "approximately",
    "style", "component", "based", "visible", "resembling",
    "presumably", "probably", "looking", "think", "pitted", "liquid",
    "broth", "clear", "cooking", "be",
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


def _filter_visible_true_ingredients(info: Dict[str, object]) -> List[str]:
    total_calories = info.get("true_calories")
    ingredients = info.get("ingredients", []) or []
    visible: List[str] = []
    fallback_candidates: List[str] = []

    def _is_meaningful(tokens: Iterable[str]) -> bool:
        token_list = list(tokens)
        if not token_list:
            return False
        if "oil" in token_list or "vinaigrette" in token_list:
            return False
        return any(token not in INVISIBLE_TOKENS for token in token_list)

    if isinstance(total_calories, (int, float)) and not math.isnan(total_calories) and total_calories > 0:
        total_cal = float(total_calories)
    else:
        total_cal = None

    for entry in ingredients:
        name = str(entry.get("name", "")).strip()
        if not name:
            continue
        grams = float(entry.get("grams", 0.0) or 0.0)
        kcal = float(entry.get("kcal", 0.0) or 0.0)
        share = (kcal / total_cal) if (total_cal and total_cal > 0) else 0.0
        tokens = _phrase_to_tokens(name)
        if not _is_meaningful(tokens):
            continue
        if grams < VISIBLE_GRAM_THRESHOLD or share < VISIBLE_KCAL_SHARE:
            fallback_candidates.append(name)
            continue
        visible.append(name)

    if not visible:
        if fallback_candidates:
            visible = fallback_candidates
        else:
            fallback: List[str] = []
            for raw in info.get("true_ingredients", []) or []:
                raw_name = str(raw).strip()
                if not raw_name:
                    continue
                tokens = _phrase_to_tokens(raw_name)
                if _is_meaningful(tokens):
                    fallback.append(raw_name)
            visible = fallback
    return visible


def _filter_predicted_ingredients(names: List[str]) -> List[str]:
    filtered: List[str] = []
    for name in names:
        tokens = _phrase_to_tokens(name)
        if not tokens:
            continue
        if all(token in INVISIBLE_TOKENS for token in tokens):
            continue
        if "vinegar" in tokens:
            continue
        if "oil" in tokens and ("olive" in tokens or len(tokens) == 1):
            continue
        filtered.append(name)
    return filtered


def _phrase_to_tokens(phrase: str) -> List[str]:
    # Keep letters only, split to tokens
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

    # If the phrase expresses "sweet potato" as two tokens, add a composite synonym
    if ("sweet" in canonical_tokens and "potato" in canonical_tokens) and "sweetpotato" not in canonical_tokens:
        canonical_tokens.append("sweetpotato")
    # If "grape" and "tomato" co-occur, add composite to help matching
    if ("grape" in canonical_tokens and "tomato" in canonical_tokens) and "grapetomato" not in canonical_tokens:
        canonical_tokens.append("grapetomato")

    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for t in canonical_tokens:
        if t not in seen:
            deduped.append(t)
            seen.add(t)
    return deduped


def _canonical_phrase_key(phrase: str) -> str:
    # A canonical representation used for deduplication
    toks = _phrase_to_tokens(phrase)
    return " ".join(sorted(toks))


def compute_jaccard(true_items: Iterable[str], predicted_items: Iterable[str]) -> float:
    """Fuzzy Jaccard matching at ingredient-phrase level.

    A predicted phrase matches a true phrase if they share at least one
    canonical token (after lowercasing, simple lemmatisation, stopword removal,
    and synonym normalisation). The intersection is the size of a maximum
    one-to-one matching under this rule. The union is |true| + |pred| - |match|.
    """

    # Deduplicate by canonical tokens
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
        return math.nan

    # Build greedy maximum matching based on largest token overlap
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
        return math.nan
    return match_count / union


def run_analysis_with_provider(
    analyzer: FoodAnalyzer,
    image_path: Path,
    dish_id: str,
    cache_dir: Optional[Path],
    force_refresh: bool = False,
    retry_cached_errors: bool = False,
) -> Optional[Dict[str, object]]:
    """Run analysis using the configured provider adapter.

    Uses the same caching mechanism as the original function.
    """

    cache_file: Optional[Path] = None
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        # We might want to include provider in cache filename to avoid conflicts if switching providers
        # but for now we keep ID-based.
        # Ideally: f"{dish_id}_{analyzer.__class__.__name__}.json"
        cache_file = cache_dir / f"{dish_id}.json"
        if cache_file.exists() and not force_refresh:
            try:
                with cache_file.open("r") as fh:
                    payload = json.load(fh)
                if (
                    retry_cached_errors
                    and isinstance(payload, dict)
                    and isinstance(payload.get("general_description"), str)
                    and payload.get("general_description", "").startswith("Error:")
                ):
                    # Treat cached error payloads as a cache miss so we can resume after quotas/IP issues.
                    payload = None
                if payload is not None:
                    return payload
            except json.JSONDecodeError:
                pass  # Fall back to re-running analysis

    try:
        with image_path.open("rb") as image_file:
            raw_bytes = image_file.read()
    except OSError as exc:
        print(f"[WARN] Could not read image for {dish_id}: {exc}")
        return None

    # Standardize image bytes so all providers see the same resized JPEG.
    # This also fixes providers that label the data URI as image/jpeg.
    image_bytes = _standardize_image_bytes(raw_bytes, MAX_PHOTO_DIMENSION)

    # Execute analysis via the adapter
    with contextlib.redirect_stdout(io.StringIO()):
        # Some adapters might print to stdout, silence it for clean progress bars
        analysis_record = analyzer.analyze(image_bytes)

    if not analysis_record or "analysis" not in analysis_record:
        print(f"[WARN] AI analysis failed for {dish_id}.")
        return None

    analysis_payload = analysis_record.get("analysis")
    if cache_file:
        try:
            with cache_file.open("w") as fh:
                json.dump(analysis_payload, fh)
        except TypeError:
            pass

    return analysis_payload


def evaluate_one_dish(
    dish_id: str,
    imagery_root: Path,
    info: Dict[str, object],
    cache_dir: Optional[Path],
    force_refresh: bool,
    analyzer: FoodAnalyzer,  # Passed in now
) -> Optional[Tuple[Dict[str, object], Dict[str, object], Optional[Tuple[float, float, float]], Optional[Tuple[float, float, float]]]]:
    """Evaluate a single dish and return CSV row, detail, cal/weight pairs.

    Returns None if the dish is skipped or evaluation failed.
    """
    image_path = imagery_root / dish_id / "rgb.png"
    if not image_path.exists():
        return None

    true_ingredients = info.get("true_ingredients", [])
    lower_true = [str(x).lower() for x in true_ingredients]
    if any("deprecated" in ph for ph in lower_true):
        return None
    if any("plate only" in ph or ph.strip() == "plate" for ph in lower_true):
        return None

    ai_payload = run_analysis_with_provider(
        analyzer=analyzer,
        image_path=image_path,
        dish_id=dish_id,
        cache_dir=cache_dir,
        force_refresh=force_refresh,
        retry_cached_errors=getattr(analyzer, "_retry_cached_errors", False),
    )
    if not ai_payload:  # Payload is the "analysis" dict itself
        return None

    # The payload is already { "items": [...], "general_description": ... }
    # Note: Adapter returns { "analysis": { ... } }, run_analysis_with_provider extracts inner dict.
    
    analysis_content = ai_payload # It is the inner dictionary already
    ai_items = analysis_content.get("items") or []
    
    predicted_raw = [item.get("name") for item in ai_items if isinstance(item, dict) and item.get("name")]
    pred_map = {}
    for name in predicted_raw:
        key = _canonical_phrase_key(name)
        if key and key not in pred_map:
            pred_map[key] = name
    predicted_ingredients = _filter_predicted_ingredients(list(pred_map.values()))

    ai_description = (
        analysis_content.get("general_description")
        or analysis_content.get("overall_description")
        or ""
    )

    # Calories: prefer top-level `estimated_calories` if present, but fall back to summing item calories.
    # Some providers (e.g., certain local/open-source adapters) may omit or incorrectly set the top-level
    # total even when per-item calorie estimates are available.
    ai_calories = analysis_content.get("estimated_calories")
    calorie_candidates = [
        item.get("estimated_calories")
        for item in ai_items
        if isinstance(item, dict)
    ]
    calorie_values = [val for val in calorie_candidates if isinstance(val, (int, float))]
    item_calories_sum = float(sum(calorie_values)) if calorie_values else None
    if not isinstance(ai_calories, (int, float)):
        ai_calories = item_calories_sum
    elif ai_calories == 0 and isinstance(item_calories_sum, float) and item_calories_sum > 0:
        ai_calories = item_calories_sum

    weight_candidates = [
        item.get("estimated_weight_grams")
        for item in ai_items
        if isinstance(item, dict)
    ]
    weight_values = [val for val in weight_candidates if isinstance(val, (int, float))]
    ai_weight = float(sum(weight_values)) if weight_values else None

    true_calories = info.get("true_calories")
    true_weight = info.get("true_weight")
    visible_true_ingredients = _filter_visible_true_ingredients(info)

    # If the provider returned an explicit error payload, do NOT treat zeros/empties as valid predictions.
    # Keep the row for bookkeeping, but leave numeric metrics blank so downstream summaries exclude them.
    has_error_payload = isinstance(ai_description, str) and ai_description.strip().startswith("Error:")
    if has_error_payload:
        predicted_ingredients = []
        ai_calories = None
        ai_weight = None
        ingredient_jaccard = math.nan
    else:
        ingredient_jaccard = compute_jaccard(visible_true_ingredients, predicted_ingredients)

    row = {
        "dish_id": dish_id,
        "true_ingredients": " | ".join(visible_true_ingredients),
        "true_calories": f"{true_calories:.2f}" if isinstance(true_calories, float) else "",
        "true_weight": f"{true_weight:.2f}" if isinstance(true_weight, float) else "",
        "ai_description": ai_description,
        "ai_calories": f"{float(ai_calories):.2f}" if isinstance(ai_calories, (int, float)) else "",
        "ai_weight": f"{ai_weight:.2f}" if isinstance(ai_weight, float) else "",
        "ingredient_jaccard": f"{ingredient_jaccard:.3f}" if not math.isnan(ingredient_jaccard) else "",
    }

    detail = {
        "dish_id": dish_id,
        "image_path": image_path,
        "true_ingredients": visible_true_ingredients,
        "true_calories": true_calories,
        "true_weight": true_weight,
        "ai_ingredients": predicted_ingredients,
        "ai_calories": float(ai_calories) if isinstance(ai_calories, (int, float)) else math.nan,
        "ai_weight": ai_weight if isinstance(ai_weight, float) else math.nan,
        "ingredient_jaccard": ingredient_jaccard,
    }

    cal_pair = None
    wt_pair = None
    if isinstance(true_calories, float) and isinstance(ai_calories, (int, float)):
        cal_pair = (true_calories, float(ai_calories), ingredient_jaccard)
    if isinstance(true_weight, float) and isinstance(ai_weight, float):
        wt_pair = (true_weight, ai_weight, ingredient_jaccard)

    return row, detail, cal_pair, wt_pair


def summarise_errors(pairs: Iterable[Tuple[float, float]]) -> Dict[str, float]:
    """Return MAE and RMSE for a list of (true, predicted) value pairs."""

    diffs = [pred - true for true, pred in pairs]
    if not diffs:
        return {"mae": math.nan, "rmse": math.nan}

    mae = sum(abs(delta) for delta in diffs) / len(diffs)
    rmse = math.sqrt(sum(delta ** 2 for delta in diffs) / len(diffs))
    return {"mae": mae, "rmse": rmse}


def summarise_percentage_errors(pairs: Iterable[Tuple[float, float]]) -> Dict[str, float]:
    """Return percentage-based error summaries.

    - MAPE (%): mean(|pred-true| / true) * 100 over true > 0
    - sMAPE (%): mean(200*|pred-true| / (|true|+|pred|)) * 100-safe when denominators are zero
    - MdAPE (%): median absolute percentage error over true > 0
    """
    apes: List[float] = []
    smapes: List[float] = []
    for true, pred in pairs:
        # sMAPE denominator; if both 0, define 0
        denom = abs(true) + abs(pred)
        if denom > 0:
            smapes.append(200.0 * abs(pred - true) / denom)
        else:
            smapes.append(0.0)
        # MAPE/APEs only for true > 0 to avoid division by zero
        if true > 0:
            apes.append(100.0 * abs(pred - true) / true)

    def mean(vals: List[float]) -> float:
        return (sum(vals) / len(vals)) if vals else float('nan')

    def median(vals: List[float]) -> float:
        if not vals:
            return float('nan')
        vs = sorted(vals)
        n = len(vs)
        mid = n // 2
        if n % 2 == 1:
            return vs[mid]
        return 0.5 * (vs[mid - 1] + vs[mid])

    return {
        "mape": mean(apes),
        "mdape": median(apes),
        "smape": mean(smapes),
    }


def compute_logfit_metrics(pairs: Iterable[Tuple[float, float]]) -> Optional[Dict[str, float]]:
    filtered = [(float(x), float(y)) for x, y in pairs if isinstance(x, (int, float)) and isinstance(y, (int, float)) and x > 0 and y > 0]
    if len(filtered) < 2:
        return None
    xs = np.array([p[0] for p in filtered], dtype=float)
    ys = np.array([p[1] for p in filtered], dtype=float)
    logx = np.log(xs)
    logy = np.log(ys)
    beta, alpha = np.polyfit(logx, logy, 1)
    logy_hat = alpha + beta * logx
    y_hat = np.exp(logy_hat)
    ape = np.abs(ys - y_hat) / ys * 100.0
    p50 = float(np.nanpercentile(ape, 50))
    p90 = float(np.nanpercentile(ape, 90))

    mean_x = float(np.mean(xs))
    mean_y = float(np.mean(ys))
    var_x = float(np.var(xs))
    var_y = float(np.var(ys))
    cov_xy = float(np.cov(xs, ys, bias=True)[0, 1])
    denom = var_x + var_y + (mean_x - mean_y) ** 2
    ccc = float((2 * cov_xy) / denom) if denom > 0 else float("nan")

    return {
        "alpha": float(alpha),
        "beta": float(beta),
        "p50": p50,
        "p90": p90,
        "ccc": ccc,
        "xs": xs,
        "ys": ys,
    }


def summarise_mismatch_tokens(
    dish_details: List[Dict[str, object]],
    jaccard_threshold: float = 0.5,
) -> Dict[str, Dict[str, int]]:
    from collections import Counter

    fn_ctr: Counter[str] = Counter()
    fp_ctr: Counter[str] = Counter()
    true_totals: Counter[str] = Counter()
    pred_totals: Counter[str] = Counter()

    for d in dish_details:
        true_items = d.get("true_ingredients", []) or []
        ai_items = d.get("ai_ingredients", []) or []
        true_tokens = set()
        for ph in true_items:
            true_tokens.update(_phrase_to_tokens(str(ph)))
        pred_tokens = set()
        for ph in ai_items:
            pred_tokens.update(_phrase_to_tokens(str(ph)))

        true_totals.update(true_tokens)
        pred_totals.update(pred_tokens)

        j = d.get("ingredient_jaccard")
        if not isinstance(j, (int, float)) or math.isnan(j) or j >= jaccard_threshold:
            continue

        matched = true_tokens & pred_tokens
        fn_ctr.update(true_tokens - matched)
        fp_ctr.update(pred_tokens - matched)

    return {
        "fn_tokens": dict(fn_ctr),
        "fp_tokens": dict(fp_ctr),
        "true_totals": dict(true_totals),
        "pred_totals": dict(pred_totals),
    }


def build_top_plots(
    calorie_pairs: List[Tuple[float, float, float]],
    weight_pairs: List[Tuple[float, float, float]],
    ingredient_scores: List[float],
    output_path: Path,
    mismatch_summary: Optional[Dict[str, Dict[str, int]]] = None,
    summary_info: Optional[Dict[str, float]] = None,
    point_size: int = 2,
) -> None:
    """Create diagnostic plots for calories/weight errors and ingredient mismatches."""

    if plt is None:
        print("[WARN] matplotlib not installed; skipping plot generation.")
        return

    if not calorie_pairs and not weight_pairs and not ingredient_scores:
        print("[WARN] No data available for plotting.")
        return

    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(24, 22))
    gs = gridspec.GridSpec(
        3,
        3,
        width_ratios=[1.0, 1.0, 1.0],
        height_ratios=[1.5, 0.45, 5.5],
        hspace=0.6,
        wspace=0.32,
    )

    ax_cal = fig.add_subplot(gs[0, 0])
    ax_weight = fig.add_subplot(gs[0, 1])
    ax_jacc = fig.add_subplot(gs[0, 2])

    summary_spec = gs[1, :].subgridspec(1, 3, wspace=0.12)
    summary_axes = [fig.add_subplot(summary_spec[0, idx]) for idx in range(3)]
    for ax in summary_axes:
        ax.axis("off")

    # Confusion matrix section (replaces FN/FP bar charts)
    mismatch_spec = gs[2, :].subgridspec(1, 2, wspace=0.3)
    ax_confusion = fig.add_subplot(mismatch_spec[0, 0])
    ax_token_summary = fig.add_subplot(mismatch_spec[0, 1])

    # --- Calories scatter ---
    cal_metrics: Optional[Dict[str, float]] = None
    if calorie_pairs:
        cal_filtered = [
            (p[0], p[1], p[2])
            for p in calorie_pairs
            if isinstance(p[0], (int, float)) and isinstance(p[1], (int, float))
        ]
        if cal_filtered:
            cal_pairs = [(item[0], item[1]) for item in cal_filtered if item[0] > 0 and item[1] > 0]
            cal_true_all = np.array([item[0] for item in cal_filtered])
            cal_pred_all = np.array([item[1] for item in cal_filtered])
            cal_jaccard_all = np.array([item[2] for item in cal_filtered])
            cal_min = 0.0
            cal_max = 1500.0
            scatter = ax_cal.scatter(
                cal_true_all,
                cal_pred_all,
                c=np.nan_to_num(cal_jaccard_all, nan=0.0),
                cmap="viridis",
                alpha=0.7,
                edgecolor="none",
                s=point_size,
            )
            ax_cal.plot([cal_min, cal_max], [cal_min, cal_max], linestyle="--", color="gray", linewidth=1.0, label="Ideal")
            ax_cal.set_xlim(cal_min, cal_max)
            ax_cal.set_ylim(cal_min, cal_max)
            # Compute metrics for summary box (no log-fit line plotted)
            fit = compute_logfit_metrics(cal_pairs)
            if fit:
                cal_metrics = fit
                ax_cal.legend(loc="upper left", fontsize=9)
            fig.colorbar(scatter, ax=ax_cal, label="Ingredient Jaccard")
        else:
            ax_cal.text(0.5, 0.5, "No valid calorie pairs", ha="center", va="center")
    ax_cal.set_xlabel("True Calories (kcal)")
    ax_cal.set_ylabel("AI Estimated Calories (kcal)")
    ax_cal.set_title("Calories: AI vs Truth")

    # --- Weight scatter ---
    weight_metrics: Optional[Dict[str, float]] = None
    if weight_pairs:
        wt_filtered = [
            (p[0], p[1], p[2])
            for p in weight_pairs
            if isinstance(p[0], (int, float)) and isinstance(p[1], (int, float))
        ]
        if wt_filtered:
            wt_pairs = [(item[0], item[1]) for item in wt_filtered if item[0] > 0 and item[1] > 0]
            wt_true_all = np.array([item[0] for item in wt_filtered])
            wt_pred_all = np.array([item[1] for item in wt_filtered])
            wt_jaccard_all = np.array([item[2] for item in wt_filtered])
            wt_min = 0.0
            wt_max = 1000.0
            scatter_w = ax_weight.scatter(
                wt_true_all,
                wt_pred_all,
                c=np.nan_to_num(wt_jaccard_all, nan=0.0),
                cmap="viridis",
                alpha=0.7,
                edgecolor="none",
                s=point_size,
            )
            ax_weight.plot([wt_min, wt_max], [wt_min, wt_max], linestyle="--", color="gray", linewidth=1.0, label="Ideal")
            ax_weight.set_xlim(wt_min, wt_max)
            ax_weight.set_ylim(wt_min, wt_max)
            # Compute metrics for summary box (no log-fit line plotted)
            fit_w = compute_logfit_metrics(wt_pairs)
            if fit_w:
                weight_metrics = fit_w
                ax_weight.legend(loc="upper left", fontsize=9)
            fig.colorbar(scatter_w, ax=ax_weight, label="Ingredient Jaccard")
        else:
            ax_weight.text(0.5, 0.5, "No valid weight pairs", ha="center", va="center")
    ax_weight.set_xlabel("True Weight (g)")
    ax_weight.set_ylabel("AI Estimated Weight (g)")
    ax_weight.set_title("Weight: AI vs Truth")

    # --- Ingredient overlap histogram ---
    valid_scores = [score for score in ingredient_scores if not math.isnan(score)]
    if valid_scores:
        bins = np.linspace(0, 1, 21)
        ax_jacc.hist(valid_scores, bins=bins, color="#1f77b4", alpha=0.85, edgecolor="black")
        ax_jacc.set_xlim(0, 1)
        ax_jacc.set_ylim(0, max(ax_jacc.get_ylim()[1], 1))
        ax_jacc.text(
            0.03,
            0.97,
            f"Mean: {np.mean(valid_scores):.2f}\nMedian: {np.median(valid_scores):.2f}",
            transform=ax_jacc.transAxes,
            verticalalignment="top",
            bbox={"facecolor": "white", "alpha": 0.8, "pad": 6},
            fontsize=9,
        )
    ax_jacc.set_xlabel("Ingredient Jaccard (AI vs Truth)")
    ax_jacc.set_ylabel("Dish Count")
    ax_jacc.set_title("Ingredient Overlap Distribution")

    # --- Confusion Matrix ---
    if mismatch_summary:
        from collections import Counter

        fn_ctr = Counter(mismatch_summary.get("fn_tokens", {}))
        fp_ctr = Counter(mismatch_summary.get("fp_tokens", {}))
        true_totals = Counter(mismatch_summary.get("true_totals", {}))
        pred_totals = Counter(mismatch_summary.get("pred_totals", {}))

        # Calculate confusion matrix values
        total_fn = sum(fn_ctr.values())
        total_fp = sum(fp_ctr.values())
        total_true = sum(true_totals.values())
        total_pred = sum(pred_totals.values())
        total_tp = total_true - total_fn  # True positives = true tokens that were matched
        
        # Create 2x2 confusion matrix: [[TP, FP], [FN, TN]]
        # TN not meaningful for token matching, use 0
        cm = np.array([[total_tp, total_fp], [total_fn, 0]])
        
        # Plot confusion matrix heatmap
        im = ax_confusion.imshow(cm, cmap='Blues', aspect='auto')
        ax_confusion.set_xticks([0, 1])
        ax_confusion.set_yticks([0, 1])
        ax_confusion.set_xticklabels(['Predicted\nPositive', 'Predicted\nNegative'], fontsize=10)
        ax_confusion.set_yticklabels(['Actual\nPositive', 'Actual\nNegative'], fontsize=10)
        ax_confusion.set_title("Ingredient Token Confusion Matrix", fontsize=12)
        
        # Add text annotations
        for i in range(2):
            for j in range(2):
                val = cm[i, j]
                label = ['TP', 'FP', 'FN', 'N/A'][i * 2 + j]
                color = 'white' if val > cm.max() / 2 else 'black'
                ax_confusion.text(j, i, f'{label}\n{val:,}', ha='center', va='center', 
                                 fontsize=14, fontweight='bold', color=color)
        
        # Add colorbar
        fig.colorbar(im, ax=ax_confusion, shrink=0.6)
        
        # Token summary on right side
        precision = total_tp / total_pred if total_pred > 0 else 0
        recall = total_tp / total_true if total_true > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        summary_text = (
            f"Token-Level Metrics:\n\n"
            f"True Positives: {total_tp:,}\n"
            f"False Positives: {total_fp:,}\n"
            f"False Negatives: {total_fn:,}\n\n"
            f"Precision: {precision:.2%}\n"
            f"Recall: {recall:.2%}\n"
            f"F1 Score: {f1:.2%}\n\n"
            f"Top Missed (FN):\n"
        )
        # Add top 5 false negatives
        for token, count in fn_ctr.most_common(5):
            summary_text += f"  • {token}: {count}\n"
        summary_text += f"\nTop False (FP):\n"
        # Add top 5 false positives
        for token, count in fp_ctr.most_common(5):
            summary_text += f"  • {token}: {count}\n"
        
        ax_token_summary.text(0.05, 0.95, summary_text, transform=ax_token_summary.transAxes,
                             fontsize=10, verticalalignment='top', fontfamily='monospace',
                             bbox=dict(boxstyle='round', facecolor='#f7f7f7', edgecolor='#444444', pad=0.5))
        ax_token_summary.axis('off')
        ax_token_summary.set_title("Token Summary", fontsize=12)
    else:
        ax_confusion.text(0.5, 0.5, "No mismatch data", ha="center", va="center")
        ax_confusion.axis('off')
        ax_token_summary.text(0.5, 0.5, "No mismatch data", ha="center", va="center")
        ax_token_summary.axis('off')

    # --- Summary panel ---
    if summary_info and len(summary_axes) == 3:
        box_kwargs = {
            "bbox": {
                "facecolor": "#f7f7f7",
                "edgecolor": "#444444",
                "boxstyle": "round,pad=0.6",
            }
        }

        def _fmt_int(value: Optional[int]) -> str:
            if isinstance(value, (int, float)):
                if isinstance(value, float) and math.isnan(value):
                    return "N/A"
                return f"{int(round(value)):,}"
            return "N/A"

        initial_val = summary_info.get("initial")
        analysed_val = summary_info.get("dishes")
        summary_axes[0].text(
            0.0,
            1.0,
            (
                f"Initial images: {_fmt_int(initial_val)}\n"
                f"Analyzed images: {_fmt_int(analysed_val)}"
            ),
            ha="left",
            va="top",
            fontsize=11,
            wrap=True,
            **box_kwargs,
        )

        if cal_metrics:
            p50 = cal_metrics.get("p50")
            p90 = cal_metrics.get("p90")
            beta = cal_metrics.get("beta")
            ccc = cal_metrics.get("ccc")
            summary_axes[1].text(
                0.0,
                1.0,
                (
                    "Calories:\n"
                    f"P50 APE: {p50:.1f}%\n"
                    f"  → Half the dishes within ±{p50:.0f}% of true\n"
                    f"P90 APE: {p90:.1f}%\n"
                    f"  → 90% within ±{p90:.0f}% of true\n"
                    f"CCC: {ccc:.2f}\n"
                    "  → Concordance (1.0 = perfect)"
                ),
                ha="left",
                va="top",
                fontsize=11,
                wrap=True,
                **box_kwargs,
            )

        if weight_metrics:
            p50_w = weight_metrics.get("p50")
            p90_w = weight_metrics.get("p90")
            beta_w = weight_metrics.get("beta")
            ccc_w = weight_metrics.get("ccc")
            summary_axes[2].text(
                0.0,
                1.0,
                (
                    "Weight:\n"
                    f"P50 APE: {p50_w:.1f}%\n"
                    f"  → Half the dishes within ±{p50_w:.0f}% of true\n"
                    f"P90 APE: {p90_w:.1f}%\n"
                    f"  → 90% within ±{p90_w:.0f}% of true\n"
                    f"CCC: {ccc_w:.2f}\n"
                    "  → Concordance (1.0 = perfect)"
                ),
                ha="left",
                va="top",
                fontsize=11,
                wrap=True,
                **box_kwargs,
            )

    fig.suptitle("AI vs Nutrition5k Ground Truth", fontsize=18, y=0.97)
    fig.subplots_adjust(left=0.06, right=0.97, top=0.93, bottom=0.06, hspace=0.5, wspace=0.28)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def build_montage(
    page_details: List[Dict[str, object]],
    output_path: Path,
    n_cols: int = 5,
    n_rows: int = 6,
) -> None:
    if plt is None:
        print("[WARN] matplotlib not installed; skipping montage generation.")
        return

    if not page_details:
        print("[WARN] No dishes provided for montage page; skipping.")
        return

    total_slots = n_cols * n_rows
    selection = page_details[:total_slots]

    import matplotlib.gridspec as gridspec
    img_h = 4.0
    txt_h = 2.2
    fig_height = (img_h + txt_h) * n_rows
    fig = plt.figure(figsize=(18, max(fig_height * 0.6, 6.5)))
    height_ratios = []
    for _ in range(n_rows):
        height_ratios.extend([img_h, txt_h])
    gs = gridspec.GridSpec(
        n_rows * 2,
        n_cols,
        figure=fig,
        hspace=0.08,
        wspace=0.08,
        height_ratios=height_ratios,
    )

    def format_value(value: Optional[float]) -> str:
        if isinstance(value, (int, float)) and not math.isnan(value):
            return f"{float(value):.0f}"
        return "--"

    def format_jaccard(value: Optional[float]) -> str:
        if isinstance(value, (int, float)) and not math.isnan(value):
            return f"{value:.2f}"
        return "--"

    def wrap_block(text: str, width: int = 28, max_lines: int = 15) -> str:
        if not text:
            return "--"
        filled = textwrap.fill(text, width=width, break_long_words=False, break_on_hyphens=False)
        lines = filled.splitlines()
        if len(lines) > max_lines:
            lines = lines[:max_lines]
            if len(lines[-1]) > 1:
                lines[-1] = lines[-1].rstrip('.') + '…'
        return "\n".join(lines)

    # Import Pillow lazily here so montage generation doesn't depend on photo_analysis imports.
    try:
        from PIL import Image as PILImage  # type: ignore
    except Exception:
        PILImage = None

    for idx in range(total_slots):
        row = idx // n_cols
        col = idx % n_cols
        ax_img = fig.add_subplot(gs[row * 2, col])
        ax_img.axis("off")
        if idx < len(selection):
            detail = selection[idx]
            image_path = detail.get("image_path")
            try:
                if PILImage is None:
                    raise RuntimeError("Pillow not available for montage rendering")
                with PILImage.open(image_path) as img_obj:
                    ax_img.imshow(img_obj)
                    ax_img.set_aspect("auto")
                    ax_img.margins(0)
            except Exception as exc:  # noqa: BLE001
                ax_img.text(0.5, 0.5, f"Image load error\n{exc}", ha="center", va="center")

            true_ingredients = ", ".join(detail.get("true_ingredients", [])) or "none"
            ai_ingredients = ", ".join(detail.get("ai_ingredients", [])) or "none"
            true_cal = format_value(detail.get("true_calories"))
            true_weight = format_value(detail.get("true_weight"))
            ai_cal = format_value(detail.get("ai_calories"))
            ai_weight = format_value(detail.get("ai_weight"))
            jaccard_str = format_jaccard(detail.get("ingredient_jaccard"))

            ax_txt = fig.add_subplot(gs[row * 2 + 1, col])
            ax_txt.axis("off")
            ax_txt.set_xlim(0, 1)
            ax_txt.set_ylim(0, 1)
            block = (
                f"True C:{true_cal} | W:{true_weight}\n"
                f"{wrap_block(true_ingredients)}\n"
                f"AI   C:{ai_cal} | W:{ai_weight} | J:{jaccard_str}\n"
                f"{wrap_block(ai_ingredients)}"
            )
            ax_txt.text(
                0.0,
                1.0,
                block,
                ha="left",
                va="top",
                fontsize=9,
                color="#000000",
                transform=ax_txt.transAxes,
                wrap=True,
                clip_on=True,
            )
        else:
            ax_img.axis("off")

    fig.suptitle("Low-Overlap Examples", fontsize=16)
    fig.subplots_adjust(left=0.03, right=0.97, top=0.94, bottom=0.04, hspace=0.08, wspace=0.08)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate VLM predictions on Nutrition5k overhead meal images.")
    default_dataset_root = (SCRIPT_DIR.parent / "extracted" / "nutrition5k_dataset").resolve()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=default_dataset_root,
        help="Path to the extracted Nutrition5k dataset root (containing metadata/imagery).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=(SCRIPT_DIR / "nutrition5k_gemini_eval.csv"),
        help="Destination CSV path for comparison results.",
    )
    parser.add_argument(
        "--output-plot",
        type=Path,
        default=(SCRIPT_DIR / "nutrition5k_gemini_eval_plots.png"),
        help="Destination image path for the top plots (no montage).",
    )
    parser.add_argument(
        "--output-montage",
        type=Path,
        default=(SCRIPT_DIR / "nutrition5k_gemini_eval_montage.png"),
        help="Destination image path for the montage of low-Jaccard examples.",
    )
    parser.add_argument(
        "--montage-cols",
        type=int,
        default=5,
        help="Number of columns in the montage grid.",
    )
    parser.add_argument(
        "--montage-rows",
        type=int,
        default=6,
        help="Number of rows in the montage grid.",
    )
    parser.add_argument(
        "--montage-pages",
        type=int,
        default=2,
        help="Maximum number of montage pages to generate.",
    )
    parser.add_argument(
        "--montage-jaccard-max",
        type=float,
        default=0.5,
        help="Include only images with ingredient Jaccard below this threshold in the montage.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Optional random seed for selecting montage examples.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of dishes analysed (for quick smoke tests).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel worker threads to use for AI analysis.",
    )
    parser.add_argument(
        "--random-sample",
        type=int,
        default=None,
        help="Randomly select N dishes (after filtering) for evaluation.",
    )
    parser.add_argument(
        "--dish-ids-file",
        type=Path,
        default=None,
        help=(
            "Optional path to a text file containing one dish_id per line. "
            "If provided, evaluate exactly these dish IDs (after verifying metadata + rgb.png exist). "
            "Overrides --limit/--random-sample/--num-shards/--shard-index/--skip-first/--limit-eval."
        ),
    )
    parser.add_argument(
        "--dish-ids-allow-missing",
        action="store_true",
        help=(
            "If --dish-ids-file is provided, allow missing/invalid dish IDs (missing metadata or rgb.png) "
            "to be skipped instead of failing."
        ),
    )
    parser.add_argument(
        "--skip-first",
        type=int,
        default=0,
        help="Skip the first N dishes (for batched evaluation across days).",
    )
    parser.add_argument(
        "--limit-eval",
        type=int,
        default=None,
        help="Limit to evaluating only N dishes after skip-first (for rate-limited APIs).",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=None,
        help="Total number of shards for distributed runs (e.g., Slurm arrays).",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=None,
        help="Index of this shard in [0, num-shards-1].",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=(SCRIPT_DIR / ".gemini_cache"),
        help="Directory for caching Gemini analysis responses.",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Ignore cached Gemini responses and re-run analysis.",
    )
    parser.add_argument(
        "--retry-cached-errors",
        action="store_true",
        help=(
            "If a cached analysis payload has general_description starting with 'Error:', treat it as a cache miss "
            "and retry the API call. Useful for resuming runs after transient quota/IP issues."
        ),
    )
    
    # --- New Arguments for Multi-Provider Support ---
    parser.add_argument(
        "--provider",
        type=str,
        default="gemini",
        # Allow open-ended strings so we can support any provider defined in factory
        help="The AI provider to use for analysis."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override the model name (e.g., gemini-2.5-flash, gpt-4o)."
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Override the API key via CLI."
    )
    parser.add_argument(
        "--prompt-variant",
        type=str,
        default=None,
        choices=["P1", "P2", "P3", "P4", "P5", "P6", "p1", "p2", "p3", "p4", "p5", "p6"],
        help="Prompt variant for ablation study (P1=minimal, P2=portion-first, P3=anti-snap, P4=constraints, P5=self-check, P6=app prompt)."
    )
    # ------------------------------------------------

    args = parser.parse_args()

    dataset_root = args.dataset_root.expanduser().resolve()
    metadata_dir = dataset_root / "metadata"
    imagery_root = dataset_root / "imagery" / "realsense_overhead"

    if not metadata_dir.exists() or not imagery_root.exists():
        raise SystemExit(
            f"Dataset directories not found under '{dataset_root}'. Did you extract Nutrition5k?"
        )

    api_key = args.api_key
    if not api_key:
        p = args.provider.lower()
        if p.startswith("gemini"):
            key_prefix = "GEMINI"
        elif p in ("openai", "gpt4", "gpt-4o", "gpt-4o-mini", "gpt5", "gpt-5", "gpt-5-mini"):
            key_prefix = "OPENAI"
        elif p in ("claude", "anthropic", "sonnet", "haiku", "claude-haiku"):
            key_prefix = "ANTHROPIC"
        elif p in ("qwen", "qwen2-vl", "qwen2vl"):
            key_prefix = "HF"
        else:
            key_prefix = p.upper()

        api_key = os.environ.get(f"{key_prefix}_API_KEY") or os.environ.get(f"{key_prefix}_TOKEN")

    local_providers = {"qwen", "qwen2-vl", "qwen2vl"}
    if not api_key and args.provider.lower() not in local_providers:
        raise SystemExit(
            f"API key for provider '{args.provider}' is missing. "
            f"Set {key_prefix}_API_KEY in your environment or pass --api-key."
        )

    # Instantiate the Analyzer
    prompt_variant = args.prompt_variant.upper() if args.prompt_variant else None
    try:
        analyzer = get_analyzer(args.provider, api_key=api_key, model_name=args.model, prompt_variant=prompt_variant)
        print(f"Using provider: {analyzer.__class__.__name__}")
        if hasattr(analyzer, 'model_name') and analyzer.model_name:
            print(f"Model: {analyzer.model_name}")
        if prompt_variant:
            print(f"Prompt variant: {prompt_variant}")
        # Thread-safe read-only flag used by evaluate_one_dish/run_analysis_with_provider
        setattr(analyzer, "_retry_cached_errors", bool(args.retry_cached_errors))
    except ValueError as e:
        raise SystemExit(f"Configuration Error: {e}")

    dish_metadata = load_dish_metadata(metadata_dir)
    if not dish_metadata:
        raise SystemExit("No dish metadata found. Ensure Nutrition5k metadata CSVs are available.")

    # Resolve candidate dish IDs
    if args.dish_ids_file is not None:
        raw_ids = []
        for line in args.dish_ids_file.read_text().splitlines():
            did = line.strip()
            if did:
                raw_ids.append(did)

        # De-dupe while preserving order
        seen = set()
        requested = []
        for did in raw_ids:
            if did in seen:
                continue
            seen.add(did)
            requested.append(did)

        missing = []
        dish_ids = []
        for did in requested:
            rgb_path = imagery_root / did / "rgb.png"
            if did not in dish_metadata or not rgb_path.exists():
                missing.append(did)
                if args.dish_ids_allow_missing:
                    continue
                raise SystemExit(
                    f"--dish-ids-file contains invalid dish_id '{did}' "
                    f"(missing metadata or rgb.png at {rgb_path}). "
                    "Rebuild the dish list from a successful run, or pass --dish-ids-allow-missing."
                )
            dish_ids.append(did)

        if not dish_ids:
            raise SystemExit(f"--dish-ids-file produced 0 valid dish IDs. Missing={len(missing)}")

        print(f"Using dish ids from file: {args.dish_ids_file} (n={len(dish_ids)}, missing={len(missing)})")
        if any(v is not None for v in (args.limit, args.random_sample, args.num_shards, args.shard_index)) or args.skip_first or args.limit_eval:
            print("  > Note: --dish-ids-file overrides --limit/--random-sample/sharding/skip-first/limit-eval for apples-to-apples runs.")
    else:
        available_dish_dirs = [
            path
            for path in imagery_root.iterdir()
            if path.is_dir() and (path / "rgb.png").exists()
        ]
        if not available_dish_dirs:
            raise SystemExit(
                "No overhead meal photos found (rgb.png). Confirm Nutrition5k extraction."
            )

        available_dish_ids = sorted(path.name for path in available_dish_dirs if path.name in dish_metadata)
        # Optional sharding support for array jobs
        if args.num_shards is not None and args.shard_index is not None:
            if args.num_shards <= 0 or not (0 <= args.shard_index < args.num_shards):
                raise SystemExit("Invalid --num-shards/--shard-index combination.")
            sharded = []
            for idx, did in enumerate(available_dish_ids):
                if idx % args.num_shards == args.shard_index:
                    sharded.append(did)
            available_dish_ids = sharded
        if not available_dish_ids:
            raise SystemExit("No overlap between metadata dish IDs and available rgb.png images.")

        total_with_images = len(available_dish_ids)
        if args.limit is not None:
            if args.limit <= 0:
                raise SystemExit("--limit must be a positive integer if provided.")
            dish_ids = available_dish_ids[: args.limit]
            print(
                f"Using first {len(dish_ids)} dishes that have rgb.png (out of {total_with_images} available with images, {len(dish_metadata)} total in metadata)."
            )
        else:
            dish_ids = available_dish_ids
            print(
                f"Using all {len(dish_ids)} dishes that have rgb.png (out of {len(dish_metadata)} total in metadata)."
            )

    results: List[Dict[str, object]] = []
    calorie_pairs: List[Tuple[float, float, float]] = []
    weight_pairs: List[Tuple[float, float, float]] = []
    ingredient_scores: List[float] = []
    dish_details: List[Dict[str, object]] = []

    analysed = 0
    skipped = 0

    print(f"Analysing {len(dish_ids)} dishes (limit={args.limit}, workers={args.workers})...")

    # Pre-filter for deprecated/plate-only to avoid scheduling unnecessary work
    filtered_pairs = []
    for did in dish_ids:
        info = dish_metadata[did]
        lower_true = [str(x).lower() for x in info.get("true_ingredients", [])]
        if any("deprecated" in ph for ph in lower_true):
            skipped += 1
            continue
        if any("plate only" in ph or ph.strip() == "plate" for ph in lower_true):
            skipped += 1
            continue
        true_cal = info.get("true_calories")
        true_wt = info.get("true_weight")
        if not isinstance(true_cal, (int, float)) or math.isnan(true_cal) or true_cal <= 0:
            skipped += 1
            continue
        if not isinstance(true_wt, (int, float)) or math.isnan(true_wt) or true_wt <= 0:
            skipped += 1
            continue
        filtered_pairs.append((did, info))

    total_candidates = len(filtered_pairs)

    if args.dish_ids_file is not None:
        # When using an explicit dish list, we do not resample.
        if total_candidates != len(dish_ids) and not args.dish_ids_allow_missing:
            raise SystemExit(
                f"--dish-ids-file requested {len(dish_ids)} dishes but {total_candidates} remained after filtering. "
                "This breaks apples-to-apples evaluation. Rebuild the dish list from a successful run or pass "
                "--dish-ids-allow-missing."
            )
        sample_size = total_candidates
    else:
        if args.random_sample is not None and filtered_pairs:
            sample_size = min(args.random_sample, total_candidates)
            rng_sample = random.Random(args.random_seed if args.random_seed is not None else 12345)
            filtered_pairs = rng_sample.sample(filtered_pairs, k=sample_size)
            print(f"  > Randomly sampled {sample_size} dishes for evaluation (from {total_candidates}).")
        else:
            sample_size = total_candidates

    # Apply skip-first and limit-eval for batched evaluation (e.g., rate-limited APIs)
    if args.dish_ids_file is None and args.skip_first > 0:
        filtered_pairs = filtered_pairs[args.skip_first:]
        print(f"  > Skipped first {args.skip_first} dishes, {len(filtered_pairs)} remaining.")
    
    if args.dish_ids_file is None and args.limit_eval is not None and args.limit_eval > 0:
        filtered_pairs = filtered_pairs[:args.limit_eval]
        print(f"  > Limited to {len(filtered_pairs)} dishes for this batch.")

    # Pass the instantiated analyzer to the worker function
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
        future_map = {
            executor.submit(
                evaluate_one_dish, 
                did, 
                imagery_root, 
                info, 
                args.cache_dir, 
                args.force_refresh,
                analyzer
            ): did
            for (did, info) in filtered_pairs
        }
        total = len(future_map)
        completed = 0
        for future in as_completed(future_map):
            res = future.result()
            completed += 1
            if res is None:
                skipped += 1
            else:
                row, detail, cal_pair, wt_pair = res
                results.append(row)
                dish_details.append(detail)
                if cal_pair is not None:
                    calorie_pairs.append(cal_pair)
                    ingredient_scores.append(cal_pair[2])
                if wt_pair is not None:
                    weight_pairs.append(wt_pair)
            if completed % 20 == 0 or completed == total:
                print(f"  - Completed {completed}/{total} evaluations...")
        analysed = len(dish_details)

    # Write CSV with requested columns
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as out_csv:
        fieldnames = [
            "dish_id",
            "true_ingredients",
            "true_calories",
            "true_weight",
            "ai_description",
            "ai_calories",
            "ai_weight",
            "ingredient_jaccard",
        ]
        writer = csv.DictWriter(out_csv, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    mismatch_summary = summarise_mismatch_tokens(dish_details, jaccard_threshold=args.montage_jaccard_max)
    jaccard_values = [d.get("ingredient_jaccard") for d in dish_details if isinstance(d.get("ingredient_jaccard"), (int, float)) and not math.isnan(d.get("ingredient_jaccard"))]

    eligible_montage = [
        d
        for d in dish_details
        if isinstance(d.get("ingredient_jaccard"), (int, float))
        and not math.isnan(d.get("ingredient_jaccard"))
        and d.get("ingredient_jaccard") < args.montage_jaccard_max
    ]
    eligible_montage.sort(key=lambda d: (d.get("ingredient_jaccard", float("inf")), d.get("dish_id", "")))
    slots_per_page = args.montage_cols * args.montage_rows
    total_candidates_montage = len(eligible_montage)
    generated_montages: List[Path] = []
    if slots_per_page > 0 and eligible_montage and args.montage_pages > 0:
        total_pages_possible = math.ceil(total_candidates_montage / slots_per_page)
        pages_to_render = min(args.montage_pages, total_pages_possible)
    else:
        pages_to_render = 0

    summary_info = {
        "initial": len(dish_ids),
        "post_filter": total_candidates,
        "sampled": sample_size,
        "dishes": len(dish_details),
        "cal_pairs": len(calorie_pairs),
        "weight_pairs": len(weight_pairs),
        "mean_jaccard": float(np.nanmean(jaccard_values)) if jaccard_values else float("nan"),
        "montage_pages": pages_to_render,
        "montage_candidates": total_candidates_montage,
    }
    build_top_plots(
        calorie_pairs,
        weight_pairs,
        ingredient_scores,
        args.output_plot,
        mismatch_summary=mismatch_summary,
        summary_info=summary_info,
    )

    if pages_to_render > 0:
        base_path = args.output_montage
        for page_idx in range(pages_to_render):
            page_items = eligible_montage[page_idx * slots_per_page : (page_idx + 1) * slots_per_page]
            if not page_items:
                break
            page_path = base_path.with_name(f"{base_path.stem}_page{page_idx + 1}{base_path.suffix}")
            build_montage(
                page_items,
                page_path,
                n_cols=args.montage_cols,
                n_rows=args.montage_rows,
            )
            generated_montages.append(page_path)
    else:
        print("[WARN] No montage images generated (insufficient low-overlap dishes).")

    cal_stats = summarise_errors([(p[0], p[1]) for p in calorie_pairs]) if calorie_pairs else {"mae": math.nan, "rmse": math.nan}
    wt_stats = summarise_errors([(p[0], p[1]) for p in weight_pairs]) if weight_pairs else {"mae": math.nan, "rmse": math.nan}
    valid_jaccard = [score for score in ingredient_scores if not math.isnan(score)]

    print("\nEvaluation complete.")
    print(f"  Initial candidates : {len(dish_ids)}")
    print(f"  After filters      : {total_candidates}")
    print(f"  Sampled            : {sample_size}")
    print(f"  Dishes analysed    : {analysed}")
    print(f"  Dishes skipped     : {skipped}")
    print(f"  Montage candidates : {total_candidates_montage}")
    print(f"  Montage pages      : {pages_to_render}")
    print(f"  CSV written to  : {args.output_csv}")
    if plt is not None:
        print(f"  Plot saved to   : {args.output_plot}")
    print(f"  Calorie MAE/RMSE: {cal_stats['mae']:.2f} / {cal_stats['rmse']:.2f}")
    print(f"  Weight MAE/RMSE : {wt_stats['mae']:.2f} / {wt_stats['rmse']:.2f}")
    if valid_jaccard:
        print(
            f"  Ingredient Jaccard -- mean: {np.mean(valid_jaccard):.2f}, "
            f"median: {np.median(valid_jaccard):.2f}"
        )
    if generated_montages:
        montage_names = ", ".join(path.name for path in generated_montages)
        print(f"  Montage files    : {montage_names}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
