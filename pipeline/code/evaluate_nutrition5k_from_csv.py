#!/usr/bin/env python3
"""
Evaluate precomputed Nutrition5k predictions from an external model (e.g. LogMeal).

This script reuses the same metrics and plotting code as `evaluate_nutrition5k.py`,
but instead of calling Gemini it reads AI predictions from a CSV.

Expected input CSV schema (one row per dish_id):
    dish_id            : string, Nutrition5k dish ID (must match metadata/imagery IDs)
    ai_ingredients     : string, list of ingredient phrases separated by ' | '
                         e.g. "chicken breast | rice | broccoli"
    ai_calories        : float, total kcal for the dish (optional, can be blank)
    ai_weight          : float, total grams for the dish (optional, can be blank)
    ai_description     : free-text description (optional, used only for CSV output)

You can produce such a CSV from any external vision API (LogMeal, Yunabite, Dragoneye)
and then plug it into this script to get directly comparable metrics and plots.
"""

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from evaluate_nutrition5k import (  # type: ignore
    load_dish_metadata,
    _filter_visible_true_ingredients,
    _filter_predicted_ingredients,
    compute_jaccard,
    summarise_errors,
    summarise_percentage_errors,
    compute_logfit_metrics,
    summarise_mismatch_tokens,
    build_top_plots,
    build_montage,
)


def _to_float(value: str) -> Optional[float]:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate external model predictions on Nutrition5k.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Path to the extracted Nutrition5k dataset root (containing metadata/imagery).",
    )
    parser.add_argument(
        "--predictions-csv",
        type=Path,
        required=True,
        help="CSV with external model predictions (dish_id, ai_ingredients, ai_calories, ai_weight, ai_description).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="external_model",
        help="Name of the model (for logging/plot titles).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        required=True,
        help="Destination CSV path for comparison results.",
    )
    parser.add_argument(
        "--output-plot",
        type=Path,
        required=True,
        help="Destination image path for the top plots (no montage).",
    )
    parser.add_argument(
        "--output-montage",
        type=Path,
        required=False,
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
    args = parser.parse_args()

    dataset_root = args.dataset_root.expanduser().resolve()
    metadata_dir = dataset_root / "metadata"
    imagery_root = dataset_root / "imagery" / "realsense_overhead"

    if not metadata_dir.exists() or not imagery_root.exists():
        raise SystemExit(
            f"Dataset directories not found under '{dataset_root}'. Did you extract Nutrition5k?"
        )

    # Load GT metadata
    dish_metadata = load_dish_metadata(metadata_dir)
    if not dish_metadata:
        raise SystemExit("No dish metadata found. Ensure Nutrition5k metadata CSVs are available.")

    # Load predictions
    preds_path = args.predictions_csv.expanduser().resolve()
    if not preds_path.exists():
        raise SystemExit(f"Predictions CSV not found: {preds_path}")

    predictions: Dict[str, Dict[str, object]] = {}
    with preds_path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            did = (row.get("dish_id") or "").strip()
            if not did:
                continue
            if did in predictions:
                # Last-write wins
                pass
            ai_ing_str = row.get("ai_ingredients") or ""
            ai_ingredients = [s.strip() for s in ai_ing_str.split("|") if s.strip()]
            ai_cal = _to_float(row.get("ai_calories"))
            ai_wt = _to_float(row.get("ai_weight"))
            predictions[did] = {
                "ai_ingredients": ai_ingredients,
                "ai_calories": ai_cal,
                "ai_weight": ai_wt,
                "ai_description": row.get("ai_description") or "",
            }

    available_dish_dirs = [
        path
        for path in imagery_root.iterdir()
        if path.is_dir() and (path / "rgb.png").exists()
    ]
    if not available_dish_dirs:
        raise SystemExit("No overhead meal photos found (rgb.png). Confirm Nutrition5k extraction.")

    available_dish_ids = sorted(path.name for path in available_dish_dirs if path.name in dish_metadata)
    if not available_dish_ids:
        raise SystemExit("No overlap between metadata dish IDs and available rgb.png images.")

    # Filter to dishes for which we have predictions
    dish_ids = [did for did in available_dish_ids if did in predictions]
    if not dish_ids:
        raise SystemExit("No overlap between predictions CSV and available dishes.")

    results: List[Dict[str, object]] = []
    calorie_pairs: List[Tuple[float, float, float]] = []
    weight_pairs: List[Tuple[float, float, float]] = []
    ingredient_scores: List[float] = []
    dish_details: List[Dict[str, object]] = []

    analysed = 0
    skipped = 0

    print(f"Evaluating {len(dish_ids)} dishes with predictions from '{args.model_name}'...")

    for did in dish_ids:
        info = dish_metadata[did]
        true_ingredients_full = [str(x).lower() for x in info.get("true_ingredients", [])]
        if any("deprecated" in ph for ph in true_ingredients_full):
            skipped += 1
            continue
        if any("plate only" in ph or ph.strip() == "plate" for ph in true_ingredients_full):
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

        pred = predictions[did]
        ai_ingredients_raw = pred.get("ai_ingredients", []) or []
        ai_ingredients_raw = [str(x).strip() for x in ai_ingredients_raw if str(x).strip()]
        # Use same prediction filtering as original evaluator
        pred_map: Dict[str, str] = {}
        for name in ai_ingredients_raw:
            key = name.lower()
            if key and key not in pred_map:
                pred_map[key] = name
        predicted_ingredients = _filter_predicted_ingredients(list(pred_map.values()))

        ai_description = str(pred.get("ai_description") or "")
        ai_calories = pred.get("ai_calories")
        ai_weight = pred.get("ai_weight")

        visible_true_ingredients = _filter_visible_true_ingredients(info)
        ingredient_jaccard = compute_jaccard(visible_true_ingredients, predicted_ingredients)

        row = {
            "dish_id": did,
            "true_ingredients": " | ".join(visible_true_ingredients),
            "true_calories": f"{float(true_cal):.2f}" if isinstance(true_cal, (int, float)) else "",
            "true_weight": f"{float(true_wt):.2f}" if isinstance(true_wt, (int, float)) else "",
            "ai_description": ai_description,
            "ai_calories": f"{float(ai_calories):.2f}" if isinstance(ai_calories, (int, float)) else "",
            "ai_weight": f"{float(ai_weight):.2f}" if isinstance(ai_weight, (int, float)) else "",
            "ingredient_jaccard": f"{ingredient_jaccard:.3f}" if not math.isnan(ingredient_jaccard) else "",
        }

        detail = {
            "dish_id": did,
            "image_path": imagery_root / did / "rgb.png",
            "true_ingredients": visible_true_ingredients,
            "true_calories": true_cal,
            "true_weight": true_wt,
            "ai_ingredients": predicted_ingredients,
            "ai_calories": float(ai_calories) if isinstance(ai_calories, (int, float)) else math.nan,
            "ai_weight": float(ai_weight) if isinstance(ai_weight, (int, float)) else math.nan,
            "ingredient_jaccard": ingredient_jaccard,
        }

        cal_pair = None
        wt_pair = None
        if isinstance(true_cal, (int, float)) and isinstance(ai_calories, (int, float)):
            cal_pair = (float(true_cal), float(ai_calories), ingredient_jaccard)
        if isinstance(true_wt, (int, float)) and isinstance(ai_weight, (int, float)):
            wt_pair = (float(true_wt), float(ai_weight), ingredient_jaccard)

        results.append(row)
        dish_details.append(detail)
        if cal_pair is not None:
            calorie_pairs.append(cal_pair)
            ingredient_scores.append(cal_pair[2])
        if wt_pair is not None:
            weight_pairs.append(wt_pair)

        analysed += 1

    # Write CSV
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
    jaccard_values = [
        d.get("ingredient_jaccard")
        for d in dish_details
        if isinstance(d.get("ingredient_jaccard"), (int, float)) and not math.isnan(d.get("ingredient_jaccard"))
    ]

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
    if args.output_montage and slots_per_page > 0 and eligible_montage and args.montage_pages > 0:
        total_pages_possible = math.ceil(total_candidates_montage / slots_per_page)
        pages_to_render = min(args.montage_pages, total_pages_possible)
    else:
        pages_to_render = 0

    summary_info = {
        "initial": len(dish_ids),
        "post_filter": len(dish_ids),
        "sampled": len(dish_ids),
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

    if pages_to_render > 0 and args.output_montage:
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
    elif args.output_montage:
        print("[WARN] No montage images generated (insufficient low-overlap dishes).")

    cal_stats = summarise_errors([(p[0], p[1]) for p in calorie_pairs]) if calorie_pairs else {"mae": math.nan, "rmse": math.nan}
    wt_stats = summarise_errors([(p[0], p[1]) for p in weight_pairs]) if weight_pairs else {"mae": math.nan, "rmse": math.nan}
    valid_jaccard = [score for score in ingredient_scores if not math.isnan(score)]

    print(f"\nEvaluation complete for model='{args.model_name}'.")
    print(f"  Dishes with images & predictions: {len(dish_ids)}")
    print(f"  Dishes analysed                 : {analysed}")
    print(f"  Dishes skipped                  : {skipped}")
    print(f"  Montage candidates              : {total_candidates_montage}")
    print(f"  Montage pages                   : {pages_to_render}")
    print(f"  CSV written to                  : {args.output_csv}")
    print(f"  Plot saved to                   : {args.output_plot}")
    print(f"  Calorie MAE/RMSE                : {cal_stats['mae']:.2f} / {cal_stats['rmse']:.2f}")
    print(f"  Weight MAE/RMSE                 : {wt_stats['mae']:.2f} / {wt_stats['rmse']:.2f}")
    if valid_jaccard:
        print(
            f"  Ingredient Jaccard -- mean: {np.mean(valid_jaccard):.2f}, "
            f"median: {np.median(valid_jaccard):.2f}"
        )
    if generated_montages:
        montage_names = ", ".join(path.name for path in generated_montages)
        print(f"  Montage files                   : {montage_names}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


