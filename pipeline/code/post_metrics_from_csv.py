#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:
    plt = None  # type: ignore


def _to_float(val: str) -> Optional[float]:
    """Parse a float from CSV cell; return None on empty/invalid."""
    if val is None:
        return None
    s = str(val).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def load_pairs_and_scores(csv_path: Path) -> Tuple[
    List[Tuple[float, float, float]],
    List[Tuple[float, float, float]],
    List[float],
]:
    """
    Read CSV rows and extract:
    - calorie_pairs: (true_cal, ai_cal, jaccard)
    - weight_pairs: (true_wt, ai_wt, jaccard)
    - ingredient_scores: [jaccard]
    """
    calorie_pairs: List[Tuple[float, float, float]] = []
    weight_pairs: List[Tuple[float, float, float]] = []
    ingredient_scores: List[float] = []

    with csv_path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            ai_desc = (row.get("ai_description") or "").strip()
            is_error = ai_desc.startswith("Error:")
            true_cal = _to_float(row.get("true_calories"))
            ai_cal = None if is_error else _to_float(row.get("ai_calories"))
            true_wt = _to_float(row.get("true_weight"))
            ai_wt = None if is_error else _to_float(row.get("ai_weight"))
            jacc = None if is_error else _to_float(row.get("ingredient_jaccard"))

            if jacc is not None and not math.isnan(jacc):
                ingredient_scores.append(float(jacc))

            if true_cal is not None and ai_cal is not None:
                calorie_pairs.append((float(true_cal), float(ai_cal), float(jacc) if jacc is not None else float("nan")))
            if true_wt is not None and ai_wt is not None:
                weight_pairs.append((float(true_wt), float(ai_wt), float(jacc) if jacc is not None else float("nan")))

    return calorie_pairs, weight_pairs, ingredient_scores


def main() -> int:
    parser = argparse.ArgumentParser(description="Recompute metrics/plots from an evaluation CSV (possibly postprocessed).")
    parser.add_argument("--input-csv", type=Path, required=True, help="Path to input CSV (e.g., nutrition5k_gemini_eval_post.csv).")
    parser.add_argument("--output-plot", type=Path, required=False, help="Destination path for plots image.")
    parser.add_argument("--cache-dir", type=Path, required=False, help="Path to per-dish cache directory to reconstruct mismatch panels.")
    parser.add_argument("--jaccard-threshold", type=float, default=0.5, help="Threshold for including dishes in mismatch analysis.")
    args = parser.parse_args()

    csv_path = args.input_csv.expanduser().resolve()
    if not csv_path.exists():
        raise SystemExit(f"Input CSV not found: {csv_path}")

    calorie_pairs, weight_pairs, ingredient_scores = load_pairs_and_scores(csv_path)

    # Import evaluator helpers for consistent metrics/plots
    try:
        import evaluate_nutrition5k as ev
    except Exception as exc:
        print(f"[WARN] Could not import evaluator helpers: {exc}")
        ev = None  # type: ignore

    def summarise_errors_local(pairs_2d: List[Tuple[float, float]]) -> Dict[str, float]:
        diffs = [pred - true for true, pred in pairs_2d]
        if not diffs:
            return {"mae": float("nan"), "rmse": float("nan")}
        mae = sum(abs(d) for d in diffs) / len(diffs)
        rmse = math.sqrt(sum(d * d for d in diffs) / len(diffs))
        return {"mae": mae, "rmse": rmse}

    cal_pairs_2d = [(p[0], p[1]) for p in calorie_pairs]
    wt_pairs_2d = [(p[0], p[1]) for p in weight_pairs]

    if ev is not None and hasattr(ev, "summarise_errors"):
        cal_stats = ev.summarise_errors(cal_pairs_2d) if cal_pairs_2d else {"mae": float("nan"), "rmse": float("nan")}
        wt_stats = ev.summarise_errors(wt_pairs_2d) if wt_pairs_2d else {"mae": float("nan"), "rmse": float("nan")}
    else:
        cal_stats = summarise_errors_local(cal_pairs_2d)
        wt_stats = summarise_errors_local(wt_pairs_2d)

    valid_jaccard = [s for s in ingredient_scores if not math.isnan(s)]
    mean_j = float(np.mean(valid_jaccard)) if valid_jaccard else float("nan")
    median_j = float(np.median(valid_jaccard)) if valid_jaccard else float("nan")

    print("Postprocessed CSV metrics:")
    print(f"  Calorie MAE/RMSE: {cal_stats['mae']:.2f} / {cal_stats['rmse']:.2f}")
    print(f"  Weight  MAE/RMSE: {wt_stats['mae']:.2f} / {wt_stats['rmse']:.2f}")
    if valid_jaccard:
        print(f"  Ingredient Jaccard -- mean: {mean_j:.2f}, median: {median_j:.2f}")

    mismatch_summary = None
    dish_details: List[Dict[str, object]] = []
    if args.cache_dir:
        cache_dir = args.cache_dir.expanduser().resolve()
        if cache_dir.exists():
            # Reconstruct dish_details minimal structures from CSV + cache
            with csv_path.open("r", newline="") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    dish_id = (row.get("dish_id") or "").strip()
                    if not dish_id:
                        continue
                    true_ing_str = row.get("true_ingredients") or ""
                    true_ingredients = [s.strip() for s in true_ing_str.split("|") if s.strip()]
                    # Split tolerant of both " | " and "|" separators:
                    true_ingredients = [t.strip() for part in true_ingredients for t in part.split(" | ") if t.strip()]
                    jacc = _to_float(row.get("ingredient_jaccard"))
                    # Load predicted ingredients from cache
                    cache_file = cache_dir / f"{dish_id}.json"
                    predicted_ingredients: List[str] = []
                    try:
                        if cache_file.exists():
                            import json
                            with cache_file.open("r") as cf:
                                data = json.load(cf)
                            # Cache may store the analysis payload directly, or wrapped in {"analysis": {...}}
                            analysis = data.get("analysis", data) if isinstance(data, dict) else data
                            if isinstance(analysis, dict):
                                items = analysis.get("items") or []
                                for it in items:
                                    if isinstance(it, dict) and it.get("name"):
                                        predicted_ingredients.append(str(it["name"]))
                    except Exception:
                        # Skip if cache missing or malformed
                        pass
                    if predicted_ingredients:
                        dish_details.append({
                            "dish_id": dish_id,
                            "true_ingredients": true_ingredients,
                            "ai_ingredients": predicted_ingredients,
                            "ingredient_jaccard": float(jacc) if jacc is not None else float("nan"),
                        })
    # Compute mismatch summary using local helpers if ev is unavailable
    if not mismatch_summary and dish_details:
        # Token helpers (subset of evaluator logic)
        _STOPWORDS = {
            "and","with","of","a","an","the","on","in","to","for","some","type","kind","few","several",
            "variety","assortment","piece","pieces","slice","slices","stick","sticks","semi","circles",
            "light","small","big","little","lot","seems","appears","salad","dressing","vinaigrette","sauce",
            "coating","cooked","raw","fried","grilled","roasted","baked","mashed","sauteed","dark","pale",
            "fresh","white","black","brown","red","green","yellow","orange","purple","blue","gold","golden",
            "creamy","buttery","zesty","savory","sweet","tangy","spicy","mild","wild","mixed","assorted",
            "varied","sliced","diced","chopped","cubed","minced","shredded","grated","julienned","whole",
            "halved","quartered","peeled","toasted","ripe","dry","dried","seasoned","plain",
            "other","others","ingredient","ingredients","etc","etcetera","misc","miscellaneous","unknown",
            "none","null","nil","na","nan","n","food","foods","item","items","stuff","anything","something",
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
        import re as _re
        _NONFOOD_PHRASE_RE = _re.compile(
            r"^(?:n/?a|nan|none|null|nil|unknown|not\\s+available|no\\s+ingredients?)$",
            _re.IGNORECASE,
        )
        def _normalize_token(token: str) -> str:
            t = token.lower()
            if t.endswith("ies") and len(t) > 3:
                t = t[:-3] + "y"
            elif t.endswith("oes") and len(t) > 3:
                t = t[:-2]
            elif t.endswith("es") and len(t) > 3:
                suffix = t[-3:]
                if suffix in {"xes","ches","shes","ses","zes"}:
                    t = t[:-2]
                else:
                    t = t[:-1]
            elif t.endswith("s") and len(t) > 3:
                t = t[:-1]
            if t.endswith("berry"):
                t = "berry"
            return t
        def _phrase_to_tokens(phrase: str) -> List[str]:
            if _NONFOOD_PHRASE_RE.match((phrase or "").strip()):
                return []
            raw = _re.findall(r"[a-zA-Z]+", phrase.lower())
            canon: List[str] = []
            for tok in raw:
                t = _normalize_token(tok)
                if not t or t in _STOPWORDS:
                    continue
                t = _SYNONYM_TOKEN.get(t, t)
                canon.append(t)
            # composites
            if ("sweet" in canon and "potato" in canon) and "sweetpotato" not in canon:
                canon.append("sweetpotato")
            if ("grape" in canon and "tomato" in canon) and "grapetomato" not in canon:
                canon.append("grapetomato")
            # dedupe
            seen = set()
            out: List[str] = []
            for t in canon:
                if t not in seen:
                    out.append(t)
                    seen.add(t)
            return out
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
            if isinstance(j, (int, float)) and not math.isnan(j) and j < args.jaccard_threshold:
                matched = true_tokens & pred_tokens
                fn_ctr.update(true_tokens - matched)
                fp_ctr.update(pred_tokens - matched)
        mismatch_summary = {
            "fn_tokens": dict(fn_ctr),
            "fp_tokens": dict(fp_ctr),
            "true_totals": dict(true_totals),
            "pred_totals": dict(pred_totals),
        }

    if args.output_plot:
        if ev is not None and hasattr(ev, "build_top_plots"):
            out_path = args.output_plot.expanduser().resolve()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                summary_info = {
                    "initial": len(calorie_pairs) or len(weight_pairs) or 0,
                    "dishes": len(calorie_pairs) or len(weight_pairs) or 0,
                    "mean_jaccard": mean_j,
                }
                ev.build_top_plots(
                    calorie_pairs=calorie_pairs,
                    weight_pairs=weight_pairs,
                    ingredient_scores=ingredient_scores,
                    output_path=out_path,
                    mismatch_summary=mismatch_summary,
                    summary_info=summary_info,
                )
                print(f"  Plot saved to: {out_path}")
            except Exception as exc:
                print(f"[WARN] Failed to render plot: {exc}")
        else:
            # Local minimal plotting fallback
            if plt is None:
                print("[WARN] Matplotlib unavailable; cannot generate plot.")
            else:
                out_path = args.output_plot.expanduser().resolve()
                out_path.parent.mkdir(parents=True, exist_ok=True)
                import matplotlib.gridspec as gridspec
                fig = plt.figure(figsize=(24, 22))
                gs = gridspec.GridSpec(3, 3, width_ratios=[1.0,1.0,1.0], height_ratios=[1.5,0.45,5.5], hspace=0.6, wspace=0.32)
                ax_cal = fig.add_subplot(gs[0,0])
                ax_weight = fig.add_subplot(gs[0,1])
                ax_jacc = fig.add_subplot(gs[0,2])
                summary_spec = gs[1, :].subgridspec(1, 3, wspace=0.12)
                summary_axes = [fig.add_subplot(summary_spec[0, idx]) for idx in range(3)]
                for ax in summary_axes:
                    ax.axis("off")
                mismatch_spec = gs[2, :].subgridspec(1, 2, wspace=0.3)
                ax_fn = fig.add_subplot(mismatch_spec[0, 0])
                ax_fp = fig.add_subplot(mismatch_spec[0, 1])
                # Calories scatter
                if calorie_pairs:
                    cal_true_all = np.array([p[0] for p in calorie_pairs])
                    cal_pred_all = np.array([p[1] for p in calorie_pairs])
                    cal_j_all = np.array([p[2] for p in calorie_pairs])
                    cal_min, cal_max = 0.0, 1500.0
                    sc = ax_cal.scatter(cal_true_all, cal_pred_all, c=np.nan_to_num(cal_j_all, nan=0.0), cmap="viridis", alpha=0.7, edgecolor="none", s=2)
                    ax_cal.plot([cal_min, cal_max], [cal_min, cal_max], linestyle="--", color="gray", linewidth=1.0)
                    ax_cal.set_xlim(cal_min, cal_max)
                    ax_cal.set_ylim(cal_min, cal_max)
                    fig.colorbar(sc, ax=ax_cal, label="Ingredient Jaccard")
                ax_cal.set_xlabel("True Calories (kcal)")
                ax_cal.set_ylabel("AI Estimated Calories (kcal)")
                ax_cal.set_title("Calories: AI vs Truth")
                # Weight scatter
                if weight_pairs:
                    wt_true_all = np.array([p[0] for p in weight_pairs])
                    wt_pred_all = np.array([p[1] for p in weight_pairs])
                    wt_j_all = np.array([p[2] for p in weight_pairs])
                    wt_min, wt_max = 0.0, 1000.0
                    sw = ax_weight.scatter(wt_true_all, wt_pred_all, c=np.nan_to_num(wt_j_all, nan=0.0), cmap="viridis", alpha=0.7, edgecolor="none", s=2)
                    ax_weight.plot([wt_min, wt_max], [wt_min, wt_max], linestyle="--", color="gray", linewidth=1.0)
                    ax_weight.set_xlim(wt_min, wt_max)
                    ax_weight.set_ylim(wt_min, wt_max)
                    fig.colorbar(sw, ax=ax_weight, label="Ingredient Jaccard")
                ax_weight.set_xlabel("True Weight (g)")
                ax_weight.set_ylabel("AI Estimated Weight (g)")
                ax_weight.set_title("Weight: AI vs Truth")
                # Jacc histogram
                val_scores = [s for s in ingredient_scores if not math.isnan(s)]
                if val_scores:
                    bins = np.linspace(0, 1, 21)
                    ax_jacc.hist(val_scores, bins=bins, color="#1f77b4", alpha=0.85, edgecolor="black")
                    ax_jacc.set_xlim(0, 1)
                    ax_jacc.text(0.03, 0.97, f"Mean: {mean_j:.2f}\nMedian: {median_j:.2f}", transform=ax_jacc.transAxes, verticalalignment="top", bbox={"facecolor":"white","alpha":0.8,"pad":6}, fontsize=9)
                ax_jacc.set_xlabel("Ingredient Jaccard (AI vs Truth)")
                ax_jacc.set_ylabel("Dish Count")
                ax_jacc.set_title("Ingredient Overlap Distribution")
                # Summary text
                def _box(ax, text):
                    ax.text(0.0, 1.0, text, ha="left", va="top", fontsize=11, wrap=True, bbox={"facecolor":"#f7f7f7","edgecolor":"#444444","boxstyle":"round,pad=0.6"})
                _box(summary_axes[0], f"Initial images: {len(calorie_pairs) or len(weight_pairs)}\nAnalyzed images: {len(calorie_pairs) or len(weight_pairs)}")
                # Local log-fit summaries
                def _logfit(pairs_2d: List[Tuple[float,float]]) -> Optional[Dict[str,float]]:
                    f = [(float(x), float(y)) for x,y in pairs_2d if x>0 and y>0]
                    if len(f) < 2:
                        return None
                    xs = np.array([p[0] for p in f], dtype=float)
                    ys = np.array([p[1] for p in f], dtype=float)
                    logx = np.log(xs); logy = np.log(ys)
                    beta, alpha = np.polyfit(logx, logy, 1)
                    y_hat = np.exp(alpha + beta*logx)
                    ape = np.abs(ys - y_hat) / ys * 100.0
                    p50 = float(np.nanpercentile(ape, 50)); p90 = float(np.nanpercentile(ape, 90))
                    mean_x = float(np.mean(xs)); mean_y = float(np.mean(ys))
                    var_x = float(np.var(xs)); var_y = float(np.var(ys))
                    cov_xy = float(np.cov(xs, ys, bias=True)[0,1])
                    den = var_x + var_y + (mean_x - mean_y)**2
                    ccc = float((2*cov_xy)/den) if den>0 else float("nan")
                    return {"alpha": float(alpha), "beta": float(beta), "p50": p50, "p90": p90, "ccc": ccc}
                cal_fit = _logfit([(p[0],p[1]) for p in calorie_pairs])
                wt_fit = _logfit([(p[0],p[1]) for p in weight_pairs])
                if cal_fit:
                    _box(summary_axes[1], f"Calories:\nBeta (slope): {cal_fit['beta']:.3f}\nP50 APE: {cal_fit['p50']:.1f}%\nP90 APE: {cal_fit['p90']:.1f}%\nCCC: {cal_fit['ccc']:.2f}")
                if wt_fit:
                    _box(summary_axes[2], f"Weight:\nBeta (slope): {wt_fit['beta']:.3f}\nP50 APE: {wt_fit['p50']:.1f}%\nP90 APE: {wt_fit['p90']:.1f}%\nCCC: {wt_fit['ccc']:.2f}")
                # Mismatch panels
                if mismatch_summary:
                    from collections import Counter
                    fn_ctr = Counter(mismatch_summary.get("fn_tokens", {}))
                    fp_ctr = Counter(mismatch_summary.get("fp_tokens", {}))
                    true_totals = Counter(mismatch_summary.get("true_totals", {}))
                    pred_totals = Counter(mismatch_summary.get("pred_totals", {}))
                    top_fn = fn_ctr.most_common(100)
                    if top_fn:
                        fn_labels = [it[0] for it in reversed(top_fn)]
                        fn_vals = [it[1] for it in reversed(top_fn)]
                        fn_totals = [true_totals.get(lbl, 0) for lbl in fn_labels]
                        pos = np.arange(len(fn_labels))
                        ax_fn.barh(pos, fn_totals, color="#c8e6c9", alpha=0.9, height=0.8, label="Total appearances")
                        ax_fn.barh(pos, fn_vals, color="#228B22", alpha=0.9, height=0.45, label="False negatives")
                        ax_fn.set_yticks(pos); ax_fn.set_yticklabels(fn_labels, fontsize=8)
                        ax_fn.set_title("False Negatives (missed true tokens)"); ax_fn.set_xlabel("Count"); ax_fn.invert_yaxis()
                        ax_fn.legend(loc="lower right", fontsize=7)
                    top_fp = fp_ctr.most_common(100)
                    if top_fp:
                        fp_labels = [it[0] for it in reversed(top_fp)]
                        fp_vals = [it[1] for it in reversed(top_fp)]
                        fp_totals = [pred_totals.get(lbl, 0) for lbl in fp_labels]
                        pos2 = np.arange(len(fp_labels))
                        ax_fp.barh(pos2, fp_totals, color="#ffe0b2", alpha=0.9, height=0.8, label="Total appearances")
                        ax_fp.barh(pos2, fp_vals, color="#ff8c00", alpha=0.9, height=0.45, label="False positives")
                        ax_fp.set_yticks(pos2); ax_fp.set_yticklabels(fp_labels, fontsize=8)
                        ax_fp.set_title("False Positives (AI-only tokens)"); ax_fp.set_xlabel("Count"); ax_fp.invert_yaxis()
                        ax_fp.legend(loc="lower right", fontsize=7)
                fig.suptitle("AI vs Nutrition5k Ground Truth (Postprocessed)", fontsize=18, y=0.97)
                fig.subplots_adjust(left=0.06, right=0.97, top=0.93, bottom=0.06, hspace=0.5, wspace=0.28)
                fig.savefig(out_path, dpi=300)
                plt.close(fig)
                print(f"  Plot saved to: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


