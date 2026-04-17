#!/usr/bin/env python3
"""
Build a manual token categorization CSV by scanning the eval CSV (ground truth ingredients)
and the Gemini cache (AI ingredients). Produces a CSV with columns:
    token,count,main_category,subcategory
You can edit this CSV manually and use it with downstream aggregation scripts
to produce category-level confusion matrices.
"""
import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Optional, Tuple

# Keep stopwords aligned with evaluation pipeline
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
    "whole", "halved", "quartered", "peeled", "toasted",
    # user-provided
    "dish", "asparagu", "strip", "herb", "breast", "or", "stir", "breaded", "steamed", "scrambled",
    "couscou", "cube", "leafy", "chunk", "floret", "meat", "cob", "mustard", "seed", "country",
    "ground", "steak", "greek", "sweetpotato", "berry", "olive", "thigh", "slice", "roasted",
    "vegetables", "salad", "wild", "wheat", "leaves", "sauce", "dressing", "raw", "mixed", "greens", "green",
}

_SYNONYM_MAP = {
    "aubergine": "eggplant",
    "yams": "sweetpotato",
    "yam": "sweetpotato",
    "strawberry": "berry",
    "blueberry": "berry",
    "raspberry": "berry",
    "blackberry": "berry",
}

MAIN_CATEGORIES = [
    "Vegetables", "Fruits", "Grains", "Legumes/Nuts/Seeds",
    "Meat/Poultry", "Seafood", "Dairy", "Eggs",
    "Fats/Oils", "Sauces/Condiments", "Baked Goods",
    "Sweets/Desserts", "Beverages", "Herbs/Spices",
    "Mixed/Prepared Foods", "Other"
]

SUBCATEGORY_HINTS = {
    "leafy greens": ["spinach", "kale", "arugula", "romaine", "lettuce", "chard", "collard", "tatsoi", "mizuna"],
    "root vegetables": ["potato", "sweetpotato", "yam", "carrot", "beet", "radish", "turnip", "parsnip"],
    "brassicas": ["broccoli", "cauliflower", "cabbage", "brussels", "bokchoy"],
    "citrus": ["orange", "lemon", "lime", "grapefruit", "mandarin"],
    "berries": ["strawberry", "blueberry", "raspberry", "blackberry", "berry"],
    "poultry": ["chicken", "turkey", "duck"],
    "red meat": ["beef", "steak", "lamb", "mutton", "pork", "bacon"],
    "fish": ["salmon", "tuna", "cod", "tilapia", "mackerel", "sardine", "trout"],
    "shellfish": ["shrimp", "prawn", "crab", "lobster", "clam", "mussel", "oyster", "scallop"],
    "grains": ["rice", "bread", "pasta", "noodle", "quinoa", "oat", "barley", "couscous", "tortilla"],
    "legumes": ["bean", "chickpea", "lentil", "pea", "edamame", "soy"],
    "nuts/seeds": ["almond", "walnut", "peanut", "cashew", "pistachio", "seed", "sesame", "pumpkinseed", "sunflower"],
    "dairy": ["milk", "cheese", "yogurt", "cream", "butter"],
    "eggs": ["egg"],
    "oils/fats": ["oil", "oliveoil", "butter", "ghee", "lard", "mayo"],
    "sauces/condiments": ["sauce", "dressing", "ketchup", "mustard", "soy", "salsa", "pesto", "gravy", "vinegar", "aioli"],
    "baked goods": ["bread", "bagel", "bun", "roll", "croissant", "muffin", "cake", "cookie", "pastry", "tortilla", "wrap"],
    "desserts": ["icecream", "pudding", "brownie", "cupcake", "dessert", "cookie", "candy", "pie"],
    "beverages": ["juice", "soda", "coffee", "tea", "smoothie", "milkshake", "water"],
    "herbs/spices": ["herb", "basil", "cilantro", "parsley", "dill", "mint", "oregano", "rosemary", "thyme", "spice", "cumin", "paprika", "pepper", "chili"],
    "mixed/prepared": ["stirfry", "curry", "stew", "soup", "salad", "sandwich", "taco", "burrito", "pizza", "pasta", "casserole", "wrap", "bowl"],
}


def _normalize_token(token: str) -> str:
    t = token.lower()
    if t.endswith("ies") and len(t) > 3:
        t = t[:-3] + "y"
    elif t.endswith("oes") and len(t) > 3:
        t = t[:-2]
    elif t.endswith("es") and len(t) > 3:
        suffix = t[-3:]
        if suffix in {"xes", "ches", "shes", "ses", "zes"}:
            t = t[:-2]
        else:
            t = t[:-1]
    elif t.endswith("s") and len(t) > 3:
        t = t[:-1]
    if t.endswith("berry"):
        t = "berry"
    return t


def phrase_to_tokens(phrase: str) -> Tuple[str, ...]:
    raw = re.findall(r"[a-zA-Z]+", phrase.lower())
    canon = []
    for tok in raw:
        t = _normalize_token(tok)
        if not t or t in _STOPWORDS:
            continue
        t = _SYNONYM_MAP.get(t, t)
        canon.append(t)
    if ("sweet" in canon and "potato" in canon) and "sweetpotato" not in canon:
        canon.append("sweetpotato")
    if ("grape" in canon and "tomato" in canon) and "grapetomato" not in canon:
        canon.append("grapetomato")
    # unique preserve order
    seen = set()
    out = []
    for t in canon:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return tuple(out)


def heuristic_category(tok: str) -> Tuple[str, str]:
    n = tok
    # subcategory hints first
    for subcat, hints in SUBCATEGORY_HINTS.items():
        if any(n.startswith(h) or h in n for h in hints):
            sc = subcat
            mc = "Vegetables"
            if subcat in ["citrus", "berries", "leafy greens"]:
                mc = "Fruits" if subcat in ["citrus", "berries"] else "Vegetables"
            elif subcat in ["poultry", "red meat"]:
                mc = "Meat/Poultry"
            elif subcat in ["fish", "shellfish"]:
                mc = "Seafood"
            elif subcat in ["grains"]:
                mc = "Grains"
            elif subcat in ["legumes", "nuts/seeds"]:
                mc = "Legumes/Nuts/Seeds"
            elif subcat in ["dairy"]:
                mc = "Dairy"
            elif subcat in ["eggs"]:
                mc = "Eggs"
            elif subcat in ["oils/fats"]:
                mc = "Fats/Oils"
            elif subcat in ["sauces/condiments"]:
                mc = "Sauces/Condiments"
            elif subcat in ["baked goods"]:
                mc = "Baked Goods"
            elif subcat in ["desserts"]:
                mc = "Sweets/Desserts"
            elif subcat in ["beverages"]:
                mc = "Beverages"
            elif subcat in ["herbs/spices"]:
                mc = "Herbs/Spices"
            elif subcat in ["mixed/prepared"]:
                mc = "Mixed/Prepared Foods"
            return (mc, sc)
    # basic fallbacks
    if n in ["chicken", "turkey", "duck"]:
        return ("Meat/Poultry", "poultry")
    if n in ["beef", "steak", "pork", "lamb", "bacon"]:
        return ("Meat/Poultry", "red meat")
    if n in ["egg", "eggs"]:
        return ("Eggs", "eggs")
    if any(k in n for k in ["salmon", "tuna", "cod", "tilapia", "mackerel", "sardine", "trout", "fish"]):
        return ("Seafood", "fish")
    if any(k in n for k in ["milk", "cheese", "yogurt", "cream", "butter"]):
        return ("Dairy", "dairy")
    if any(k in n for k in ["rice", "bread", "pasta", "noodle", "quinoa", "oat", "barley", "tortilla"]):
        return ("Grains", "grains")
    if any(k in n for k in ["bean", "lentil", "chickpea", "pea", "edamame", "soy"]):
        return ("Legumes/Nuts/Seeds", "legumes")
    if any(k in n for k in ["almond", "walnut", "peanut", "cashew", "pistachio", "sesame", "seed"]):
        return ("Legumes/Nuts/Seeds", "nuts/seeds")
    if any(k in n for k in ["oil", "oliveoil", "lard", "ghee", "mayo"]):
        return ("Fats/Oils", "oils/fats")
    if any(k in n for k in ["sauce", "dressing", "ketchup", "mustard", "soy", "salsa", "pesto", "gravy", "vinegar", "aioli"]):
        return ("Sauces/Condiments", "sauces/condiments")
    if any(k in n for k in ["lettuce", "spinach", "kale", "broccoli", "carrot", "tomato", "onion", "pepper", "cabbage", "cauliflower", "mushroom"]):
        return ("Vegetables", "vegetables")
    if any(k in n for k in ["apple", "banana", "orange", "lemon", "berry", "grape", "pear", "mango", "pineapple"]):
        return ("Fruits", "fruits")
    if any(k in n for k in ["cookie", "cake", "icecream", "brownie", "candy", "dessert", "pie"]):
        return ("Sweets/Desserts", "desserts")
    if any(k in n for k in ["coffee", "tea", "soda", "juice", "smoothie", "water", "milkshake"]):
        return ("Beverages", "beverages")
    if any(k in n for k in ["curry", "stew", "soup", "salad", "sandwich", "taco", "burrito", "pizza", "casserole", "wrap", "bowl"]):
        return ("Mixed/Prepared Foods", "mixed/prepared")
    if any(k in n for k in ["basil", "cilantro", "parsley", "dill", "mint", "oregano", "rosemary", "thyme", "cumin", "paprika", "pepper", "chili"]):
        return ("Herbs/Spices", "herbs/spices")
    return ("Other", "Other")


def collect_tokens(input_csv: Path, cache_dir: Path) -> Counter:
    counts: Counter = Counter()
    with input_csv.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            dish_id = row.get("dish_id", "").strip()
            true_ing_str = row.get("true_ingredients", "") or ""
            true_phrases = [s.strip() for s in true_ing_str.split("|") if s.strip()]
            for phrase in true_phrases:
                for t in phrase_to_tokens(phrase):
                    counts[t] += 1
            # AI from cache
            cache_file = cache_dir / f"{dish_id}.json"
            if cache_file.exists():
                try:
                    data = json.loads(cache_file.read_text())
                    analysis = data.get("analysis", data) if isinstance(data, dict) else data
                    items = analysis.get("items", []) if isinstance(analysis, dict) else []
                    for item in items:
                        name = item.get("name") if isinstance(item, dict) else None
                        if not name:
                            continue
                        for t in phrase_to_tokens(str(name)):
                            counts[t] += 1
                except Exception:
                    continue
    return counts


def main() -> int:
    ap = argparse.ArgumentParser(description="Create a token categorization CSV from eval CSV + cache.")
    ap.add_argument("--input-csv", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--output-csv", type=Path, required=True)
    ap.add_argument("--min-count", type=int, default=2)
    args = ap.parse_args()

    input_csv = args.input_csv.expanduser().resolve()
    cache_dir = args.cache_dir.expanduser().resolve()
    output_csv = args.output_csv.expanduser().resolve()
    if not input_csv.exists():
        raise SystemExit(f"CSV not found: {input_csv}")
    if not cache_dir.exists():
        raise SystemExit(f"Cache not found: {cache_dir}")

    counts = collect_tokens(input_csv, cache_dir)
    rows = []
    for tok, cnt in counts.most_common():
        if cnt < args.min_count:
            continue
        mc, sc = heuristic_category(tok)
        rows.append((tok, cnt, mc, sc))

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["token", "count", "main_category", "subcategory"])
        w.writerows(rows)
    print(f"Wrote {len(rows)} tokens to: {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


