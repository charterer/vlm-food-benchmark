#!/usr/bin/env python3
"""
Export game results and recalculate Jaccard for all models.
Reads from CSV annotations (portable), outputs comparison CSV.
"""

import csv
import json
import re
from pathlib import Path
from datetime import datetime
from collections import defaultdict

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
EXPORTS_DIR = APP_DIR / "exports"
BENCHMARK_DIR = APP_DIR.parent / "pipeline" / "outputs" / "benchmark_results"

ANNOTATIONS_CSV = DATA_DIR / "annotations.csv"
EQUIVALENCES_CSV = DATA_DIR / "equivalences.csv"

# Fuzzy matching (same as app.py)
MODIFIERS = {'roasted','grilled','fried','baked','steamed','sauteed','boiled','raw','fresh','frozen','canned','dried','cooked','sliced','diced','chopped','minced','cubed','shredded','red','green','yellow','orange','white','black','brown','purple','sweet','sour','spicy','large','small','medium','organic','natural','plain','seasoned','baby','young','smoked','cured','pickled','mashed','pureed','crushed','ground','crispy','crunchy','soft','tender','mixed','assorted','various'}

def normalize(name):
    name = name.lower().strip()
    name = re.sub(r'\s*\(.*\)\s*', '', name)
    return re.sub(r'\s+', ' ', name)

def get_base(name):
    words = normalize(name).split()
    while words and words[0] in MODIFIERS: words.pop(0)
    while words and words[-1] in MODIFIERS: words.pop()
    return ' '.join(words) or normalize(name)

def fuzzy_match(a, b):
    n1, n2 = normalize(a), normalize(b)
    if n1 == n2: return True
    b1, b2 = get_base(a), get_base(b)
    if b1 == b2 or b1 in b2 or b2 in b1: return True
    if b1.rstrip('s') == b2.rstrip('s'): return True
    return False

def load_annotations():
    """Load annotations from CSV."""
    annotations = {}
    if ANNOTATIONS_CSV.exists():
        with open(ANNOTATIONS_CSV, 'r') as f:
            for row in csv.DictReader(f):
                if row['completed'] == '1':
                    key = f"{row['dataset']}:{row['image_id']}"
                    annotations[key] = {
                        'dataset': row['dataset'],
                        'image_id': row['image_id'],
                        'gt_verified': json.loads(row['gt_verified'] or '{}'),
                        'user_added': json.loads(row['user_added'] or '[]'),
                        'original_jaccard': float(row['original_jaccard'] or 0),
                        'updated_jaccard': float(row['updated_jaccard'] or 0)
                    }
    return annotations

def load_equivalences():
    """Load global equivalences."""
    equivs = defaultdict(list)
    if EQUIVALENCES_CSV.exists():
        with open(EQUIVALENCES_CSV, 'r') as f:
            for row in csv.DictReader(f):
                equivs[normalize(row['ai_ingredient'])].append(normalize(row['gt_ingredient']))
    return dict(equivs)

def extract_ai_ingredients(description):
    """Extract ingredients from AI description."""
    if not description: return []
    stopwords = {'a','an','the','of','with','and','on','in','to','for','plate','bowl','dish','serving','portion','small','large','medium','approximately','about','some','several','few','pieces','piece','slices','slice','cubes','cube','chunks','white','appears','consisting','containing','topped','mixed','assorted','various','total','cooked','raw','fresh','diced','chopped','sliced','roasted','grilled','fried','baked','steamed','sauteed','boiled'}
    desc = description.lower()
    desc = re.sub(r'^(a |an |the )?(plate|bowl|dish|serving|small plate|small bowl) (of |with |containing )?', '', desc)
    ingredients = []
    for part in re.split(r',\s*|\s+and\s+|\s+with\s+', desc):
        part = part.strip()
        part = re.sub(r'[.,!?;:]$', '', part)
        part = re.sub(r'^\d+[-–]?\d*\s*', '', part)
        words = part.split()
        while words and words[0].lower() in stopwords: words.pop(0)
        while words and words[-1].lower() in stopwords: words.pop()
        part = ' '.join(words)
        if 3 <= len(part) <= 50 and part.lower() not in stopwords:
            ingredients.append(part)
    return list(dict.fromkeys(ingredients))

def calc_jaccard_with_equiv(gt_list, ai_list, equivs):
    """Calculate Jaccard with fuzzy matching + equivalences."""
    if not gt_list and not ai_list: return 1.0
    if not gt_list or not ai_list: return 0.0
    
    matched = 0
    matched_gt = set()
    
    for ai in ai_list:
        ai_norm = normalize(ai)
        for gt in gt_list:
            if gt in matched_gt: continue
            if fuzzy_match(ai, gt):
                matched += 1
                matched_gt.add(gt)
                break
        else:
            # Check equivalences
            for eq_gt in equivs.get(ai_norm, []):
                for gt in gt_list:
                    if gt in matched_gt: continue
                    if normalize(gt) == eq_gt or get_base(gt) == get_base(eq_gt):
                        matched += 1
                        matched_gt.add(gt)
                        break
                if matched > len(matched_gt) - 1: break
    
    union = len(gt_list) + len(ai_list) - matched
    return matched / union if union > 0 else 1.0

def process_model_csv(csv_path, validated_gt, equivs):
    """Process a model CSV and recalculate Jaccard."""
    results = []
    
    with open(csv_path, 'r') as f:
        for row in csv.DictReader(f):
            dish_id = row.get('dish_id', row.get('image_id', ''))
            
            # Get original GT
            gt_col = 'true_ingredients' if 'true_ingredients' in row else 'true_labels'
            sep = '|' if 'true_ingredients' in row else ', '
            original_gt = [i.strip() for i in (row.get(gt_col, '') or '').split(sep) if i.strip()]
            
            # Get validated GT if available
            key = f"nutrition5k:{dish_id}"
            if key in validated_gt:
                gt_for_calc = validated_gt[key]
            else:
                gt_for_calc = original_gt
            
            # Get AI ingredients
            ai_col = 'ai_description' if 'ai_description' in row else 'pred_labels'
            ai_raw = row.get(ai_col, '')
            if 'ai_description' in row:
                ai_list = extract_ai_ingredients(ai_raw)
            else:
                ai_list = [i.strip() for i in ai_raw.split(', ') if i.strip()]
            
            # Get original Jaccard
            j_col = 'ingredient_jaccard' if 'ingredient_jaccard' in row else 'jaccard'
            try:
                orig_j = float(row.get(j_col, 0) or 0)
            except: orig_j = 0
            
            # Calculate new Jaccard
            new_j = calc_jaccard_with_equiv(gt_for_calc, ai_list, equivs)
            
            results.append({
                'image_id': dish_id,
                'original_jaccard': orig_j,
                'validated_jaccard': new_j,
                'improvement': new_j - orig_j
            })
    
    return results

def main():
    print("=" * 60)
    print("Food Annotation Game - Export Results")
    print("=" * 60)
    
    EXPORTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Load annotations
    annotations = load_annotations()
    print(f"\nLoaded {len(annotations)} completed annotations")
    
    if not annotations:
        print("No completed annotations found. Play the game first!")
        return
    
    # Build validated GT
    validated_gt = {}
    for key, ann in annotations.items():
        # This is simplified - full version would parse original CSV
        validated_gt[key] = ann['user_added']  # Just tracking user additions for now
    
    # Load equivalences
    equivs = load_equivalences()
    print(f"Loaded {len(equivs)} ingredient equivalence mappings")
    
    # Export annotations summary
    summary_file = EXPORTS_DIR / f"annotations_summary_{timestamp}.csv"
    with open(summary_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, ['dataset', 'image_id', 'original_jaccard', 'updated_jaccard', 'improvement'])
        writer.writeheader()
        for ann in annotations.values():
            writer.writerow({
                'dataset': ann['dataset'],
                'image_id': ann['image_id'],
                'original_jaccard': round(ann['original_jaccard'], 4),
                'updated_jaccard': round(ann['updated_jaccard'], 4),
                'improvement': round(ann['updated_jaccard'] - ann['original_jaccard'], 4)
            })
    print(f"\nSaved: {summary_file}")
    
    # Export equivalences
    equiv_file = EXPORTS_DIR / f"equivalences_{timestamp}.csv"
    with open(equiv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, ['ai_ingredient', 'gt_ingredient'])
        writer.writeheader()
        for ai, gts in equivs.items():
            for gt in gts:
                writer.writerow({'ai_ingredient': ai, 'gt_ingredient': gt})
    print(f"Saved: {equiv_file}")
    
    # Process benchmark models (if available)
    model_csvs = list(BENCHMARK_DIR.glob("*P3_full3229.csv")) if BENCHMARK_DIR.exists() else []
    
    if model_csvs:
        print(f"\nRecalculating Jaccard for {len(model_csvs)} models...")
        
        comparison = []
        for csv_path in sorted(model_csvs):
            model_name = csv_path.stem.replace("_P3_full3229", "")
            results = process_model_csv(csv_path, validated_gt, equivs)
            
            orig_mean = sum(r['original_jaccard'] for r in results) / len(results)
            new_mean = sum(r['validated_jaccard'] for r in results) / len(results)
            
            comparison.append({
                'model': model_name,
                'original_jaccard': round(orig_mean, 4),
                'validated_jaccard': round(new_mean, 4),
                'improvement': round(new_mean - orig_mean, 4)
            })
            
            print(f"  {model_name}: {orig_mean:.4f} → {new_mean:.4f} ({new_mean-orig_mean:+.4f})")
        
        comp_file = EXPORTS_DIR / f"model_comparison_{timestamp}.csv"
        with open(comp_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, ['model', 'original_jaccard', 'validated_jaccard', 'improvement'])
            writer.writeheader()
            writer.writerows(comparison)
        print(f"\nSaved: {comp_file}")
    
    print(f"\nAll exports saved to: {EXPORTS_DIR}/")

if __name__ == '__main__':
    main()
