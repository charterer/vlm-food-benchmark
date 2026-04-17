#!/usr/bin/env python3
"""
Nutrition5k & FoodSeg103 Annotation Minigame
A crowdsourcing tool to validate AI predictions against ground truth labels.

Storage: CSV files (portable, git-friendly)
"""

import os
import re
import csv
import json
import random
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template, jsonify, request, send_from_directory

app = Flask(__name__)

# Blind mode: randomize columns and hide which is GT vs AI
BLIND_MODE = os.environ.get('BLIND_MODE', '').lower() in ('1', 'true', 'yes')

# Single-blind mode: two-phase annotation (Phase 1: merged list, Phase 2: equivalencies)
SINGLE_BLIND_MODE = os.environ.get('SINGLE_BLIND_MODE', '').lower() in ('1', 'true', 'yes')

# Paths
APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
EXPORTS_DIR = APP_DIR / "exports"

# JSON cache with structured AI predictions (items[].name)
# Contains clean ingredient names from the original API responses
NUTRITION5K_JSON_CACHE = APP_DIR / "json_cache"

# Dataset configurations
DATASETS = {
    'nutrition5k': {
        'name': 'Nutrition5k',
        'csv_path': APP_DIR / "gemini_2.0_flash_P3_full3229.csv",
        'images_dir': APP_DIR / "images",
        'image_pattern': '{dish_id}/rgb.png',  # dish_id folder with rgb.png
        'id_column': 'dish_id',
        'gt_column': 'true_ingredients',
        'gt_separator': '|',
        'ai_column': 'ai_description',  # fallback if JSON cache missing
        'ai_is_description': True,
        'json_cache_dir': NUTRITION5K_JSON_CACHE,  # primary source for AI ingredients
        'jaccard_column': 'ingredient_jaccard'
    },
    'foodseg103': {
        'name': 'FoodSeg103',
        'csv_path': APP_DIR / "foodseg103_gemini.csv",
        'images_dir': APP_DIR / "foodseg103_images",
        'image_pattern': '{file_name}',  # use file_name column
        'id_column': 'file_name',  # use file_name as ID
        'gt_column': 'true_labels',
        'gt_separator': ', ',
        'ai_column': 'pred_labels',  # already extracted
        'ai_is_description': False,
        'json_cache_dir': None,  # no JSON cache for foodseg103
        'jaccard_column': 'jaccard'
    }
}

# CSV file paths for annotations
ANNOTATIONS_CSV = DATA_DIR / "annotations.csv"
EQUIVALENCES_CSV = DATA_DIR / "equivalences.csv"

# CSV-based Storage

def load_annotations():
    """Load annotations from CSV."""
    annotations = {}
    if ANNOTATIONS_CSV.exists():
        with open(ANNOTATIONS_CSV, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                annotations[f"{row['dataset']}:{row['image_id']}"] = {
                    'dataset': row['dataset'],
                    'image_id': row['image_id'],
                    'annotator': row.get('annotator', ''),
                    'gt_verified': json.loads(row['gt_verified'] or '{}'),
                    'ai_verified': json.loads(row['ai_verified'] or '{}'),
                    'equivalences': json.loads(row['equivalences'] or '{}'),
                    'user_added': json.loads(row['user_added'] or '[]'),
                    'challenge_flag': row['challenge_flag'] == '1',
                    'completed': row['completed'] == '1',
                    'original_jaccard': float(row['original_jaccard'] or 0),
                    'updated_jaccard': float(row['updated_jaccard'] or 0),
                    'timestamp': row['timestamp'],
                    # Blind mode fields
                    'column_a_is_gt': row.get('column_a_is_gt', '1') == '1',
                    'col_a_rejected': json.loads(row.get('col_a_rejected') or '[]'),
                    'col_b_rejected': json.loads(row.get('col_b_rejected') or '[]'),
                    # Single-blind mode fields (source-separated)
                    'phase': row.get('phase', 'complete'),
                    'gt_approved': json.loads(row.get('gt_approved') or '[]'),
                    'gt_rejected': json.loads(row.get('gt_rejected') or '[]'),
                    'gt_unsure': json.loads(row.get('gt_unsure') or '[]'),
                    'ai_approved': json.loads(row.get('ai_approved') or '[]'),
                    'ai_rejected': json.loads(row.get('ai_rejected') or '[]'),
                    'ai_unsure': json.loads(row.get('ai_unsure') or '[]'),
                    'user_added': json.loads(row.get('user_added') or '[]'),
                    'phase2_equivalences': json.loads(row.get('phase2_equivalences') or '{}'),
                }
    return annotations

def save_annotations(annotations):
    """Save annotations to CSV."""
    DATA_DIR.mkdir(exist_ok=True)
    with open(ANNOTATIONS_CSV, 'w', newline='') as f:
        fieldnames = ['dataset', 'image_id', 'annotator', 'gt_verified', 'ai_verified', 'equivalences',
                      'user_added', 'challenge_flag', 'completed', 'original_jaccard',
                      'updated_jaccard', 'timestamp', 'column_a_is_gt', 'col_a_rejected', 'col_b_rejected',
                      'phase', 'gt_approved', 'gt_rejected', 'gt_unsure', 
                      'ai_approved', 'ai_rejected', 'ai_unsure', 'phase2_equivalences']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for key, ann in annotations.items():
            writer.writerow({
                'dataset': ann['dataset'],
                'image_id': ann['image_id'],
                'annotator': ann.get('annotator', ''),
                'gt_verified': json.dumps(ann['gt_verified']),
                'ai_verified': json.dumps(ann['ai_verified']),
                'equivalences': json.dumps(ann['equivalences']),
                'user_added': json.dumps(ann['user_added']),
                'challenge_flag': '1' if ann['challenge_flag'] else '0',
                'completed': '1' if ann['completed'] else '0',
                'original_jaccard': ann['original_jaccard'],
                'updated_jaccard': ann['updated_jaccard'],
                'timestamp': ann['timestamp'],
                # Blind mode fields
                'column_a_is_gt': '1' if ann.get('column_a_is_gt', True) else '0',
                'col_a_rejected': json.dumps(ann.get('col_a_rejected', [])),
                'col_b_rejected': json.dumps(ann.get('col_b_rejected', [])),
                # Single-blind mode fields (source-separated)
                'phase': ann.get('phase', 'complete'),
                'gt_approved': json.dumps(ann.get('gt_approved', [])),
                'gt_rejected': json.dumps(ann.get('gt_rejected', [])),
                'gt_unsure': json.dumps(ann.get('gt_unsure', [])),
                'ai_approved': json.dumps(ann.get('ai_approved', [])),
                'ai_rejected': json.dumps(ann.get('ai_rejected', [])),
                'ai_unsure': json.dumps(ann.get('ai_unsure', [])),
                'phase2_equivalences': json.dumps(ann.get('phase2_equivalences', {})),
            })

def load_equivalences():
    """Load global equivalences from CSV."""
    equivalences = {}
    if EQUIVALENCES_CSV.exists():
        with open(EQUIVALENCES_CSV, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = row['ai_ingredient']
                if key not in equivalences:
                    equivalences[key] = []
                equivalences[key].append({
                    'gt': row['gt_ingredient'],
                    'count': int(row['count'])
                })
    return equivalences

def save_equivalences(equivalences):
    """Save global equivalences to CSV."""
    DATA_DIR.mkdir(exist_ok=True)
    with open(EQUIVALENCES_CSV, 'w', newline='') as f:
        fieldnames = ['ai_ingredient', 'gt_ingredient', 'count']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for ai_ing, gt_list in equivalences.items():
            for item in gt_list:
                writer.writerow({
                    'ai_ingredient': ai_ing,
                    'gt_ingredient': item['gt'],
                    'count': item['count']
                })

def add_equivalence(ai_ingredient, gt_ingredient):
    """Add or increment a global equivalence."""
    equivs = load_equivalences()
    if ai_ingredient not in equivs:
        equivs[ai_ingredient] = []
    
    # Check if this mapping already exists
    for item in equivs[ai_ingredient]:
        if item['gt'] == gt_ingredient:
            item['count'] += 1
            save_equivalences(equivs)
            return
    
    # Add new mapping
    equivs[ai_ingredient].append({'gt': gt_ingredient, 'count': 1})
    save_equivalences(equivs)

# Fuzzy Matching

MODIFIERS = {
    'roasted', 'grilled', 'fried', 'baked', 'steamed', 'sauteed', 'boiled',
    'raw', 'fresh', 'frozen', 'canned', 'dried', 'cooked', 'uncooked',
    'sliced', 'diced', 'chopped', 'minced', 'cubed', 'shredded', 'grated',
    'whole', 'halved', 'quartered',
    'red', 'green', 'yellow', 'orange', 'white', 'black', 'brown', 'purple',
    'sweet', 'sour', 'spicy', 'hot', 'cold', 'warm',
    'large', 'small', 'medium', 'big', 'little', 'tiny',
    'organic', 'natural', 'plain', 'seasoned', 'marinated',
    'baby', 'young', 'mature', 'ripe', 'unripe',
    'wild', 'farm', 'smoked', 'cured', 'pickled', 'fermented',
    'mashed', 'pureed', 'crushed', 'ground',
    'crispy', 'crunchy', 'soft', 'tender',
    'mixed', 'assorted', 'various'
}

def normalize_ingredient(name):
    """Normalize ingredient name for comparison."""
    name = name.lower().strip()
    name = re.sub(r'\s*\(.*\)\s*', '', name)
    name = re.sub(r'\s+', ' ', name)
    return name

def get_base_ingredient(name):
    """Extract base ingredient by removing modifiers."""
    norm = normalize_ingredient(name)
    words = norm.split()
    while words and words[0] in MODIFIERS:
        words.pop(0)
    while words and words[-1] in MODIFIERS:
        words.pop()
    return ' '.join(words) if words else norm

def ingredients_match_fuzzy(ing1, ing2):
    """Check if two ingredients match using fuzzy logic."""
    norm1 = normalize_ingredient(ing1)
    norm2 = normalize_ingredient(ing2)
    if norm1 == norm2:
        return True
    
    base1 = get_base_ingredient(ing1)
    base2 = get_base_ingredient(ing2)
    if base1 == base2:
        return True
    if base1 in base2 or base2 in base1:
        return True
    if base1.rstrip('s') == base2.rstrip('s'):
        return True
    if base1.rstrip('es') == base2.rstrip('es'):
        return True
    
    return False

def get_auto_matches(gt_ingredients, ai_ingredients):
    """Get automatic fuzzy matches between GT and AI ingredients."""
    auto_matches = {}
    matched_gt = set()
    
    for ai_ing in ai_ingredients:
        for gt_ing in gt_ingredients:
            if gt_ing not in matched_gt and ingredients_match_fuzzy(ai_ing, gt_ing):
                auto_matches[ai_ing] = gt_ing
                matched_gt.add(gt_ing)
                break
    
    return auto_matches

def create_merged_ingredient_list(gt_ingredients, ai_ingredients):
    """
    Create a merged, deduplicated list of ingredients for single-blind mode.
    Uses fuzzy matching (not just exact) to merge overlapping items.
    Returns: list of {name, source: 'gt'|'ai'|'both', gt_name, ai_name, fuzzy_match: bool}
    """
    merged = []
    used_gt = set()
    used_ai = set()
    
    # First: fuzzy matches (appear in both lists, possibly with different spelling)
    for gt in gt_ingredients:
        if gt in used_gt:
            continue
        for ai in ai_ingredients:
            if ai in used_ai:
                continue
            if ingredients_match_fuzzy(gt, ai):
                merged.append({
                    'name': gt,  # Use GT spelling for display
                    'gt_name': gt,
                    'ai_name': ai,
                    'source': 'both',
                    'fuzzy_match': True
                })
                used_gt.add(gt)
                used_ai.add(ai)
                break
    
    # Second: GT-only items
    for gt in gt_ingredients:
        if gt not in used_gt:
            merged.append({
                'name': gt,
                'gt_name': gt,
                'ai_name': None,
                'source': 'gt',
                'fuzzy_match': False
            })
            used_gt.add(gt)
    
    # Third: AI-only items
    for ai in ai_ingredients:
        if ai not in used_ai:
            merged.append({
                'name': ai,
                'gt_name': None,
                'ai_name': ai,
                'source': 'ai',
                'fuzzy_match': False
            })
            used_ai.add(ai)
    
    return merged

# AI Ingredient Extraction

def load_ingredients_from_json_cache(cache_dir, dish_id):
    """
    Load structured AI ingredient names from JSON cache files.
    These are the original API responses with items[].name fields.
    This is the CORRECT source - not parsing free-text descriptions.
    """
    if not cache_dir or not cache_dir.exists():
        return None
    
    cache_file = cache_dir / f"{dish_id}.json"
    if not cache_file.exists():
        return None
    
    try:
        with open(cache_file, 'r') as f:
            data = json.load(f)
        
        # Handle both direct format and nested 'analysis' format
        analysis = data.get('analysis', data)
        if not isinstance(analysis, dict):
            return None
        
        items_list = analysis.get('items', [])
        ingredients = []
        for item in items_list:
            if isinstance(item, dict) and item.get('name'):
                name = str(item['name']).strip()
                if name and len(name) >= 2:
                    ingredients.append(name)
        
        return ingredients if ingredients else None
    except (json.JSONDecodeError, IOError, KeyError):
        return None


def extract_ingredients_from_description(description):
    """Extract ingredient names from AI description text."""
    if not description:
        return []
    
    stopwords = {
        'a', 'an', 'the', 'of', 'with', 'and', 'on', 'in', 'to', 'for',
        'plate', 'bowl', 'dish', 'serving', 'portion', 'small', 'large',
        'medium', 'approximately', 'about', 'some', 'several', 'few',
        'pieces', 'piece', 'slices', 'slice', 'cubes', 'cube', 'chunks',
        'white', 'appears', 'consisting', 'containing', 'topped', 'mixed',
        'assorted', 'various', 'total', 'cooked', 'raw', 'fresh', 'diced',
        'chopped', 'sliced', 'roasted', 'grilled', 'fried', 'baked',
        'steamed', 'sauteed', 'boiled'
    }
    
    desc = description.lower()
    desc = re.sub(r'^(a |an |the )?(plate|bowl|dish|serving|small plate|small bowl) (of |with |containing )?', '', desc)
    
    ingredients = []
    parts = re.split(r',\s*|\s+and\s+|\s+with\s+', desc)
    
    for part in parts:
        part = part.strip()
        part = re.sub(r'[.,!?;:]$', '', part)
        part = re.sub(r'^\d+[-–]\d+\s*', '', part)
        part = re.sub(r'^\d+\s*', '', part)
        
        words = part.split()
        while words and words[0].lower() in stopwords:
            words.pop(0)
        while words and words[-1].lower() in stopwords:
            words.pop()
        
        part = ' '.join(words)
        
        if len(part) < 3 or len(part) > 50:
            continue
        if part.lower() in stopwords:
            continue
        if part:
            ingredients.append(part)
    
    seen = set()
    unique = []
    for ing in ingredients:
        if ing.lower() not in seen:
            seen.add(ing.lower())
            unique.append(ing)
    
    return unique

# Dataset Loading

# Cache for loaded datasets
DATASET_CACHE = {}

def load_dataset(dataset_key):
    """Load a dataset from its CSV, using JSON cache for AI ingredients when available."""
    if dataset_key in DATASET_CACHE:
        return DATASET_CACHE[dataset_key]
    
    config = DATASETS.get(dataset_key)
    if not config:
        return []
    
    csv_path = config['csv_path']
    if not csv_path.exists():
        print(f"Warning: CSV not found for {dataset_key}: {csv_path}")
        return []
    
    images_dir = config['images_dir']
    json_cache_dir = config.get('json_cache_dir')
    dataset = []
    
    # Track cache hits for logging
    cache_hits = 0
    cache_misses = 0
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = row[config['id_column']]
            
            # Build image path
            image_pattern = config['image_pattern'].format(
                dish_id=image_id, image_id=image_id, file_name=row.get('file_name', image_id)
            )
            image_path = images_dir / image_pattern
            
            if not image_path.exists():
                continue
            
            # Parse GT ingredients
            gt_raw = row.get(config['gt_column'], '')
            gt_ingredients = [i.strip() for i in gt_raw.split(config['gt_separator']) if i.strip()]
            
            # Get AI ingredients - prefer JSON cache (structured data) over description parsing
            ai_raw = row.get(config['ai_column'], '')
            ai_ingredients = None
            ai_description = ai_raw
            
            # Try JSON cache first (this has the clean structured item names)
            if json_cache_dir:
                ai_ingredients = load_ingredients_from_json_cache(json_cache_dir, image_id)
                if ai_ingredients:
                    cache_hits += 1
                else:
                    cache_misses += 1
            
            # Fallback to parsing/extraction if no cache
            if ai_ingredients is None:
                if config.get('ai_is_description'):
                    ai_ingredients = extract_ingredients_from_description(ai_raw)
                else:
                    ai_ingredients = [i.strip() for i in ai_raw.split(config['gt_separator']) if i.strip()]
                    ai_description = f"Predicted: {ai_raw}"
            
            # Get original Jaccard (None if missing/empty, to exclude from mean calculation)
            jaccard_str = row.get(config['jaccard_column'], '').strip()
            try:
                original_jaccard = float(jaccard_str) if jaccard_str else None
            except (ValueError, TypeError):
                original_jaccard = None
            
            dataset.append({
                'image_id': str(image_id),
                'image_path': str(image_pattern),
                'gt_ingredients': gt_ingredients,
                'ai_ingredients': ai_ingredients,
                'ai_description': ai_description,
                'original_jaccard': original_jaccard
            })
    
    if json_cache_dir:
        print(f"[{dataset_key}] JSON cache: {cache_hits} hits, {cache_misses} misses")
    
    DATASET_CACHE[dataset_key] = dataset
    return dataset

def calculate_fuzzy_jaccard(gt_set, ai_set):
    """Calculate Jaccard with fuzzy matching."""
    if not gt_set and not ai_set:
        return 1.0
    if not gt_set or not ai_set:
        return 0.0
    
    gt_list = list(gt_set)
    ai_list = list(ai_set)
    
    matched_ai = set()
    matched_gt = set()
    
    for ai_ing in ai_list:
        for gt_ing in gt_list:
            if gt_ing not in matched_gt and ingredients_match_fuzzy(ai_ing, gt_ing):
                matched_ai.add(ai_ing)
                matched_gt.add(gt_ing)
                break
    
    intersection = len(matched_ai)
    union = len(gt_list) + len(ai_list) - intersection
    
    return intersection / union if union > 0 else 1.0

# API Routes

@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')

@app.route('/api/datasets')
def get_datasets():
    """Get available datasets."""
    available = []
    for key, config in DATASETS.items():
        csv_exists = config['csv_path'].exists()
        images_dir_exists = config['images_dir'].exists()
        
        if csv_exists:
            dataset = load_dataset(key)
            available.append({
                'key': key,
                'name': config['name'],
                'total_images': len(dataset),
                'csv_found': True,
                'images_dir_found': images_dir_exists
            })
        else:
            # Still show the dataset but with 0 images so user knows it's configured
            available.append({
                'key': key,
                'name': config['name'],
                'total_images': 0,
                'csv_found': False,
                'images_dir_found': images_dir_exists
            })
    return jsonify(available)

@app.route('/api/config')
def get_config():
    """Get app configuration including blind mode."""
    return jsonify({
        'blind_mode': BLIND_MODE,
        'single_blind_mode': SINGLE_BLIND_MODE
    })

@app.route('/api/stats/<dataset_key>')
def get_stats(dataset_key):
    """Get statistics for a dataset, including main/challenge set counts."""
    dataset = load_dataset(dataset_key)
    if not dataset:
        return jsonify({'error': 'Dataset not found'}), 404
    
    annotations = load_annotations()
    
    # Cutoff date: only include annotations from 2026-02-06 onwards
    CUTOFF_DATE = '2026-02-06'
    
    # Separate dishes into perfect and imperfect fuzzy Jaccard
    perfect_jaccard = []
    needs_annotation = []
    for d in dataset:
        fuzzy_j = calculate_fuzzy_jaccard(set(d['gt_ingredients']), set(d['ai_ingredients']))
        d['fuzzy_jaccard'] = fuzzy_j
        if fuzzy_j >= 0.999:
            perfect_jaccard.append(d)
        else:
            needs_annotation.append(d)
    
    # Count completed annotations and separate main vs challenge sets
    # Challenge set = completed images with any unsure_ingredients
    all_completed_ids = set()
    challenge_ids = set()  # Images with unsure items
    
    for key, ann in annotations.items():
        if ann['dataset'] == dataset_key and ann['completed']:
            # Filter by cutoff date
            timestamp = ann.get('timestamp', '')
            if timestamp and timestamp < CUTOFF_DATE:
                continue  # Skip old annotations
            
            all_completed_ids.add(ann['image_id'])
            has_unsure = len(ann.get('gt_unsure', [])) > 0 or len(ann.get('ai_unsure', [])) > 0
            if has_unsure:
                challenge_ids.add(ann['image_id'])
    
    # All completed annotations (from needs_annotation set, not perfect jaccard)
    all_annotated = len([d for d in needs_annotation if d['image_id'] in all_completed_ids])
    main_remaining = len(needs_annotation) - all_annotated
    
    # Validated = perfect Jaccard + all annotated
    validated_count = len(perfect_jaccard) + all_annotated
    
    # Calculate Jaccard means over ENTIRE dataset (not just imperfect ones)
    original_jaccards = [d['original_jaccard'] for d in dataset if d['original_jaccard'] is not None]
    original_mean = sum(original_jaccards) / len(original_jaccards) if original_jaccards else 0
    
    # Updated mean (include all completed annotations)
    updated_jaccards = []
    for d in dataset:
        key = f"{dataset_key}:{d['image_id']}"
        if d.get('fuzzy_jaccard', 0) >= 0.999:
            updated_jaccards.append(1.0)
        elif key in annotations and annotations[key]['completed']:
            # Filter by cutoff date
            timestamp = annotations[key].get('timestamp', '')
            if timestamp and timestamp >= CUTOFF_DATE:
                updated_jaccards.append(annotations[key]['updated_jaccard'])
            elif d['original_jaccard'] is not None:
                updated_jaccards.append(d['original_jaccard'])
        elif d['original_jaccard'] is not None:
            updated_jaccards.append(d['original_jaccard'])
    
    updated_mean = sum(updated_jaccards) / len(updated_jaccards) if updated_jaccards else 0
    
    return jsonify({
        'total_images': len(dataset),
        'validated_count': validated_count,
        'perfect_jaccard_count': len(perfect_jaccard),
        'annotated_count': all_annotated,
        'remaining': main_remaining,
        'challenge_count': len(challenge_ids),
        'original_jaccard_mean': round(original_mean, 4),
        'updated_jaccard_mean': round(updated_mean, 4)
    })

@app.route('/api/next/<dataset_key>')
def get_next_image(dataset_key):
    """Get the next image to annotate. Supports ?set=main or ?set=challenge."""
    dataset = load_dataset(dataset_key)
    if not dataset:
        return jsonify({'error': 'Dataset not found'}), 404
    
    # Which set to pull from: main (default) or challenge
    image_set = request.args.get('set', 'main')
    
    annotations = load_annotations()
    
    # Categorize completed images by set
    main_completed_ids = set()
    challenge_ids = set()  # Images with unsure items
    
    for key, ann in annotations.items():
        if ann['dataset'] == dataset_key and ann['completed']:
            has_unsure = len(ann.get('gt_unsure', [])) > 0 or len(ann.get('ai_unsure', [])) > 0
            if has_unsure:
                challenge_ids.add(ann['image_id'])
            else:
                main_completed_ids.add(ann['image_id'])
    
    # Filter available images based on set
    available = []
    
    if image_set == 'challenge':
        # Challenge set: only images that have been annotated with unsure items
        for d in dataset:
            if d['image_id'] in challenge_ids:
                available.append(d)
    else:
        # Main set: images not yet completed or not in challenge set
        for d in dataset:
            # Skip if already in main completed
            if d['image_id'] in main_completed_ids:
                continue
            # Skip if in challenge set (has unsure items)
            if d['image_id'] in challenge_ids:
                continue
            # Skip perfect Jaccard
            fuzzy_j = calculate_fuzzy_jaccard(set(d['gt_ingredients']), set(d['ai_ingredients']))
            if fuzzy_j >= 0.999:
                continue
            
            available.append(d)
    
    if not available:
        return jsonify({'error': 'No more images', 'done': True, 'set': image_set})
    
    # Pick random image
    dish = random.choice(available)
    
    # Get auto-matches
    auto_matches = get_auto_matches(dish['gt_ingredients'], dish['ai_ingredients'])
    
    # Get existing annotation if any
    key = f"{dataset_key}:{dish['image_id']}"
    existing = annotations.get(key)
    
    # Get global equivalences
    global_equiv = load_equivalences()
    
    # In blind mode, randomize which column is A vs B
    # column_a_is_gt = True means column A shows GT, column B shows AI
    # column_a_is_gt = False means column A shows AI, column B shows GT
    column_a_is_gt = True
    if BLIND_MODE:
        column_a_is_gt = random.choice([True, False])
    
    response = {
        'dataset': dataset_key,
        'image_id': dish['image_id'],
        'image_url': f'/images/{dataset_key}/{dish["image_path"]}',
        'gt_ingredients': dish['gt_ingredients'],
        'ai_ingredients': dish['ai_ingredients'],
        'ai_description': dish['ai_description'],
        'original_jaccard': dish['original_jaccard'],
        'auto_matches': auto_matches,
        'global_equivalences': global_equiv,
        'is_challenged': dish['image_id'] in challenge_ids,
        'blind_mode': BLIND_MODE,
        'single_blind_mode': SINGLE_BLIND_MODE,
        'column_a_is_gt': column_a_is_gt  # For blind mode decoding
    }
    
    # For single-blind mode, include the merged ingredient list (shuffled)
    if SINGLE_BLIND_MODE:
        merged = create_merged_ingredient_list(
            dish['gt_ingredients'], 
            dish['ai_ingredients']
        )
        random.shuffle(merged)  # Shuffle to prevent source inference from position
        response['merged_ingredients'] = merged
    
    if existing:
        response['existing_annotation'] = {
            'gt_verified': existing['gt_verified'],
            'ai_verified': existing['ai_verified'],
            'equivalences': existing['equivalences'],
            'user_added': existing['user_added'],
            'challenge_flag': existing['challenge_flag'],
            'column_a_is_gt': existing.get('column_a_is_gt', True),  # Preserve column assignment
            # Single-blind mode fields (source-separated)
            'phase': existing.get('phase', 'complete'),
            'gt_approved': existing.get('gt_approved', []),
            'gt_rejected': existing.get('gt_rejected', []),
            'gt_unsure': existing.get('gt_unsure', []),
            'ai_approved': existing.get('ai_approved', []),
            'ai_rejected': existing.get('ai_rejected', []),
            'ai_unsure': existing.get('ai_unsure', []),
            'phase2_equivalences': existing.get('phase2_equivalences', {}),
        }
        # In blind mode with existing annotation, use the same column assignment
        if BLIND_MODE and 'column_a_is_gt' in existing:
            response['column_a_is_gt'] = existing['column_a_is_gt']
    
    return jsonify(response)

@app.route('/api/annotate', methods=['POST'])
def save_annotation():
    """Save an annotation."""
    data = request.json
    dataset_key = data.get('dataset')
    image_id = data.get('image_id')
    
    if not dataset_key or not image_id:
        return jsonify({'error': 'Missing dataset or image_id'}), 400
    
    annotations = load_annotations()
    key = f"{dataset_key}:{image_id}"
    
    # Get original Jaccard
    dataset = load_dataset(dataset_key)
    dish = next((d for d in dataset if d['image_id'] == image_id), None)
    original_jaccard = dish['original_jaccard'] if dish and dish['original_jaccard'] is not None else 0
    
    # Check if this is a single-blind mode submission
    phase = data.get('phase', 'complete')
    approved_ingredients = data.get('approved_ingredients', [])
    phase2_equivalences = data.get('phase2_equivalences', {})
    
    # Blind mode data
    column_a_is_gt = data.get('column_a_is_gt', True)
    col_a_rejected = data.get('col_a_rejected', [])
    col_b_rejected = data.get('col_b_rejected', [])
    
    # Calculate updated Jaccard
    gt_verified = data.get('gt_verified', {})
    ai_verified = data.get('ai_verified', {})
    equivalences = data.get('equivalences', {})
    user_added = data.get('user_added', [])
    challenge_flag = data.get('challenge_flag', False)
    
    # For single-blind mode, Phase 1 now completes the image directly
    # (global Phase 2 equivalency review happens after ALL images are done)
    if phase == 'phase1':
        # Source-separated ingredient decisions
        gt_approved = data.get('gt_approved', [])
        gt_rejected = data.get('gt_rejected', [])
        gt_unsure = data.get('gt_unsure', [])
        ai_approved = data.get('ai_approved', [])
        ai_rejected = data.get('ai_rejected', [])
        ai_unsure = data.get('ai_unsure', [])
        
        # Calculate preliminary Jaccard using fuzzy matching only (no equivalencies yet)
        # Effective GT = approved GT items + user-added items
        effective_gt = set(gt_approved) | set(user_added)
        
        # All original AI items stay in effective_ai
        effective_ai = set(dish['ai_ingredients']) if dish else set()
        
        # If an AI item was approved and doesn't fuzzy-match any GT → GT was incomplete
        for ai_ing in ai_approved:
            matches_gt = any(ingredients_match_fuzzy(ai_ing, gt) for gt in effective_gt)
            if not matches_gt:
                effective_gt.add(ai_ing)
        
        updated_jaccard = calculate_fuzzy_jaccard(effective_gt, effective_ai)
        
        # Determine if this goes to challenge set (any unsure items)
        has_unsure = len(gt_unsure) > 0 or len(ai_unsure) > 0
        
        annotations[key] = {
            'dataset': dataset_key,
            'image_id': image_id,
            'annotator': data.get('annotator', ''),
            'gt_verified': {},
            'ai_verified': {},
            'equivalences': {},
            'user_added': user_added,
            'challenge_flag': has_unsure,
            'completed': True,  # Always mark as completed so it shows in dashboard/export
            'original_jaccard': original_jaccard,
            'updated_jaccard': updated_jaccard,
            'timestamp': datetime.now().isoformat(),
            'column_a_is_gt': True,
            'col_a_rejected': [],
            'col_b_rejected': [],
            'phase': 'complete',
            'gt_approved': gt_approved,
            'gt_rejected': gt_rejected,
            'gt_unsure': gt_unsure,
            'ai_approved': ai_approved,
            'ai_rejected': ai_rejected,
            'ai_unsure': ai_unsure,
            'phase2_equivalences': {},
        }
        save_annotations(annotations)
        
        return jsonify({
            'success': True,
            'phase': 'complete',
            'original_jaccard': round(original_jaccard, 4),
            'updated_jaccard': round(updated_jaccard, 4),
            'gt_approved_count': len(gt_approved),
            'ai_approved_count': len(ai_approved),
            'user_added_count': len(user_added)
        })
    
    # Normal mode (or direct completion)
    # Build effective GT and AI sets
    effective_gt = set()
    if dish:
        for ing in dish['gt_ingredients']:
            if gt_verified.get(ing) != 'rejected':
                effective_gt.add(ing)
    effective_gt.update(user_added)
    
    effective_ai = set()
    if dish:
        for ing in dish['ai_ingredients']:
            # ALL AI items stay in effective_ai (even rejected ones)
            # Rejecting confirms the model was wrong - it's still a false positive
            effective_ai.add(ing)
            
            # If AI ingredient is VERIFIED and doesn't match any GT,
            # add it to effective GT (the AI was right, GT was incomplete)
            if ai_verified.get(ing) == 'verified':
                # Check if it matches any existing GT via fuzzy matching
                matches_gt = any(ingredients_match_fuzzy(ing, gt) for gt in effective_gt)
                # Check if it has an equivalence mapping
                has_equiv = ing in equivalences and len(equivalences[ing]) > 0
                if not matches_gt and not has_equiv:
                    effective_gt.add(ing)
    
    updated_jaccard = calculate_fuzzy_jaccard(effective_gt, effective_ai)
    
    # Save annotation
    annotations[key] = {
        'dataset': dataset_key,
        'image_id': image_id,
        'annotator': data.get('annotator', ''),
        'gt_verified': gt_verified,
        'ai_verified': ai_verified,
        'equivalences': equivalences,
        'user_added': user_added,
        'challenge_flag': challenge_flag,
        'completed': not challenge_flag,
        'original_jaccard': original_jaccard,
        'updated_jaccard': updated_jaccard,
        'timestamp': datetime.now().isoformat(),
        # Blind mode fields
        'column_a_is_gt': column_a_is_gt,
        'col_a_rejected': col_a_rejected,
        'col_b_rejected': col_b_rejected,
        # Single-blind mode fields (source-separated)
        'phase': 'complete',
        'gt_approved': [],
        'gt_rejected': [],
        'gt_unsure': [],
        'ai_approved': [],
        'ai_rejected': [],
        'ai_unsure': [],
        'phase2_equivalences': {},
    }
    
    save_annotations(annotations)
    
    # Update global equivalences
    for ai_ing, gt_ings in equivalences.items():
        for gt_ing in gt_ings:
            add_equivalence(ai_ing, gt_ing)
    
    return jsonify({
        'success': True,
        'original_jaccard': round(original_jaccard, 4),
        'updated_jaccard': round(updated_jaccard, 4)
    })

@app.route('/api/equivalency-candidates/<dataset_key>')
def get_equivalency_candidates(dataset_key):
    """
    Get potential equivalency candidates aggregated across ALL completed images.
    Used in global Phase 2 after all per-image Phase 1 annotations are done.
    
    Algorithm:
    1. For each completed image, find approved GT-only and AI-only items
    2. Generate candidate pairs from unmatched items co-occurring on same image
    3. Aggregate across images with count
    4. Return sorted by count (most frequent = most likely real synonyms)
    """
    dataset = load_dataset(dataset_key)
    if not dataset:
        return jsonify({'error': 'Dataset not found'}), 404
    
    annotations = load_annotations()
    
    # Existing global equivalences (already confirmed) - skip these
    global_equivs = load_equivalences()
    confirmed_pairs = set()
    for ai_ing, gt_list in global_equivs.items():
        for item in gt_list:
            pair = tuple(sorted([normalize_ingredient(ai_ing), normalize_ingredient(item['gt'])]))
            confirmed_pairs.add(pair)
    
    # Collect candidate pairs across all completed images
    pair_data = {}  # (norm_a, norm_b) -> {term_a, term_b, count, images}
    total_completed = 0
    
    for dish in dataset:
        key = f"{dataset_key}:{dish['image_id']}"
        ann = annotations.get(key)
        if not ann or not ann.get('completed'):
            continue
        
        total_completed += 1
        
        # Get source-separated approved items directly from annotation
        gt_approved = ann.get('gt_approved', [])
        ai_approved = ann.get('ai_approved', [])
        
        if not gt_approved and not ai_approved:
            continue
        
        # Find auto fuzzy matches between approved GT and AI
        auto_matches = get_auto_matches(gt_approved, ai_approved)
        matched_gt = set(auto_matches.values())
        matched_ai = set(auto_matches.keys())
        
        # Unmatched approved items from each source
        unmatched_gt = [g for g in gt_approved if g not in matched_gt]
        unmatched_ai = [a for a in ai_approved if a not in matched_ai]
        
        # Generate candidate pairs from co-occurring unmatched items
        for ai_ing in unmatched_ai:
            for gt_ing in unmatched_gt:
                norm_pair = tuple(sorted([normalize_ingredient(ai_ing), normalize_ingredient(gt_ing)]))
                
                # Skip already-confirmed equivalences
                if norm_pair in confirmed_pairs:
                    continue
                
                if norm_pair not in pair_data:
                    pair_data[norm_pair] = {
                        'term_a': gt_ing,
                        'term_b': ai_ing,
                        'count': 0,
                        'images': []
                    }
                pair_data[norm_pair]['count'] += 1
                pair_data[norm_pair]['images'].append(dish['image_id'])
    
    # Sort by count descending (most frequent pairs first)
    candidates = sorted(pair_data.values(), key=lambda x: -x['count'])
    
    return jsonify({
        'candidates': candidates,
        'total_completed': total_completed,
        'total_images': len(dataset)
    })


@app.route('/api/save-equivalencies', methods=['POST'])
def save_global_equivalencies():
    """
    Save global equivalency decisions from Phase 2 and recalculate Jaccards.
    
    Receives: {dataset, decisions: [{term_a, term_b, equivalent: bool}]}
    For each confirmed equivalent pair, adds bidirectional mappings to equivalences.csv.
    Then recalculates updated_jaccard for all completed images.
    """
    data = request.json
    dataset_key = data.get('dataset')
    decisions = data.get('decisions', [])
    
    if not dataset_key:
        return jsonify({'error': 'Missing dataset'}), 400
    
    # Save confirmed equivalencies (bidirectional)
    equiv_count = 0
    for decision in decisions:
        if decision.get('equivalent'):
            add_equivalence(decision['term_a'], decision['term_b'])
            add_equivalence(decision['term_b'], decision['term_a'])
            equiv_count += 1
    
    # Recalculate Jaccards for all completed images using new equivalencies
    dataset = load_dataset(dataset_key)
    annotations = load_annotations()
    global_equivs = load_equivalences()
    
    updated_count = 0
    for dish in dataset:
        key = f"{dataset_key}:{dish['image_id']}"
        ann = annotations.get(key)
        if not ann or not ann.get('completed'):
            continue
        
        # Use source-separated approved items
        gt_approved = set(ann.get('gt_approved', []))
        ai_approved = set(ann.get('ai_approved', []))
        
        # Build effective GT = approved GT items + user-added
        effective_gt = gt_approved | set(ann.get('user_added', []))
        
        # All AI items remain in effective AI
        effective_ai = set(dish['ai_ingredients'])
        
        # Approved AI items not matching GT → add to effective GT
        for ai_ing in ai_approved:
            matches_gt = any(ingredients_match_fuzzy(ai_ing, gt) for gt in effective_gt)
            # Also check global equivalencies
            has_equiv = False
            for eq_key, eq_list in global_equivs.items():
                if ingredients_match_fuzzy(ai_ing, eq_key):
                    for eq_item in eq_list:
                        if any(ingredients_match_fuzzy(eq_item['gt'], gt) for gt in effective_gt):
                            has_equiv = True
                            break
                if has_equiv:
                    break
            if not matches_gt and not has_equiv:
                effective_gt.add(ai_ing)
        
        new_jaccard = calculate_fuzzy_jaccard(effective_gt, effective_ai)
        ann['updated_jaccard'] = new_jaccard
        updated_count += 1
    
    save_annotations(annotations)
    
    return jsonify({
        'success': True,
        'updated_count': updated_count,
        'equivalencies_saved': equiv_count
    })


@app.route('/api/annotators/<dataset_key>')
def get_annotators(dataset_key):
    """Get list of unique annotators for a dataset."""
    annotations = load_annotations()
    
    # Cutoff date
    CUTOFF_DATE = '2026-02-06'
    
    annotators = set()
    for key, ann in annotations.items():
        if ann['dataset'] == dataset_key and ann.get('completed'):
            timestamp = ann.get('timestamp', '')
            if timestamp and timestamp < CUTOFF_DATE:
                continue
            annotator = ann.get('annotator', '')
            if annotator:
                annotators.add(annotator)
    
    return jsonify(sorted(list(annotators)))


@app.route('/api/dashboard/<dataset_key>')
def get_dashboard_stats(dataset_key):
    """
    Get statistics for the dashboard.
    Returns JSON with precision metrics and counts.
    Query params:
      - exclude_perfect=1: exclude auto-perfect images from ingredient stats
      - annotator: filter by annotator name (optional)
    """
    dataset = load_dataset(dataset_key)
    if not dataset:
        return jsonify({'error': 'Dataset not found'}), 404
    
    exclude_perfect = request.args.get('exclude_perfect', '0') == '1'
    annotator_filter = request.args.get('annotator', '')  # Empty means all
    annotations = load_annotations()
    
    # Cutoff date: only include annotations from 2026-02-06 onwards
    CUTOFF_DATE = '2026-02-06'
    
    # Statistics accumulators
    total_gt_ingredients = 0
    total_ai_ingredients = 0
    gt_approved_count = 0
    gt_rejected_count = 0
    gt_unsure_count = 0
    ai_approved_count = 0
    ai_rejected_count = 0
    ai_unsure_count = 0
    manual_count = 0
    auto_perfect_count = 0
    
    # Per-image data for charts
    image_stats = []
    
    for dish in dataset:
        image_id = dish['image_id']
        key = f"{dataset_key}:{image_id}"
        
        # Check if auto-perfect
        fuzzy_j = calculate_fuzzy_jaccard(set(dish['gt_ingredients']), set(dish['ai_ingredients']))
        is_auto_perfect = fuzzy_j >= 0.999
        
        ann = annotations.get(key)
        
        if is_auto_perfect:
            auto_perfect_count += 1
            # For auto-perfect, all ingredients are approved (only count if not excluding)
            if not exclude_perfect:
                total_gt_ingredients += len(dish['gt_ingredients'])
                total_ai_ingredients += len(dish['ai_ingredients'])
                gt_approved_count += len(dish['gt_ingredients'])
                ai_approved_count += len(dish['ai_ingredients'])
            
        elif ann and ann.get('completed'):
            # Filter by timestamp
            timestamp = ann.get('timestamp', '')
            if timestamp and timestamp < CUTOFF_DATE:
                continue
            
            # Filter by annotator if specified
            if annotator_filter and ann.get('annotator', '') != annotator_filter:
                continue
            
            gt_app = ann.get('gt_approved', [])
            gt_rej = ann.get('gt_rejected', [])
            gt_uns = ann.get('gt_unsure', [])
            ai_app = ann.get('ai_approved', [])
            ai_rej = ann.get('ai_rejected', [])
            ai_uns = ann.get('ai_unsure', [])
            
            manual_count += 1
            
            # Accumulate statistics
            total_gt_ingredients += len(gt_app) + len(gt_rej) + len(gt_uns)
            total_ai_ingredients += len(ai_app) + len(ai_rej) + len(ai_uns)
            gt_approved_count += len(gt_app)
            gt_rejected_count += len(gt_rej)
            gt_unsure_count += len(gt_uns)
            ai_approved_count += len(ai_app)
            ai_rejected_count += len(ai_rej)
            ai_unsure_count += len(ai_uns)
            
            # Per-image stats
            image_stats.append({
                'image_id': image_id,
                'original_jaccard': ann.get('original_jaccard', 0),
                'updated_jaccard': ann.get('updated_jaccard', 0),
                'gt_total': len(gt_app) + len(gt_rej) + len(gt_uns),
                'gt_approved': len(gt_app),
                'ai_total': len(ai_app) + len(ai_rej) + len(ai_uns),
                'ai_approved': len(ai_app)
            })
    
    # Calculate precision metrics
    # Total for each source
    gt_total = gt_approved_count + gt_rejected_count + gt_unsure_count
    ai_total = ai_approved_count + ai_rejected_count + ai_unsure_count
    
    # Strict precision = approved / (approved + rejected + unsure) - unsure treated as wrong
    gt_precision_strict = gt_approved_count / gt_total if gt_total > 0 else 0
    ai_precision_strict = ai_approved_count / ai_total if ai_total > 0 else 0
    
    # Precision with uncertain = (approved + unsure) / (approved + rejected + unsure) - unsure treated as correct
    gt_precision_with_uncertain = (gt_approved_count + gt_unsure_count) / gt_total if gt_total > 0 else 0
    ai_precision_with_uncertain = (ai_approved_count + ai_unsure_count) / ai_total if ai_total > 0 else 0
    
    return jsonify({
        'dataset': dataset_key,
        'cutoff_date': CUTOFF_DATE,
        'counts': {
            'manual': manual_count,
            'auto_perfect': auto_perfect_count,
            'total': manual_count + auto_perfect_count,
            'total_images_in_dataset': len(dataset)
        },
        'gt': {
            'total_ingredients': total_gt_ingredients,
            'approved': gt_approved_count,
            'rejected': gt_rejected_count,
            'unsure': gt_unsure_count,
            'precision_strict': round(gt_precision_strict, 4),
            'precision_with_uncertain': round(gt_precision_with_uncertain, 4)
        },
        'ai': {
            'total_ingredients': total_ai_ingredients,
            'approved': ai_approved_count,
            'rejected': ai_rejected_count,
            'unsure': ai_unsure_count,
            'precision_strict': round(ai_precision_strict, 4),
            'precision_with_uncertain': round(ai_precision_with_uncertain, 4)
        },
        'image_stats': image_stats
    })


@app.route('/api/export/<dataset_key>')
def export_annotations(dataset_key):
    """
    Export annotations as Excel file with two sheets:
    1. Annotations - detailed per-image data
    2. Summary - aggregate statistics including precision metrics
    
    Query params:
      - include_perfect=1 (default): include auto-perfect Jaccard images
      - include_perfect=0: only manually annotated images
      - format=csv: return CSV instead of Excel (for backwards compatibility)
    """
    import io
    from datetime import datetime as dt
    
    dataset = load_dataset(dataset_key)
    if not dataset:
        return jsonify({'error': 'Dataset not found'}), 404
    
    include_perfect = request.args.get('include_perfect', '1') == '1'
    export_format = request.args.get('format', 'xlsx')
    annotations = load_annotations()
    
    # Cutoff date: only include annotations from 2026-02-06 onwards
    CUTOFF_DATE = '2026-02-06'
    
    # Build export data
    rows = []
    
    # Statistics accumulators
    total_gt_ingredients = 0
    total_ai_ingredients = 0
    gt_approved_count = 0
    gt_rejected_count = 0
    gt_unsure_count = 0
    ai_approved_count = 0
    ai_rejected_count = 0
    ai_unsure_count = 0
    manual_count = 0
    auto_perfect_count = 0
    
    for dish in dataset:
        image_id = dish['image_id']
        key = f"{dataset_key}:{image_id}"
        
        # Check if auto-perfect
        fuzzy_j = calculate_fuzzy_jaccard(set(dish['gt_ingredients']), set(dish['ai_ingredients']))
        is_auto_perfect = fuzzy_j >= 0.999
        
        ann = annotations.get(key)
        
        if is_auto_perfect:
            if not include_perfect:
                continue
            # Auto-perfect: create a synthetic annotation (no timestamp filter for auto-perfect)
            rows.append({
                'image_id': image_id,
                'annotator': '',  # Auto-perfect has no annotator
                'auto_perfect': True,
                'gt_ingredients': '|'.join(dish['gt_ingredients']),
                'ai_ingredients': '|'.join(dish['ai_ingredients']),
                'gt_approved': '|'.join(dish['gt_ingredients']),
                'gt_rejected': '',
                'gt_unsure': '',
                'ai_approved': '|'.join(dish['ai_ingredients']),
                'ai_rejected': '',
                'ai_unsure': '',
                'user_added': '',
                'original_jaccard': dish['original_jaccard'] or 1.0,
                'updated_jaccard': 1.0,
                'original_precision': 1.0,  # Auto-perfect = 100% precision
                'updated_precision': 1.0,   # Auto-perfect = 100% precision
                'completed': True,
                'timestamp': ''
            })
            auto_perfect_count += 1
            # For auto-perfect, all ingredients are approved
            total_gt_ingredients += len(dish['gt_ingredients'])
            total_ai_ingredients += len(dish['ai_ingredients'])
            gt_approved_count += len(dish['gt_ingredients'])
            ai_approved_count += len(dish['ai_ingredients'])
            
        elif ann and ann.get('completed'):
            # Filter by timestamp - only include annotations from cutoff date onwards
            timestamp = ann.get('timestamp', '')
            if timestamp and timestamp < CUTOFF_DATE:
                continue  # Skip old annotations
            
            # Manually annotated
            gt_app = ann.get('gt_approved', [])
            gt_rej = ann.get('gt_rejected', [])
            gt_uns = ann.get('gt_unsure', [])
            ai_app = ann.get('ai_approved', [])
            ai_rej = ann.get('ai_rejected', [])
            ai_uns = ann.get('ai_unsure', [])
            
            # Calculate per-image precision (matches dashboard: unsure counted in denominator)
            gt_total_decisions = len(gt_app) + len(gt_rej) + len(gt_uns)
            ai_total_decisions = len(ai_app) + len(ai_rej) + len(ai_uns)
            original_precision = len(gt_app) / gt_total_decisions if gt_total_decisions > 0 else None
            updated_precision = len(ai_app) / ai_total_decisions if ai_total_decisions > 0 else None
            
            rows.append({
                'image_id': image_id,
                'annotator': ann.get('annotator', ''),
                'auto_perfect': False,
                'gt_ingredients': '|'.join(dish['gt_ingredients']),
                'ai_ingredients': '|'.join(dish['ai_ingredients']),
                'gt_approved': '|'.join(gt_app),
                'gt_rejected': '|'.join(gt_rej),
                'gt_unsure': '|'.join(gt_uns),
                'ai_approved': '|'.join(ai_app),
                'ai_rejected': '|'.join(ai_rej),
                'ai_unsure': '|'.join(ai_uns),
                'user_added': '|'.join(ann.get('user_added', [])),
                'original_jaccard': ann.get('original_jaccard', 0),
                'updated_jaccard': ann.get('updated_jaccard', 0),
                'original_precision': original_precision,
                'updated_precision': updated_precision,
                'completed': True,
                'timestamp': timestamp
            })
            manual_count += 1
            
            # Accumulate statistics
            total_gt_ingredients += len(gt_app) + len(gt_rej) + len(gt_uns)
            total_ai_ingredients += len(ai_app) + len(ai_rej) + len(ai_uns)
            gt_approved_count += len(gt_app)
            gt_rejected_count += len(gt_rej)
            gt_unsure_count += len(gt_uns)
            ai_approved_count += len(ai_app)
            ai_rejected_count += len(ai_rej)
            ai_unsure_count += len(ai_uns)
    
    if not rows:
        return jsonify({'error': 'No annotations to export'}), 404
    
    # Sort rows by timestamp (A to Z = oldest first)
    # Auto-perfect images (empty timestamp) go at the end
    rows.sort(key=lambda x: x.get('timestamp') or 'zzzz')
    
    # Calculate precision metrics (matches dashboard: unsure counted in denominator)
    gt_total = gt_approved_count + gt_rejected_count + gt_unsure_count
    ai_total = ai_approved_count + ai_rejected_count + ai_unsure_count
    gt_precision = gt_approved_count / gt_total if gt_total > 0 else 0
    ai_precision = ai_approved_count / ai_total if ai_total > 0 else 0
    
    # Build summary statistics
    summary = {
        'Metric': [
            'Completed Photos (manual)',
            'Auto-Perfect Photos',
            'Total Completed Photos',
            '',
            'N5K GT Total Ingredients',
            'N5K GT Approved',
            'N5K GT Rejected',
            'N5K GT Unsure',
            'N5K GT Precision',
            '',
            'AI Total Ingredients',
            'AI Approved',
            'AI Rejected', 
            'AI Unsure',
            'AI Precision',
            '',
            'Notes'
        ],
        'Value': [
            manual_count,
            auto_perfect_count,
            manual_count + auto_perfect_count,
            '',
            total_gt_ingredients,
            gt_approved_count,
            gt_rejected_count,
            gt_unsure_count,
            f'{gt_precision:.2%}',
            '',
            total_ai_ingredients,
            ai_approved_count,
            ai_rejected_count,
            ai_unsure_count,
            f'{ai_precision:.2%}',
            '',
            f'Precision = Approved / (Approved + Rejected + Unsure). Excludes auto-perfect from ingredient counts. Data from {CUTOFF_DATE} onwards.'
        ]
    }
    
    # Return CSV format if requested
    if export_format == 'csv':
        output = io.StringIO()
        fieldnames = ['image_id', 'annotator', 'auto_perfect', 'gt_ingredients', 'ai_ingredients',
                      'gt_approved', 'gt_rejected', 'gt_unsure',
                      'ai_approved', 'ai_rejected', 'ai_unsure',
                      'user_added', 'original_jaccard', 'updated_jaccard',
                      'original_precision', 'updated_precision', 'completed', 'timestamp']
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        
        from flask import Response
        csv_content = output.getvalue()
        return Response(
            csv_content,
            mimetype='text/csv',
            headers={'Content-Disposition': f'attachment; filename={dataset_key}_annotations.csv'}
        )
    
    # Generate Excel with two sheets
    try:
        from openpyxl import Workbook
        from openpyxl.utils.dataframe import dataframe_to_rows
    except ImportError:
        # Fallback to CSV if openpyxl not available
        output = io.StringIO()
        fieldnames = ['image_id', 'annotator', 'auto_perfect', 'gt_ingredients', 'ai_ingredients',
                      'gt_approved', 'gt_rejected', 'gt_unsure',
                      'ai_approved', 'ai_rejected', 'ai_unsure',
                      'user_added', 'original_jaccard', 'updated_jaccard',
                      'original_precision', 'updated_precision', 'completed', 'timestamp']
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        
        from flask import Response
        csv_content = output.getvalue()
        return Response(
            csv_content,
            mimetype='text/csv',
            headers={'Content-Disposition': f'attachment; filename={dataset_key}_annotations.csv'}
        )
    
    wb = Workbook()
    
    # Sheet 1: Summary Statistics
    ws_summary = wb.active
    ws_summary.title = "Summary"
    ws_summary.append(['Metric', 'Value'])
    for i in range(len(summary['Metric'])):
        ws_summary.append([summary['Metric'][i], summary['Value'][i]])
    
    # Auto-size columns for summary
    ws_summary.column_dimensions['A'].width = 30
    ws_summary.column_dimensions['B'].width = 80
    
    # Sheet 2: Annotations Data
    ws_data = wb.create_sheet("Annotations")
    fieldnames = ['image_id', 'annotator', 'auto_perfect', 'gt_ingredients', 'ai_ingredients',
                  'gt_approved', 'gt_rejected', 'gt_unsure',
                  'ai_approved', 'ai_rejected', 'ai_unsure',
                  'user_added', 'original_jaccard', 'updated_jaccard',
                  'original_precision', 'updated_precision', 'completed', 'timestamp']
    ws_data.append(fieldnames)
    for row in rows:
        ws_data.append([row.get(f, '') for f in fieldnames])
    
    # Save to BytesIO
    output = io.BytesIO()
    wb.save(output)
    output.seek(0)
    
    from flask import Response
    return Response(
        output.getvalue(),
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        headers={'Content-Disposition': f'attachment; filename={dataset_key}_annotations.xlsx'}
    )


@app.route('/images/<dataset_key>/<path:filepath>')
def serve_image(dataset_key, filepath):
    """Serve images."""
    config = DATASETS.get(dataset_key)
    if not config:
        return "Dataset not found", 404
    
    images_dir = config['images_dir']
    return send_from_directory(images_dir, filepath)


@app.route('/api/admin/cleanup-old-annotations', methods=['POST'])
def cleanup_old_annotations():
    """
    Remove annotations before a specified cutoff date.
    POST body: {"cutoff_date": "2026-02-06", "confirm": true}
    """
    data = request.json
    cutoff_date = data.get('cutoff_date', '2026-02-06')
    confirm = data.get('confirm', False)
    
    annotations = load_annotations()
    
    # Find annotations to remove
    to_remove = []
    to_keep = {}
    
    for key, ann in annotations.items():
        timestamp = ann.get('timestamp', '')
        if timestamp and timestamp < cutoff_date:
            to_remove.append({
                'key': key,
                'timestamp': timestamp,
                'image_id': ann.get('image_id', '')
            })
        else:
            to_keep[key] = ann
    
    if not confirm:
        # Dry run - just report what would be removed
        return jsonify({
            'dry_run': True,
            'would_remove': len(to_remove),
            'would_keep': len(to_keep),
            'annotations_to_remove': to_remove,
            'message': 'Add "confirm": true to actually delete these annotations'
        })
    
    # Actually remove
    save_annotations(to_keep)
    
    return jsonify({
        'success': True,
        'removed': len(to_remove),
        'kept': len(to_keep),
        'removed_annotations': to_remove
    })

# Main

if __name__ == '__main__':
    DATA_DIR.mkdir(exist_ok=True)
    EXPORTS_DIR.mkdir(exist_ok=True)
    
    # Get port from environment variable, default to 5050
    port = int(os.environ.get('PORT', 5050))
    
    print("=" * 50)
    print("Nutrition5k & FoodSeg103 Annotation Game")
    if SINGLE_BLIND_MODE:
        print("*** SINGLE-BLIND MODE ENABLED (Two-Phase) ***")
    elif BLIND_MODE:
        print("*** BLIND MODE ENABLED ***")
    print("=" * 50)
    
    for key, config in DATASETS.items():
        if config['csv_path'].exists():
            dataset = load_dataset(key)
            print(f"  {config['name']}: {len(dataset)} images")
        else:
            print(f"  {config['name']}: CSV not found")
    
    print()
    print(f"Starting server at http://localhost:{port}")
    app.run(debug=True, host='0.0.0.0', port=port)
