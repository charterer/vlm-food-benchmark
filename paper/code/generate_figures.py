#!/usr/bin/env python3
"""Generate paper figures from benchmark CSVs."""

import os
import glob
import random
import re
import json
import warnings
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont

warnings.filterwarnings('ignore')
random.seed(42)
np.random.seed(42)

FONT_SCALE = 1.55
FIG1_FONT_SCALE = 1.15
LINE_SCALE = 1.30
MAX_FIG_PX = 1900


def fs(size, scale=FONT_SCALE):
    return size * scale


def fs_base(size):
    return size * FIG1_FONT_SCALE


def lw(size):
    return size * LINE_SCALE


def lw_base(size):
    return size


def save_fig(fig, path, facecolor=None):
    w_in, h_in = fig.get_size_inches()
    dpi_max = (MAX_FIG_PX - 1) / max(w_in, h_in)
    dpi = min(plt.rcParams.get('savefig.dpi', 200), dpi_max)
    kwargs = {"dpi": dpi, "bbox_inches": "tight"}
    if facecolor is not None:
        kwargs["facecolor"] = facecolor
    fig.savefig(path, **kwargs)


plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': fs(10),
    'font.weight': 'normal',
    'axes.labelweight': 'normal',
    'axes.titleweight': 'normal',
    'figure.titleweight': 'normal',
    'axes.labelsize': fs(11),
    'axes.titlesize': fs(12),
    'xtick.labelsize': fs(9),
    'ytick.labelsize': fs(9),
    'legend.fontsize': fs(9),
    'lines.linewidth': lw(1.0),
    'axes.linewidth': lw(1.0),
    'grid.linewidth': lw(1.0),
    'xtick.major.width': lw(0.8),
    'ytick.major.width': lw(0.8),
    'xtick.minor.width': lw(0.6),
    'ytick.minor.width': lw(0.6),
    'patch.linewidth': lw(1.0),
    'figure.dpi': 200,
    'savefig.dpi': 450,
    'savefig.bbox': 'tight',
})


BASE_DIR = str(Path(__file__).resolve().parents[2])
PIPELINE_DIR = os.path.join(BASE_DIR, 'pipeline')
RESULTS_DIR = os.environ.get('N5K_RESULTS_DIR', os.path.join(PIPELINE_DIR, 'outputs', 'benchmark_results'))
PROMPT_ABLATION_DIR = os.path.join(RESULTS_DIR, 'prompt_ablation_full3229')
OUTPUT_DIR = os.path.join(BASE_DIR, 'paper', 'figures')
CACHE_DIR = os.path.join(PIPELINE_DIR, 'outputs', '.gemini_cache')

NUTRITION5K_DIR = os.environ.get('N5K_DATASET_DIR',
                                 os.path.join(BASE_DIR, 'datasets', 'nutrition5k', 'imagery', 'realsense_overhead'))
FOODINSSEG_DIR = os.path.join(BASE_DIR, 'datasets', 'foodinsseg', 'images')
RECIPE1M_DIR = os.path.join(BASE_DIR, 'datasets', 'recipe1m', 'images_5000')

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 10 models - consistent colors across ALL plots
MODEL_ORDER = [
    'GPT-5 Mini', 'Gemini 3.0', 'Gemini 3.1 Lite', 'GPT-4o', 'Gemini 2.0',
    'GPT-4o-mini', 'Gemini 2.5', 'Haiku 4.5', 'Qwen2-VL-7B', 'FatSecret',
]

MODEL_COLORS = {
    'Gemini 2.0': '#4285F4',
    'Gemini 2.5': '#EA4335',
    'Gemini 3.0': '#0F9D58',
    'Gemini 3.1 Lite': '#F4B400',
    'GPT-4o': '#10A37F',
    'GPT-4o-mini': '#74AA9C',
    'GPT-5 Mini': '#1A7F64',
    'Haiku 4.5': '#D97706',
    'Qwen2-VL-7B': '#FF6B35',
    'FatSecret': '#9B59B6',
}

MODEL_FILES = {
    'Gemini 2.0': 'gemini_2.0_flash_P3_full3229.csv',
    'Gemini 2.5': 'gemini_2.5_flash_P3_full3229.csv',
    'Gemini 3.0': 'gemini_3.0_flash_P3_full3229.csv',
    'Gemini 3.1 Lite': 'gemini_3.1_flash_lite_P3_full3229.csv',
    'GPT-4o': 'gpt_4o_P3_full3229.csv',
    'GPT-4o-mini': 'gpt_4o_mini_P3_full3229.csv',
    'GPT-5 Mini': 'gpt5_mini_P3_full3229.csv',
    'Haiku 4.5': 'haiku_4.5_P3_full3229.csv',
    'Qwen2-VL-7B': 'qwen2vl_7b_P3_full3229.csv',
    'FatSecret': 'fatsecret_full3229.csv',
}

# Pastel colors for prompt ablation
PASTEL_BLUE = '#B3D9FF'
PASTEL_GREEN = '#B8E6B8'


def calculate_ccc(y_true, y_pred):
    """Lin's Concordance Correlation Coefficient."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) < 2:
        return np.nan
    mean_true, mean_pred = np.mean(y_true), np.mean(y_pred)
    var_true, var_pred = np.var(y_true, ddof=1), np.var(y_pred, ddof=1)
    covar = np.cov(y_true, y_pred)[0, 1]
    return (2 * covar) / (var_true + var_pred + (mean_true - mean_pred)**2)


def load_data():
    """Load all model data."""
    data = {}
    for name, fname in MODEL_FILES.items():
        fpath = os.path.join(RESULTS_DIR, fname)
        if os.path.exists(fpath):
            df = pd.read_csv(fpath)
            for col in ['ai_calories', 'ai_weight', 'true_calories', 'true_weight', 'ingredient_jaccard']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            if 'ai_calories' in df.columns:
                df = df[df['ai_calories'].notna()]
            data[name] = df
            print(f"  Loaded {name}: {len(df)} rows")
    return data


def _load_font(size=14, scale=FONT_SCALE):
    for font_name in ["DejaVuSans.ttf", "Arial.ttf", "LiberationSans-Regular.ttf"]:
        try:
            return ImageFont.truetype(font_name, size=int(round(fs(size, scale=scale))))
        except:
            continue
    return ImageFont.load_default()


def _wrap_text(text, max_chars=30):
    """Wrap text to multiple lines if too long."""
    import textwrap
    if len(text) <= max_chars:
        return text
    lines = textwrap.wrap(text, width=max_chars)
    return '\n'.join(lines[:2])  # Max 2 lines


def fig1_overview():
    """Fig 1: Experiment overview — A. dataset montage, B. prompt + models, C. JSON output."""
    print("\nGenerating Fig 1 (Overview)...")

    # ── helpers ──────────────────────────────────────────────────────────
    def _get_complex_dish_images(n=12):
        """Return paths to N5K dishes with 3+ pipe-separated ingredients."""
        csv_path = os.path.join(RESULTS_DIR, MODEL_FILES['Gemini 2.0'])
        if not os.path.exists(csv_path):
            return []
        df = pd.read_csv(csv_path)
        df = df[df['true_ingredients'].notna()]
        df = df[df['true_ingredients'].str.count(r'\|') >= 2]
        dish_ids = df['dish_id'].astype(str).tolist()
        random.shuffle(dish_ids)
        paths = []
        for did in dish_ids:
            p = os.path.join(NUTRITION5K_DIR, did, 'rgb.png')
            if os.path.exists(p):
                paths.append(p)
            if len(paths) >= n:
                break
        return paths

    def _create_montage(image_paths, rows=4, cols=3, cell_w=120, cell_h=90, pad=4):
        w = cols * cell_w + (cols + 1) * pad
        h = rows * cell_h + (rows + 1) * pad
        canvas = Image.new('RGB', (w, h), (255, 255, 255))
        for idx, img_path in enumerate(image_paths[:rows * cols]):
            r, c = divmod(idx, cols)
            try:
                img = Image.open(img_path).convert('RGB')
                img = img.resize((cell_w, cell_h), Image.LANCZOS)
                canvas.paste(img, (pad + c * (cell_w + pad), pad + r * (cell_h + pad)))
            except Exception:
                pass
        return canvas

    # ── figure setup ─────────────────────────────────────────────────────
    from matplotlib.patches import FancyBboxPatch
    fig = plt.figure(figsize=(16, 7))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.0, 1.05, 1.05], wspace=0.08,
                  left=0.02, right=0.98, top=0.88, bottom=0.02)
    title_fs = fs_base(16)
    body_fs = fs_base(11.5)

    title_y = 0.96
    content_top = 0.99
    bpad = 0.003
    box_l, box_r = 0.03, 0.97
    box_inner_w = box_r - box_l - 2 * bpad

    # ── Panel A: dataset montage (4x3, complex dishes) ──────────────────
    ax_a = fig.add_subplot(gs[0])
    ax_a.axis('off')
    pos_a = ax_a.get_position()
    fig.text(pos_a.x0, title_y, 'A.  Nutrition5k Dataset',
             fontsize=title_fs, fontweight='bold', va='top')
    imgs = _get_complex_dish_images(n=9)
    if imgs:
        montage = _create_montage(imgs, rows=3, cols=3, cell_w=160, cell_h=120, pad=5)
        ax_a.imshow(montage)
    ax_a.set_anchor('N')

    # ── Panel B: prompt box + model list ─────────────────────────────────
    ax_b = fig.add_subplot(gs[1])
    ax_b.set_xlim(0, 1)
    ax_b.set_ylim(0, 1)
    ax_b.axis('off')
    pos_b = ax_b.get_position()
    fig.text(pos_b.x0, title_y, 'B.  Prompt & Models',
             fontsize=title_fs, fontweight='bold', va='top')

    # Prompt box — 9 lines of monospace, sized to fit snugly
    prompt_text = ('P3 Prompt ("Anti-snap"):\n\n'
                   '"Identify all visible food items.\n'
                   'Estimate the ACTUAL portion size\n'
                   'visible in this photo, NOT\n'
                   'standard serving sizes.\n'
                   'Provide weight in grams and\n'
                   'calories for each item."')
    n_prompt_lines = prompt_text.count('\n') + 1
    line_h = 0.033
    text_pad = 0.025
    prompt_h = n_prompt_lines * line_h + 2 * text_pad
    prompt_y = content_top - prompt_h

    ax_b.add_patch(FancyBboxPatch(
        (box_l + bpad, prompt_y + bpad), box_inner_w, prompt_h - 2 * bpad,
        boxstyle='round,pad=%s' % bpad, facecolor='#E3F2FD',
        edgecolor='#90CAF9', linewidth=lw_base(1.2),
        transform=ax_b.transAxes, clip_on=False))
    ax_b.text(box_l + 0.06, content_top - text_pad, prompt_text,
              transform=ax_b.transAxes, fontsize=body_fs,
              ha='left', va='top', fontfamily='monospace', color='#1565C0')

    # Model rows — directly below prompt box
    models_data = [
        ('GPT-5 Mini',          '$0.87 / 1 K img', MODEL_COLORS['GPT-5 Mini']),
        ('Gemini 3.0 Flash',    '$1.18 / 1 K img', MODEL_COLORS['Gemini 3.0']),
        ('Gemini 3.1 FL-Lite',  '$0.59 / 1 K img', MODEL_COLORS['Gemini 3.1 Lite']),
        ('Gemini 2.0 Flash',    '$0.10 / 1 K img', MODEL_COLORS['Gemini 2.0']),
        ('Gemini 2.5 Flash',    '$0.55 / 1 K img', MODEL_COLORS['Gemini 2.5']),
        ('GPT-4o',              '$2.00 / 1 K img', MODEL_COLORS['GPT-4o']),
        ('GPT-4o-mini',         '$0.15 / 1 K img', MODEL_COLORS['GPT-4o-mini']),
        ('Claude Haiku 4.5',    '$3.24 / 1 K img', MODEL_COLORS['Haiku 4.5']),
        ('Qwen2-VL-7B',        'Free (local)',     MODEL_COLORS['Qwen2-VL-7B']),
        ('FatSecret API *',     'Varies',           MODEL_COLORS['FatSecret']),
    ]
    row_h = 0.038
    gap = 0.008
    y_cursor = prompt_y - gap
    for i, (name, cost, color) in enumerate(models_data):
        y = y_cursor - (i + 1) * row_h - i * gap
        light = color + '22'
        ax_b.add_patch(FancyBboxPatch(
            (box_l + bpad, y + bpad), box_inner_w, row_h - 2 * bpad,
            boxstyle='round,pad=%s' % bpad, facecolor=light, edgecolor=color,
            linewidth=lw_base(1.2), transform=ax_b.transAxes, clip_on=False))
        ax_b.text(box_l + 0.06, y + row_h / 2, name, transform=ax_b.transAxes,
                  va='center', fontsize=body_fs, fontweight='bold', color='#222')
        ax_b.text(box_r - 0.04, y + row_h / 2, cost, transform=ax_b.transAxes,
                  va='center', ha='right', fontsize=fs_base(10), color='#555')

    # ── Panel C: structured JSON output ──────────────────────────────────
    ax_c = fig.add_subplot(gs[2])
    ax_c.set_xlim(0, 1)
    ax_c.set_ylim(0, 1)
    ax_c.axis('off')
    pos_c = ax_c.get_position()
    fig.text(pos_c.x0, title_y, 'C.  Structured Output',
             fontsize=title_fs, fontweight='bold', va='top')

    output_text = ('{\n'
                   '  "general_description":\n'
                   '    "Grilled chicken with rice",\n'
                   '  "items": [\n'
                   '    {\n'
                   '      "name": "grilled chicken",\n'
                   '      "estimated_calories": 165,\n'
                   '      "estimated_weight_grams": 120\n'
                   '    },\n'
                   '    {\n'
                   '      "name": "white rice",\n'
                   '      "estimated_calories": 130,\n'
                   '      "estimated_weight_grams": 100\n'
                   '    }\n'
                   '  ]\n'
                   '}')
    n_json_lines = output_text.count('\n') + 1
    json_h = n_json_lines * line_h + 2 * text_pad
    json_y = content_top - json_h

    ax_c.add_patch(FancyBboxPatch(
        (box_l + bpad, json_y + bpad), box_inner_w, json_h - 2 * bpad,
        boxstyle='round,pad=%s' % bpad, facecolor='#E8F5E9',
        edgecolor='#A5D6A7', linewidth=lw_base(1.2),
        transform=ax_c.transAxes, clip_on=False))
    ax_c.text(box_l + 0.06, content_top - text_pad, output_text,
              transform=ax_c.transAxes, fontsize=body_fs,
              ha='left', va='top', fontfamily='monospace', color='#2E7D32')

    save_fig(fig, os.path.join(OUTPUT_DIR, 'fig1_overview.png'), facecolor='white')
    plt.close()
    print("  Saved fig1_overview.png")


def fig2_prompt_ablation():
    """Fig 2: Prompt Ablation Study - P1-P6, P3 = pastel green, rest = pastel blue."""
    print("\nGenerating Fig 2 (Prompt Ablation)...")
    
    prompts = ['P1', 'P2', 'P3', 'P4', 'P5']
    
    metrics = {'CCC': [], 'MAE': [], 'Within-25%': []}
    valid_prompts = []
    
    for p in prompts:
        # Files are in prompt_ablation_full3229 subfolder
        fpath = os.path.join(PROMPT_ABLATION_DIR, f'gemini_2.0_flash_{p}_full3229.csv')
        if not os.path.exists(fpath):
            print(f"  Warning: {fpath} not found")
            continue
            
        df = pd.read_csv(fpath)
        df['true_calories'] = pd.to_numeric(df['true_calories'], errors='coerce')
        df['ai_calories'] = pd.to_numeric(df['ai_calories'], errors='coerce')
        
        df = df.dropna(subset=['true_calories', 'ai_calories'])
        df = df[(df['true_calories'] > 0) & (df['ai_calories'] > 0)]
        df = df[(df['true_calories'] <= 1500) & (df['ai_calories'] <= 1500)]
        
        if len(df) < 10:
            continue
            
        tc, ac = df['true_calories'].values, df['ai_calories'].values
        
        metrics['CCC'].append(calculate_ccc(tc, ac))
        metrics['MAE'].append(np.mean(np.abs(tc - ac)))
        metrics['Within-25%'].append(np.mean(np.abs(ac - tc) / tc <= 0.25) * 100)
        valid_prompts.append(p)
    
    if not valid_prompts:
        print("  No prompt data found, skipping Fig 2")
        return
    
    print(f"  Found data for prompts: {valid_prompts}")
    
    # P3 = pastel green, rest = pastel blue
    colors = [PASTEL_GREEN if p == 'P3' else PASTEL_BLUE for p in valid_prompts]
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    x = np.arange(len(valid_prompts))
    
    titles = ['A. Concordance Correlation', 'B. Mean Absolute Error (kcal)', 'C. Within-25% Accuracy (%)']
    ylims = [(0, 1), (0, max(metrics['MAE']) * 1.2 if metrics['MAE'] else 200), (0, 100)]
    
    for ax, (metric_name, values), title, ylim in zip(axes, metrics.items(), titles, ylims):
        # Box around plot
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(lw(1))
        
        bars = ax.bar(x, values, color=colors, edgecolor='black', linewidth=lw(0.8), width=0.7)
        # Value labels on top
        for bar, val in zip(bars, values):
            fmt = f'{val:.3f}' if metric_name == 'CCC' else (f'{val:.1f}' if metric_name == 'MAE' else f'{val:.1f}%')
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (ylim[1] * 0.01),
                    fmt, ha='center', va='bottom', fontsize=fs(8))
        ax.set_xticks(x)
        ax.set_xticklabels(valid_prompts, fontsize=fs(11))
        ax.set_title(title, fontsize=fs(12), pad=10)
        ax.set_ylim(ylim[0], ylim[1] * 1.08)
        ax.grid(axis='y', alpha=0.3, linestyle='-')
        ax.set_axisbelow(True)
    
    plt.suptitle('Prompt Ablation Study (Gemini 2.0 Flash, N=3,229)', fontsize=fs(14), y=1.02)
    plt.tight_layout()
    save_fig(fig, os.path.join(OUTPUT_DIR, 'fig2_prompt_ablation.png'))
    plt.close()
    print("  Saved fig2_prompt_ablation.png")


def fig3_scatter_calories(data):
    """Fig 3: Calorie Scatter - grid of models, axes 0-1500."""
    print("\nGenerating Fig 3 (Calorie Scatter)...")
    n_models = len(MODEL_ORDER)
    ncols = 5
    nrows = (n_models + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))
    axes = axes.flatten()
    
    for idx, model in enumerate(MODEL_ORDER):
        ax = axes[idx]
        if model not in data:
            ax.text(0.5, 0.5, f'{model}\nNo data', ha='center', va='center', transform=ax.transAxes)
            ax.set_xlim(0, 1500)
            ax.set_ylim(0, 1500)
            continue
        
        df = data[model].copy()
        df = df.dropna(subset=['true_calories', 'ai_calories'])
        df = df[(df['true_calories'] > 0) & (df['ai_calories'] > 0)]
        df = df[(df['true_calories'] <= 1500) & (df['ai_calories'] <= 1500)]
        
        x, y = df['true_calories'], df['ai_calories']
        ax.scatter(x, y, c=MODEL_COLORS[model], s=2, alpha=0.8, linewidths=0)
        ax.plot([0, 1500], [0, 1500], 'k--', alpha=0.5, linewidth=lw(1))
        
        ax.set_xlim(0, 1500)
        ax.set_ylim(0, 1500)
        
        ccc = calculate_ccc(x, y)
        ax.set_title(f'{model}\nCCC={ccc:.3f}, n={len(df)}')
        ax.set_xlabel('True Calories (kcal)')
        ax.set_ylabel('Predicted Calories (kcal)')
    
    plt.suptitle('Calorie Estimation Accuracy', fontsize=fs(14))
    plt.tight_layout()
    save_fig(fig, os.path.join(OUTPUT_DIR, 'fig3_scatter_calories.png'))
    plt.close()
    print("  Saved fig3_scatter_calories.png")


def fig4_scatter_weight(data):
    """Fig 4: Weight Scatter - grid of models, axes 0-1200."""
    print("\nGenerating Fig 4 (Weight Scatter)...")
    n_models = len(MODEL_ORDER)
    ncols = 5
    nrows = (n_models + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))
    axes = axes.flatten()
    
    for idx, model in enumerate(MODEL_ORDER):
        ax = axes[idx]
        if model not in data:
            ax.text(0.5, 0.5, f'{model}\nNo data', ha='center', va='center', transform=ax.transAxes)
            ax.set_xlim(0, 1200)
            ax.set_ylim(0, 1200)
            continue
        
        df = data[model].copy()
        df = df.dropna(subset=['true_weight', 'ai_weight'])
        df = df[(df['true_weight'] > 0) & (df['ai_weight'] > 0)]
        df = df[(df['true_weight'] <= 1200) & (df['ai_weight'] <= 1200)]
        
        x, y = df['true_weight'], df['ai_weight']
        ax.scatter(x, y, c=MODEL_COLORS[model], s=2, alpha=0.8, linewidths=0)
        ax.plot([0, 1200], [0, 1200], 'k--', alpha=0.5, linewidth=lw(1))
        
        ax.set_xlim(0, 1200)
        ax.set_ylim(0, 1200)
        
        ccc = calculate_ccc(x, y)
        ax.set_title(f'{model}\nCCC={ccc:.3f}, n={len(df)}')
        ax.set_xlabel('True Weight (g)')
        ax.set_ylabel('Predicted Weight (g)')
    
    plt.suptitle('Weight Estimation Accuracy', fontsize=fs(14))
    plt.tight_layout()
    save_fig(fig, os.path.join(OUTPUT_DIR, 'fig4_scatter_weight.png'))
    plt.close()
    print("  Saved fig4_scatter_weight.png")


def fig5_model_comparison(data):
    """Fig 5: Model Comparison with ±1 SD error bars.

    - CCC error bars: bootstrap SD over dishes (resampling pairs).
    - Jaccard error bars: bootstrap SD of the *mean* ingredient_jaccard (resampling dishes).
    """
    print("\nGenerating Fig 5 (Model Comparison)...")
    
    metrics = {'Calorie CCC': [], 'Weight CCC': [], 'Jaccard': []}
    errors = {'Calorie CCC': [], 'Weight CCC': [], 'Jaccard': []}
    model_names = []
    colors = []

    def _bootstrap_ccc_sd(x, y, n_boot=250, seed=42):
        x = np.asarray(x)
        y = np.asarray(y)
        mask = ~(np.isnan(x) | np.isnan(y))
        x = x[mask]
        y = y[mask]
        if len(x) < 3:
            return np.nan
        rng = np.random.RandomState(seed)
        vals = []
        n = len(x)
        for _ in range(n_boot):
            idx = rng.randint(0, n, size=n)
            v = calculate_ccc(x[idx], y[idx])
            if not np.isnan(v):
                vals.append(v)
        return float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan

    def _bootstrap_mean_sd(vals, n_boot=250, seed=42):
        vals = np.asarray(vals, dtype=float)
        vals = vals[~np.isnan(vals)]
        if len(vals) < 3:
            return np.nan
        rng = np.random.RandomState(seed)
        n = len(vals)
        means = []
        for _ in range(n_boot):
            idx = rng.randint(0, n, size=n)
            means.append(float(np.mean(vals[idx])))
        return float(np.std(means, ddof=1)) if len(means) > 1 else np.nan
    
    for model in MODEL_ORDER:
        if model not in data:
            continue
        df = data[model]
        
        df_cal = df.dropna(subset=['true_calories', 'ai_calories'])
        df_cal = df_cal[(df_cal['ai_calories'] > 0) & (df_cal['true_calories'] <= 1500) & (df_cal['ai_calories'] <= 1500)]
        cal_ccc = calculate_ccc(df_cal['true_calories'], df_cal['ai_calories'])
        cal_ccc_sd = _bootstrap_ccc_sd(df_cal['true_calories'].values, df_cal['ai_calories'].values)
        
        df_wt = df.dropna(subset=['true_weight', 'ai_weight'])
        df_wt = df_wt[(df_wt['ai_weight'] > 0) & (df_wt['true_weight'] <= 1500) & (df_wt['ai_weight'] <= 1500)]
        wt_ccc = calculate_ccc(df_wt['true_weight'], df_wt['ai_weight'])
        wt_ccc_sd = _bootstrap_ccc_sd(df_wt['true_weight'].values, df_wt['ai_weight'].values)
        
        jvals = df['ingredient_jaccard'].dropna().values
        jaccard = float(np.mean(jvals)) if len(jvals) else np.nan
        jaccard_sd = _bootstrap_mean_sd(jvals)
        
        metrics['Calorie CCC'].append(cal_ccc if not np.isnan(cal_ccc) else 0)
        metrics['Weight CCC'].append(wt_ccc if not np.isnan(wt_ccc) else 0)
        metrics['Jaccard'].append(jaccard if not np.isnan(jaccard) else 0)
        errors['Calorie CCC'].append(cal_ccc_sd if not np.isnan(cal_ccc_sd) else 0)
        errors['Weight CCC'].append(wt_ccc_sd if not np.isnan(wt_ccc_sd) else 0)
        errors['Jaccard'].append(jaccard_sd if not np.isnan(jaccard_sd) else 0)
        model_names.append(model)
        colors.append(MODEL_COLORS[model])  # SAME colors as violin
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    x = np.arange(len(model_names))

    titles = ['A. Calorie CCC', 'B. Weight CCC', 'C. Ingredient Jaccard']

    for ax, (metric_name, values), title in zip(axes, metrics.items(), titles):
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(lw(1))

        yerr = np.asarray(errors[metric_name], dtype=float)
        bars = ax.bar(
            x,
            values,
            yerr=yerr,
            capsize=3,
            ecolor='black',
            error_kw={'elinewidth': lw(1.0), 'capthick': lw(1.0)},
            color=colors,
            edgecolor='black',
            linewidth=lw(0.8),
            width=0.7,
            alpha=0.7,
        )
        # Value labels on top of error bars
        for bar, val, err in zip(bars, values, yerr):
            ax.text(bar.get_x() + bar.get_width()/2, val + err + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=fs(6))
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=55, ha='right', fontsize=fs(7))
        ax.set_title(title, fontsize=fs(12), pad=10)
        ax.set_ylim(0, 1.05)
        ax.grid(axis='y', alpha=0.3, linestyle='-')
        ax.set_axisbelow(True)

    plt.suptitle('Model Performance Comparison', fontsize=fs(14), y=1.02)
    plt.tight_layout()
    save_fig(fig, os.path.join(OUTPUT_DIR, 'fig5_model_comparison.png'))
    plt.close()
    print("  Saved fig5_model_comparison.png")


def fig6_violin_radar(data):
    """Fig 6: Violin + Radar side by side."""
    print("\nGenerating Fig 6 (Violin + Radar)...")
    
    fig = plt.figure(figsize=(16, 8))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 1])
    
    # Panel A: Box plot (ingredient Jaccard distribution)
    ax_violin = fig.add_subplot(gs[0])
    for spine in ax_violin.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(lw(1))
        spine.set_zorder(0)
    
    jaccard_data = []
    model_names = []
    colors = []
    
    for model in MODEL_ORDER:
        if model in data:
            df = data[model]
            jaccards = df['ingredient_jaccard'].dropna().values
            # Clamp to valid range to prevent whiskers from extending below 0
            jaccards = np.clip(jaccards, 0, 1)
            if len(jaccards) > 0:
                jaccard_data.append(jaccards)
                model_names.append(model)
                colors.append(MODEL_COLORS[model])
    
    stats = []
    for vals in jaccard_data:
        vals = np.clip(np.asarray(vals), 0, 1)
        if len(vals) == 0:
            continue
        q1, q3 = np.percentile(vals, [25, 75])
        med = np.percentile(vals, 50)
        whislo = float(max(0.0, np.min(vals)))
        whishi = float(min(1.0, np.max(vals)))
        stats.append({
            'med': float(med),
            'q1': float(q1),
            'q3': float(q3),
            'whislo': whislo,
            'whishi': whishi,
            'fliers': [],
        })

    parts = ax_violin.bxp(
        stats,
        positions=range(len(stats)),
        patch_artist=True,
        showfliers=False,
        widths=0.6,
    )

    for i, box in enumerate(parts['boxes']):
        box.set_facecolor(colors[i])
        box.set_edgecolor('black')
        box.set_linewidth(lw(1.0))
        box.set_alpha(0.7)
        box.set_clip_on(True)
        box.set_zorder(2)

    for key in ['whiskers', 'caps', 'medians']:
        for item in parts[key]:
            if key in {'whiskers', 'caps'}:
                item.set_color('red')
            else:
                item.set_color('black')
            item.set_linewidth(lw(1.0))
            if hasattr(item, "get_ydata"):
                ydata = item.get_ydata()
                if ydata is not None:
                    item.set_ydata(np.clip(ydata, 0, 1))
            item.set_clip_on(True)
            if hasattr(item, "set_solid_capstyle"):
                item.set_solid_capstyle('butt')
            if key == 'caps':
                item.set_zorder(5)
            elif key == 'whiskers':
                item.set_zorder(4)
            else:
                item.set_zorder(3)
    
    ax_violin.set_xticks(range(len(model_names)))
    ax_violin.set_xticklabels(model_names, rotation=45, ha='right')
    # Remove x-axis tick marks so they don't look like whisker continuations
    ax_violin.tick_params(axis='x', length=0)
    ax_violin.set_ylabel('Ingredient Jaccard Score')
    ax_violin.set_title('A. Ingredient Detection (Boxplot)')
    ax_violin.set_ylim(0, 1.05)
    ax_violin.grid(axis='y', alpha=0.3)
    ax_violin.set_axisbelow(True)
    
    # Panel B: Radar
    ax_radar = fig.add_subplot(gs[1], polar=True)
    
    metrics_names = ['Cal CCC', 'Wt CCC', 'Within-25%', 'Jaccard', 'Cost Eff']
    num_vars = len(metrics_names)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    
    costs = {'Gemini 2.0': 0.10, 'Gemini 2.5': 0.55, 'GPT-4o': 2.00,
             'GPT-4o-mini': 0.15, 'Qwen2-VL-7B': 0.01, 'FatSecret': 0.50}
    
    for model in MODEL_ORDER:
        if model not in data:
            continue
        df = data[model]
        
        df_cal = df.dropna(subset=['true_calories', 'ai_calories'])
        df_cal = df_cal[(df_cal['ai_calories'] > 0) & (df_cal['true_calories'] <= 1500) & (df_cal['ai_calories'] <= 1500)]
        cal_ccc = calculate_ccc(df_cal['true_calories'], df_cal['ai_calories'])
        
        df_wt = df.dropna(subset=['true_weight', 'ai_weight'])
        df_wt = df_wt[(df_wt['ai_weight'] > 0) & (df_wt['true_weight'] <= 1500) & (df_wt['ai_weight'] <= 1500)]
        wt_ccc = calculate_ccc(df_wt['true_weight'], df_wt['ai_weight'])
        
        within_25 = 0
        if len(df_cal) > 0:
            within_25 = (np.abs(df_cal['ai_calories'] - df_cal['true_calories']) / df_cal['true_calories'] <= 0.25).mean()
        
        jaccard = df['ingredient_jaccard'].mean()
        cost_eff = min(1 / (costs.get(model, 1) + 0.01) / 10, 1)
        
        values = [
            cal_ccc if not np.isnan(cal_ccc) else 0,
            wt_ccc if not np.isnan(wt_ccc) else 0,
            within_25,
            jaccard if not np.isnan(jaccard) else 0,
            cost_eff
        ]
        values += values[:1]
        
        ax_radar.plot(angles, values, 'o-', linewidth=lw(1.0), label=model,
                      color=MODEL_COLORS[model], markersize=3)
        ax_radar.fill(angles, values, alpha=0.05, color=MODEL_COLORS[model])
    
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(metrics_names, fontsize=fs(9))
    ax_radar.set_ylim(0, 1.15)
    ax_radar.tick_params(pad=fs(8))
    # Use a single-row legend below both plots
    handles, labels = ax_radar.get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc='lower center',
            bbox_to_anchor=(0.5, 0.02),
            ncol=max(1, len(labels)),
            fontsize=fs(11),
            frameon=False,
        )
    ax_radar.set_title('B. Performance Radar')
    
    plt.tight_layout(rect=[0, 0.10, 1, 1])
    save_fig(fig, os.path.join(OUTPUT_DIR, 'fig6_violin_radar.png'))
    plt.close()
    print("  Saved fig6_violin_radar.png")


def count_ingredients(s):
    if pd.isna(s) or s == '' or str(s) == 'nan':
        return 0
    parts = [p.strip() for p in str(s).split('|') if p.strip()]
    return len(parts)


def fig7_complexity_heatmap(data):
    """Fig 7: Heatmap with columns 1, 2, 3, 4+."""
    print("\nGenerating Fig 7 (Complexity Heatmap)...")
    
    heatmap_data = []
    model_labels = []
    count_labels = {1: 0, 2: 0, 3: 0, '4+': 0}
    count_source_done = False
    
    for model in MODEL_ORDER:
        if model not in data:
            continue
        df = data[model].copy()
        df['n_ingredients'] = df['true_ingredients'].apply(count_ingredients)
        if not count_source_done:
            count_labels[1] = int((df['n_ingredients'] == 1).sum())
            count_labels[2] = int((df['n_ingredients'] == 2).sum())
            count_labels[3] = int((df['n_ingredients'] == 3).sum())
            count_labels['4+'] = int((df['n_ingredients'] >= 4).sum())
            count_source_done = True
        
        row = []
        for n in [1, 2, 3]:
            subset = df[df['n_ingredients'] == n]
            if len(subset) > 0:
                row.append(subset['ingredient_jaccard'].mean())
            else:
                row.append(np.nan)
        
        subset = df[df['n_ingredients'] >= 4]
        if len(subset) > 0:
            row.append(subset['ingredient_jaccard'].mean())
        else:
            row.append(np.nan)
        
        heatmap_data.append(row)
        model_labels.append(model)
    
    heatmap_arr = np.array(heatmap_data)
    labels = [
        f"1 (n={count_labels[1]})",
        f"2 (n={count_labels[2]})",
        f"3 (n={count_labels[3]})",
        f"4+ (n={count_labels['4+']})",
    ]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(heatmap_arr, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticks(range(len(model_labels)))
    ax.set_yticklabels(model_labels)
    ax.set_xlabel('Number of Visible Ingredients')
    ax.set_ylabel('Model')
    ax.set_title('Ingredient Detection by Dish Complexity')
    
    for i in range(len(model_labels)):
        for j in range(len(labels)):
            val = heatmap_arr[i, j]
            if not np.isnan(val):
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', color='black', fontsize=fs(10))
    
    plt.colorbar(im, ax=ax, label='Mean Jaccard Score')
    plt.tight_layout()
    save_fig(fig, os.path.join(OUTPUT_DIR, 'fig7_complexity_heatmap.png'))
    plt.close()
    print("  Saved fig7_complexity_heatmap.png")


def fig8_jaccard_scatter(data):
    """Fig 8: Jaccard-colored scatter."""
    print("\nGenerating Fig 8 (Jaccard Scatter)...")
    
    if 'Gemini 2.0' not in data:
        print("  Skipping - no Gemini 2.0 data")
        return
    
    df = data['Gemini 2.0'].copy()
    df = df.dropna(subset=['true_calories', 'ai_calories', 'ingredient_jaccard'])
    df = df[(df['true_calories'] > 0) & (df['ai_calories'] > 0)]
    df = df[(df['true_calories'] <= 1500) & (df['ai_calories'] <= 1500)]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    ax = axes[0]
    scatter = ax.scatter(df['true_calories'], df['ai_calories'],
                         c=df['ingredient_jaccard'], cmap='RdYlGn',
                         s=3, alpha=0.8, vmin=0, vmax=1)
    ax.plot([0, 1500], [0, 1500], 'k--', alpha=0.5)
    ax.set_xlim(0, 1500)
    ax.set_ylim(0, 1500)
    ax.set_xlabel('True Calories (kcal)')
    ax.set_ylabel('Predicted Calories (kcal)')
    ax.set_title('A. Calorie Estimation vs. Ingredient Recognition', pad=fs(12))
    plt.colorbar(scatter, ax=ax, label='Jaccard Score')
    
    ax = axes[1]
    df2 = df.dropna(subset=['true_weight', 'ai_weight'])
    df2 = df2[(df2['true_weight'] > 0) & (df2['ai_weight'] > 0)]
    df2 = df2[(df2['true_weight'] <= 1200) & (df2['ai_weight'] <= 1200)]
    
    scatter = ax.scatter(df2['true_weight'], df2['ai_weight'],
                         c=df2['ingredient_jaccard'], cmap='RdYlGn',
                         s=3, alpha=0.8, vmin=0, vmax=1)
    ax.plot([0, 1200], [0, 1200], 'k--', alpha=0.5)
    ax.set_xlim(0, 1200)
    ax.set_ylim(0, 1200)
    ax.set_xlabel('True Weight (g)')
    ax.set_ylabel('Predicted Weight (g)')
    ax.set_title('B. Weight Estimation vs. Ingredient Recognition', pad=fs(12))
    plt.colorbar(scatter, ax=ax, label='Jaccard Score')
    
    plt.tight_layout()
    save_fig(fig, os.path.join(OUTPUT_DIR, 'fig8_jaccard_scatter.png'))
    plt.close()
    print("  Saved fig8_jaccard_scatter.png")


def fig9_low_jaccard_montage(data):
    """
    Fig 9: 4x4 montage with detailed labels:
    True: C:305 | W:365
    apple, canteloupe, almonds
    AI: C:317 | W:382 | J:0.33
    Apple, cauliflower florets, almonds
    """
    print("\nGenerating Fig 9 (Low Jaccard Montage)...")
    
    if 'Gemini 2.0' not in data:
        print("  Skipping - no Gemini 2.0 data")
        return
    
    df = data['Gemini 2.0'].copy()
    df['n_ingredients'] = df['true_ingredients'].apply(count_ingredients)
    
    # Complex dishes with low Jaccard
    sub = df[(df['n_ingredients'] >= 3) & df['ingredient_jaccard'].notna()]
    sub = sub.sort_values('ingredient_jaccard', ascending=True).head(16)
    
    if len(sub) == 0:
        print("  No suitable dishes found")
        return
    
    cols, rows = 4, 4
    cell_w, cell_h = 210, 200
    padding = 12

    font = _load_font(11)
    font_small = _load_font(9)

    def _line_height(font_obj, pad=3):
        try:
            bbox = font_obj.getbbox("Ag")
            return int((bbox[3] - bbox[1]) + pad)
        except Exception:
            return 12 + pad

    line_height = _line_height(font_small, pad=3)
    label_text_h = line_height

    items = []
    max_label_h = 0
    for _, row in sub.iterrows():
        true_ing = str(row.get('true_ingredients', '')).replace('|', ', ')
        dish_id = str(row.get('dish_id', ''))
        cache_file = os.path.join(CACHE_DIR, f"{dish_id}.json")
        ai_ing = ""
        if os.path.exists(cache_file):
            try:
                with open(cache_file) as cf:
                    cdata = json.load(cf)
                analysis = cdata.get('analysis', cdata)
                if isinstance(analysis, dict):
                    items_list = analysis.get('items', [])
                    ai_ing = ', '.join([str(item.get('name', '')) for item in items_list if isinstance(item, dict)])
            except Exception:
                pass

        true_ing_wrapped = _wrap_text(true_ing, max_chars=24)
        ai_ing_wrapped = _wrap_text(ai_ing, max_chars=24)
        true_lines = len(true_ing_wrapped.split('\n'))
        ai_lines = len(ai_ing_wrapped.split('\n'))
        needed = (
            2
            + label_text_h
            + true_lines * line_height
            + 6
            + label_text_h
            + ai_lines * line_height
            + 2
        )
        max_label_h = max(max_label_h, needed)
        items.append((row, true_ing_wrapped, ai_ing_wrapped, true_lines, ai_lines))

    label_h = max(110, max_label_h)
    
    canvas_w = padding + cols * (cell_w + padding)
    canvas_h = padding + rows * (cell_h + label_h + padding)
    canvas = Image.new('RGB', (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    
    for idx, (row, true_ing_wrapped, ai_ing_wrapped, true_lines, ai_lines) in enumerate(items):
        if idx >= cols * rows:
            break
        
        dish_id = str(row.get('dish_id', ''))
        img_path = os.path.join(NUTRITION5K_DIR, dish_id, 'rgb.png')
        
        if not os.path.exists(img_path):
            continue
        
        try:
            img = Image.open(img_path).convert('RGB')
            img.thumbnail((cell_w, cell_h), Image.LANCZOS)
            tile = Image.new('RGB', (cell_w, cell_h), (240, 240, 240))
            off_x = (cell_w - img.size[0]) // 2
            off_y = (cell_h - img.size[1]) // 2
            tile.paste(img, (off_x, off_y))
            
            c = idx % cols
            r = idx // cols
            x0 = padding + c * (cell_w + padding)
            y0 = padding + r * (cell_h + label_h + padding)
            
            canvas.paste(tile, (x0, y0))
            
            # Labels
            true_cal = row.get('true_calories', 0)
            true_wt = row.get('true_weight', 0)
            ai_cal = row.get('ai_calories', 0)
            ai_wt = row.get('ai_weight', 0)
            j = row.get('ingredient_jaccard', 0)
            label_x = x0 + 2
            label_y = y0 + cell_h + 2
            draw.text((label_x, label_y), f"Original: C:{int(true_cal)} | W:{int(true_wt)}", fill=(0, 100, 0), font=font_small)
            true_text_y = label_y + label_text_h
            for i, line in enumerate(true_ing_wrapped.split('\n')):
                draw.text((label_x, true_text_y + i * line_height), line, fill=(0, 0, 0), font=font_small)

            ai_label_y = true_text_y + true_lines * line_height + 6
            draw.text((label_x, ai_label_y), f"AI: C:{int(ai_cal)} | W:{int(ai_wt)} | J:{j:.2f}", fill=(0, 0, 150), font=font_small)
            ai_text_y = ai_label_y + label_text_h
            for i, line in enumerate(ai_ing_wrapped.split('\n')):
                draw.text((label_x, ai_text_y + i * line_height), line, fill=(0, 0, 0), font=font_small)
            
        except Exception as e:
            print(f"  Error: {e}")
    
    canvas.save(os.path.join(OUTPUT_DIR, 'fig9_low_jaccard_montage.png'))
    print("  Saved fig9_low_jaccard_montage.png")


def fig10_confusion_matrix():
    """Fig 10: Confusion Matrix - top 20, n= counts, recall colorbar."""
    print("\nGenerating Fig 10 (Confusion Matrix)...")
    
    CSV_PATH = os.path.join(RESULTS_DIR, 'gemini_2.0_flash_P3_full3229.csv')
    
    if not os.path.exists(CSV_PATH):
        print(f"  CSV not found: {CSV_PATH}")
        return
    
    STOPWORDS = {
        "and", "with", "of", "a", "an", "the", "on", "in", "to", "for",
        "some", "type", "kind", "few", "several", "cooked", "raw", "fried",
        "grilled", "roasted", "baked", "mashed", "sauteed", "fresh", "dried",
        "sliced", "diced", "chopped", "mixed", "steamed", "scrambled",
    }
    
    SYNONYMS = {"aubergine": "eggplant", "courgette": "zucchini", "cherry": "tomato"}
    
    EXCLUDED = {
        "nan", "", "caesar", "brussel", "bell", "cream", "crouton", "vegetable",
        "green", "slice", "sliced", "sweet", "berry", "red", "white", "brown",
    }
    
    def normalize_token(token):
        t = token.lower().strip()
        if t.endswith("ies") and len(t) > 3:
            t = t[:-3] + "y"
        elif t.endswith("es") and len(t) > 3:
            t = t[:-2]
        elif t.endswith("s") and len(t) > 3:
            t = t[:-1]
        return SYNONYMS.get(t, t)
    
    def phrase_to_tokens(phrase):
        raw = re.findall(r"[a-zA-Z]+", phrase.lower())
        tokens = []
        for tok in raw:
            t = normalize_token(tok)
            if t and t not in STOPWORDS and t not in EXCLUDED and len(t) > 2:
                tokens.append(t)
        return list(set(tokens))
    
    true_counts = Counter()
    pred_counts = Counter()
    confusion = defaultdict(lambda: defaultdict(int))
    
    import csv
    with open(CSV_PATH, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            dish_id = row.get('dish_id', '').strip()
            true_ing_str = row.get('true_ingredients', '') or ''
            
            cache_file = os.path.join(CACHE_DIR, f"{dish_id}.json")
            ai_ingredients = []
            if os.path.exists(cache_file):
                try:
                    with open(cache_file) as cf:
                        cdata = json.load(cf)
                    analysis = cdata.get('analysis', cdata) if isinstance(cdata, dict) else cdata
                    if isinstance(analysis, dict):
                        for item in analysis.get('items') or []:
                            if isinstance(item, dict) and item.get('name'):
                                ai_ingredients.append(str(item['name']))
                except:
                    continue
            
            true_tokens = set()
            for phrase in true_ing_str.split('|'):
                true_tokens.update(phrase_to_tokens(phrase))
            
            pred_tokens = set()
            for phrase in ai_ingredients:
                pred_tokens.update(phrase_to_tokens(phrase))
            
            for t in true_tokens:
                true_counts[t] += 1
            for p in pred_tokens:
                pred_counts[p] += 1
            
            for t in true_tokens:
                for p in pred_tokens:
                    confusion[t][p] += 1
    
    top_n = 20
    combined = Counter()
    for tok in set(list(true_counts.keys()) + list(pred_counts.keys())):
        combined[tok] = true_counts[tok] + pred_counts[tok]
    
    top_tokens = [tok for tok, _ in combined.most_common(top_n)]
    
    matrix = np.zeros((len(top_tokens), len(top_tokens)), dtype=float)
    for i, pred_tok in enumerate(top_tokens):
        for j, true_tok in enumerate(top_tokens):
            matrix[i, j] = confusion[true_tok].get(pred_tok, 0)
    
    col_sums = np.array([true_counts[tok] for tok in top_tokens], dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        matrix_norm = matrix / col_sums * 100
    matrix_norm = np.nan_to_num(matrix_norm, nan=0.0)
    
    y_labels = [f"{tok.title()} ({pred_counts[tok]})" for tok in top_tokens]
    x_labels = [f"{tok.title()} ({true_counts[tok]})" for tok in top_tokens]
    
    fig, ax = plt.subplots(figsize=(18, 12))
    
    sns.heatmap(
        matrix_norm,
        xticklabels=x_labels,
        yticklabels=y_labels,
        cmap='YlOrRd',
        vmin=0, vmax=100,
        annot=np.round(matrix_norm, 0).astype(int),
        fmt='d',
        annot_kws={'fontsize': fs(7) * 1.5},
        cbar_kws={'label': 'Recall (%)'},
        linewidths=lw(0.5),
        linecolor='gray',
        ax=ax
    )
    
    ax.set_xlabel('Ground Truth Ingredient')
    ax.set_ylabel('AI Predicted Ingredient')
    ax.set_title(f'Ingredient Confusion Matrix (Top {top_n}, Gemini 2.0 Flash, N=3,229)')
    
    plt.xticks(rotation=45, ha='right', fontsize=fs(9))
    plt.yticks(rotation=0, fontsize=fs(9))
    plt.subplots_adjust(bottom=0.35)
    
    plt.tight_layout()
    save_fig(fig, os.path.join(OUTPUT_DIR, 'fig10_confusion_matrix.png'))
    plt.close()
    print("  Saved fig10_confusion_matrix.png")


def fig11_cross_dataset_montages():
    """
    Fig 11: Cross-dataset montages with True/AI labels.
    """
    print("\nGenerating Fig 11 (Cross-dataset Montages)...")
    
    # Load FoodSeg103 results
    fis_csv = os.path.join(RESULTS_DIR, 'foodinsseg', 'foodinsseg_gemini_visible_ingredients.csv')
    r1m_csv = os.path.join(RESULTS_DIR, 'recipe1m', 'recipe1m_gemini_results.csv')
    
    def create_labeled_montage(csv_path, image_base_dir, output_name, title, true_col='true_ingredients', pred_col='predicted_ingredients', img_col='image_id'):
        if not os.path.exists(csv_path):
            print(f"  CSV not found: {csv_path}")
            return
        
        df = pd.read_csv(csv_path)
        df = df.head(8)  # 2x4 grid
        
        cols, rows = 4, 2
        cell_w, cell_h = 220, 200
        padding = 12

        font = _load_font(14)
        font_small = _load_font(10)

        def _line_height(font_obj, pad=2):
            try:
                bbox = font_obj.getbbox("Ag")
                return int((bbox[3] - bbox[1]) + pad)
            except Exception:
                return 12 + pad

        line_height = _line_height(font_small, pad=2)
        label_text_h = line_height

        # Pre-wrap labels and compute required label height (avoid cutoff)
        items = []
        max_label_h = 0
        for _, row in df.iterrows():
            true_ing = str(row.get(true_col, '')).replace('|', ', ')
            pred_ing = str(row.get(pred_col, '')).replace('|', ', ')
            true_ing_wrapped = _wrap_text(true_ing, max_chars=22)
            pred_ing_wrapped = _wrap_text(pred_ing, max_chars=22)
            true_lines = len(true_ing_wrapped.split('\n'))
            pred_lines = len(pred_ing_wrapped.split('\n'))
            needed = (
                2  # top padding
                + label_text_h
                + true_lines * line_height
                + 4
                + label_text_h
                + pred_lines * line_height
                + 2
            )
            max_label_h = max(max_label_h, needed)
            items.append((row, true_ing_wrapped, pred_ing_wrapped, true_lines, pred_lines))

        label_h = max(70, max_label_h)
        
        canvas_w = padding + cols * (cell_w + padding)
        canvas_h = 30 + padding + rows * (cell_h + label_h + padding)
        canvas = Image.new('RGB', (canvas_w, canvas_h), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        
        # Title (left aligned)
        draw.text((5, 10), title, fill=(0, 0, 0), font=font)
        
        for idx, (row, true_ing_wrapped, pred_ing_wrapped, true_lines, pred_lines) in enumerate(items):
            if idx >= cols * rows:
                break
            
            # Find image
            img_id = str(row.get(img_col, row.get('image_id', row.get('dish_id', ''))))
            img_path = None
            
            # Try different paths
            for ext in ['', '.jpg', '.png']:
                for subdir in ['', 'train/', 'test/']:
                    test_path = os.path.join(image_base_dir, subdir, img_id + ext)
                    if os.path.exists(test_path):
                        img_path = test_path
                        break
                if img_path:
                    break
            
            if not img_path:
                print(f"  Image not found for {img_id}")
                continue
            
            try:
                img = Image.open(img_path).convert('RGB')
                w, h = img.size
                min_dim = min(w, h)
                left, top = (w - min_dim) // 2, (h - min_dim) // 2
                img = img.crop((left, top, left + min_dim, top + min_dim))
                img = img.resize((cell_w, cell_h), Image.LANCZOS)
                
                c = idx % cols
                r = idx // cols
                x0 = padding + c * (cell_w + padding)
                y0 = 30 + padding + r * (cell_h + label_h + padding)
                
                canvas.paste(img, (x0, y0))
                
                label_x = x0 + 2
                label_y = y0 + cell_h + 2
                draw.text((label_x, label_y), "True:", fill=(0, 100, 0), font=font_small)
                true_text_y = label_y + label_text_h
                for i, line in enumerate(true_ing_wrapped.split('\n')):
                    draw.text((label_x, true_text_y + i * line_height), line, fill=(0, 0, 0), font=font_small)

                ai_label_y = true_text_y + true_lines * line_height + 4
                draw.text((label_x, ai_label_y), "AI:", fill=(0, 0, 150), font=font_small)
                ai_text_y = ai_label_y + label_text_h
                for i, line in enumerate(pred_ing_wrapped.split('\n')):
                    draw.text((label_x, ai_text_y + i * line_height), line, fill=(0, 0, 0), font=font_small)
                
            except Exception as e:
                print(f"  Error: {e}")
        
        canvas.save(os.path.join(OUTPUT_DIR, output_name))
        print(f"  Saved {output_name}")
    
    create_labeled_montage(fis_csv, FOODINSSEG_DIR, 'fig11_foodinsseg_montage.png', 
                          'FoodSeg103 Cross-Dataset Validation',
                          true_col='true_labels', pred_col='pred_labels',
                          img_col='file_name')
    
    create_labeled_montage(r1m_csv, RECIPE1M_DIR, 'fig11_recipe1m_montage.png',
                          'Recipe1M Cross-Dataset Validation', 
                          true_col='true_ingredients', pred_col='pred_ingredients',
                          img_col='image_id')


def fig12_component_count_comparison(data):
    """
    Fig 12: Compare mean ingredient Jaccard by component count across datasets
    (Nutrition5k vs FoodSeg103) for Gemini 2.0 Flash, with bootstrap 95% CI
    and sample-size annotations.
    """
    print("\nGenerating Fig 12 (Component Count Comparison)...")

    n5k_path = os.path.join(RESULTS_DIR, 'gemini_2.0_flash_P3_full3229.csv')
    fis_path = os.path.join(RESULTS_DIR, 'foodinsseg', 'foodinsseg_gemini_visible_ingredients.csv')
    if not os.path.exists(n5k_path) or not os.path.exists(fis_path):
        print("  Missing CSV(s) for Fig 12")
        return

    n5k = pd.read_csv(n5k_path)
    n5k['ingredient_jaccard'] = pd.to_numeric(n5k.get('ingredient_jaccard'), errors='coerce')
    n5k['n_components'] = n5k.get('true_ingredients', '').fillna('').apply(count_ingredients)

    fis = pd.read_csv(fis_path)
    fis['jaccard'] = pd.to_numeric(fis.get('jaccard'), errors='coerce')

    def _count_labels(val):
        if pd.isna(val) or str(val).strip() == '':
            return 0
        parts = [p.strip() for p in str(val).split(',') if p.strip()]
        return len(parts)

    fis['n_components'] = fis.get('true_labels', '').fillna('').apply(_count_labels)

    def _bucket(n):
        if n <= 1:
            return '1'
        if n == 2:
            return '2'
        if n == 3:
            return '3'
        return '4+'

    n5k['bucket'] = n5k['n_components'].apply(_bucket)
    fis['bucket'] = fis['n_components'].apply(_bucket)

    order = ['1', '2', '3', '4+']

    def _boot_ci(vals, n_boot=500, seed=42):
        vals = np.asarray(vals, dtype=float)
        vals = vals[~np.isnan(vals)]
        if len(vals) < 3:
            return (np.nan, np.nan, np.nan)
        rng = np.random.RandomState(seed)
        means = []
        n = len(vals)
        for _ in range(n_boot):
            idx = rng.randint(0, n, size=n)
            means.append(float(np.mean(vals[idx])))
        return (float(np.mean(vals)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))

    summary_rows = []
    for dataset, df, col in [
        ('Nutrition5k', n5k, 'ingredient_jaccard'),
        ('FoodSeg103', fis, 'jaccard'),
    ]:
        for b in order:
            vals = df[df['bucket'] == b][col].values
            mean, lo, hi = _boot_ci(vals)
            summary_rows.append({
                'dataset': dataset,
                'bucket': b,
                'mean': mean,
                'lo': lo,
                'hi': hi,
                'n': int(np.sum(~np.isnan(vals))),
            })

    summary = pd.DataFrame(summary_rows)

    fig, ax = plt.subplots(figsize=(10, 6.5))
    x = np.arange(len(order))
    offset = 0.12
    colors = {'Nutrition5k': '#4C78A8', 'FoodSeg103': '#F58518'}

    for dataset, dx in [('Nutrition5k', -offset), ('FoodSeg103', offset)]:
        sub = summary[summary['dataset'] == dataset].set_index('bucket').reindex(order)
        means = sub['mean'].values
        lows = sub['lo'].values
        highs = sub['hi'].values
        yerr = np.vstack([means - lows, highs - means])
        ax.errorbar(
            x + dx,
            means,
            yerr=yerr,
            fmt='o',
            capsize=3,
            color=colors[dataset],
            label=dataset,
        )

    # Add sample sizes to tick labels
    n5k_counts = summary[summary['dataset'] == 'Nutrition5k'].set_index('bucket')['n']
    fis_counts = summary[summary['dataset'] == 'FoodSeg103'].set_index('bucket')['n']
    xticks = [
        f"{b}\n(n5k={n5k_counts.get(b, 0)}, fis={fis_counts.get(b, 0)})"
        for b in order
    ]

    ax.set_xticks(x)
    ax.set_xticklabels(xticks)
    ax.set_ylabel('Mean Ingredient Jaccard')
    ax.set_xlabel('Component Count')
    ax.set_ylim(0, 1)
    ax.set_title('Jaccard by Component Count (Gemini 2.0 Flash)')
    ax.legend(frameon=False, ncol=2, loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    save_fig(fig, os.path.join(OUTPUT_DIR, 'fig12_component_count.png'))
    plt.close()
    print("  Saved fig12_component_count.png")


def fig12_component_count_boxplot():
    """
    Fig 12 (alternative): Boxplot comparison of Jaccard by component count
    for Nutrition5k vs FoodSeg103 (Gemini 2.0 Flash).
    """
    print("\nGenerating Fig 12 (Component Count Boxplot)...")

    n5k_path = os.path.join(RESULTS_DIR, 'gemini_2.0_flash_P3_full3229.csv')
    fis_path = os.path.join(RESULTS_DIR, 'foodinsseg', 'foodinsseg_gemini_visible_ingredients.csv')
    if not os.path.exists(n5k_path) or not os.path.exists(fis_path):
        print("  Missing CSV(s) for Fig 12 boxplot")
        return

    n5k = pd.read_csv(n5k_path)
    n5k['ingredient_jaccard'] = pd.to_numeric(n5k.get('ingredient_jaccard'), errors='coerce')
    n5k['n_components'] = n5k.get('true_ingredients', '').fillna('').apply(count_ingredients)

    fis = pd.read_csv(fis_path)
    fis['jaccard'] = pd.to_numeric(fis.get('jaccard'), errors='coerce')

    def _count_labels(val):
        if pd.isna(val) or str(val).strip() == '':
            return 0
        parts = [p.strip() for p in str(val).split(',') if p.strip()]
        return len(parts)

    fis['n_components'] = fis.get('true_labels', '').fillna('').apply(_count_labels)

    def _bucket(n):
        if n <= 1:
            return '1'
        if n == 2:
            return '2'
        if n == 3:
            return '3'
        return '4+'

    n5k['bucket'] = n5k['n_components'].apply(_bucket)
    fis['bucket'] = fis['n_components'].apply(_bucket)

    order = ['1', '2', '3', '4+']
    colors = {'Nutrition5k': '#4C78A8', 'FoodSeg103': '#F58518'}

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(order))
    offset = 0.18

    box_data = []
    positions = []
    box_colors = []

    for i, b in enumerate(order):
        for dataset, df, col, dx in [
            ('Nutrition5k', n5k, 'ingredient_jaccard', -offset),
            ('FoodSeg103', fis, 'jaccard', offset),
        ]:
            vals = df[df['bucket'] == b][col].values
            vals = vals[~np.isnan(vals)]
            box_data.append(vals)
            positions.append(i + dx)
            box_colors.append(colors[dataset])

    parts = ax.boxplot(
        box_data,
        positions=positions,
        widths=0.30,
        patch_artist=True,
        showfliers=False,
    )

    border_color = '#1F4E79'  # dark blue

    for i, box in enumerate(parts['boxes']):
        box.set_facecolor(box_colors[i])
        box.set_edgecolor(border_color)
        box.set_linewidth(lw(1.0))
        box.set_alpha(0.7)
        box.set_zorder(2)

    for key in ['whiskers', 'caps', 'medians']:
        for item in parts[key]:
            if key in {'whiskers', 'caps'}:
                item.set_color(border_color)
            else:
                item.set_color('black')
            item.set_linewidth(lw(1.0))
            item.set_zorder(5)
            item.set_clip_on(False)

    # Add sample sizes to tick labels
    n5k_counts = n5k.groupby('bucket')['ingredient_jaccard'].apply(lambda s: int(np.sum(~np.isnan(s)))).reindex(order)
    fis_counts = fis.groupby('bucket')['jaccard'].apply(lambda s: int(np.sum(~np.isnan(s)))).reindex(order)
    xticks = [
        f"{b}\n(n5k={int(n5k_counts.get(b, 0))})\n(fis={int(fis_counts.get(b, 0))})"
        for b in order
    ]

    ax.set_xticks(x)
    ax.set_xticklabels(xticks, fontsize=fs(8))
    ax.set_ylabel('Ingredient Jaccard')
    ax.set_xlabel('Component Count')
    ax.set_ylim(0, 1)
    ax.set_title('Jaccard by Component Count')
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)

    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=colors['Nutrition5k'], edgecolor='black', label='Nutrition5k', alpha=0.7),
        Patch(facecolor=colors['FoodSeg103'], edgecolor='black', label='FoodSeg103', alpha=0.7),
    ]
    fig.legend(
        handles=handles,
        frameon=False,
        ncol=2,
        loc='lower center',
        bbox_to_anchor=(0.5, 0.02),
    )

    fig.subplots_adjust(bottom=0.30)
    save_fig(fig, os.path.join(OUTPUT_DIR, 'fig12_component_count_boxplot.png'))
    plt.close()
    print("  Saved fig12_component_count_boxplot.png")


def fig6_component_count_radar_variant(data):
    """
    Fig 6 variant: Left panel shows mean Jaccard by model for Nutrition5k
    (Gemini 2.0 / P3 runs), right panel retains the radar plot from Fig 6.
    """
    print("\nGenerating Fig 6 Variant (Model Jaccard + Radar)...")

    def _boot_ci(vals, n_boot=500, seed=42):
        vals = np.asarray(vals, dtype=float)
        vals = vals[~np.isnan(vals)]
        if len(vals) < 3:
            return (np.nan, np.nan, np.nan)
        rng = np.random.RandomState(seed)
        means = []
        n = len(vals)
        for _ in range(n_boot):
            idx = rng.randint(0, n, size=n)
            means.append(float(np.mean(vals[idx])))
        return (float(np.mean(vals)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))

    fig = plt.figure(figsize=(16, 8))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 1])

    # Left panel: mean Jaccard by model
    ax_left = fig.add_subplot(gs[0])
    model_names = []
    means = []
    lows = []
    highs = []
    colors = []

    for model in MODEL_ORDER:
        if model not in data:
            continue
        dfm = data[model]
        jvals = pd.to_numeric(dfm.get('ingredient_jaccard'), errors='coerce').dropna().values
        mean, lo, hi = _boot_ci(jvals)
        model_names.append(model)
        means.append(mean)
        lows.append(lo)
        highs.append(hi)
        colors.append(MODEL_COLORS[model])

    x = np.arange(len(model_names))
    yerr = np.vstack([np.array(means) - np.array(lows), np.array(highs) - np.array(means)])
    ax_left.errorbar(x, means, yerr=yerr, fmt='o', capsize=3, color='black')
    ax_left.scatter(x, means, c=colors, s=30, zorder=3)

    ax_left.set_xticks(x)
    ax_left.set_xticklabels(model_names, rotation=45, ha='right')
    ax_left.set_ylabel('Mean Ingredient Jaccard')
    ax_left.set_ylim(0, 1)
    ax_left.set_title('A. Jaccard by Model (Nutrition5k, P3)')
    ax_left.grid(axis='y', alpha=0.3)
    ax_left.set_axisbelow(True)

    # Right panel: radar (reuse from fig6)
    ax_radar = fig.add_subplot(gs[1], polar=True)
    metrics_names = ['Cal CCC', 'Wt CCC', 'Within-25%', 'Jaccard', 'Cost Eff']
    num_vars = len(metrics_names)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    costs = {'Gemini 2.0': 0.10, 'Gemini 2.5': 0.55, 'GPT-4o': 2.00,
             'GPT-4o-mini': 0.15, 'Qwen2-VL-7B': 0.01, 'FatSecret': 0.50}

    for model in MODEL_ORDER:
        if model not in data:
            continue
        dfm = data[model]

        df_cal = dfm.dropna(subset=['true_calories', 'ai_calories'])
        df_cal = df_cal[(df_cal['ai_calories'] > 0) & (df_cal['true_calories'] <= 1500) & (df_cal['ai_calories'] <= 1500)]
        cal_ccc = calculate_ccc(df_cal['true_calories'], df_cal['ai_calories'])

        df_wt = dfm.dropna(subset=['true_weight', 'ai_weight'])
        df_wt = df_wt[(df_wt['ai_weight'] > 0) & (df_wt['true_weight'] <= 1500) & (df_wt['ai_weight'] <= 1500)]
        wt_ccc = calculate_ccc(df_wt['true_weight'], df_wt['ai_weight'])

        within_25 = 0
        if len(df_cal) > 0:
            within_25 = (np.abs(df_cal['ai_calories'] - df_cal['true_calories']) / df_cal['true_calories'] <= 0.25).mean()

        jaccard = dfm['ingredient_jaccard'].mean()
        cost_eff = min(1 / (costs.get(model, 1) + 0.01) / 10, 1)

        values = [
            cal_ccc if not np.isnan(cal_ccc) else 0,
            wt_ccc if not np.isnan(wt_ccc) else 0,
            within_25,
            jaccard if not np.isnan(jaccard) else 0,
            cost_eff
        ]
        values += values[:1]

        ax_radar.plot(angles, values, 'o-', linewidth=lw(1.0), label=model,
                      color=MODEL_COLORS[model], markersize=3)
        ax_radar.fill(angles, values, alpha=0.05, color=MODEL_COLORS[model])

    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(metrics_names, fontsize=fs(9))
    ax_radar.set_ylim(0, 1.15)
    ax_radar.tick_params(pad=fs(8))
    ax_radar.set_title('B. Performance Radar')

    # Single-row legend beneath both panels
    handles, labels = ax_radar.get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc='lower center',
            bbox_to_anchor=(0.5, 0.02),
            ncol=max(1, len(labels)),
            fontsize=fs(11),
            frameon=False,
        )

    plt.tight_layout(rect=[0, 0.10, 1, 1])
    save_fig(fig, os.path.join(OUTPUT_DIR, 'fig6_component_count_radar.png'))
    plt.close()
    print("  Saved fig6_component_count_radar.png")


def fig_grant_constellate():
    """
    Grant figure: Constellate VLM-powered food recognition pipeline + performance.

    Layout (2 rows):
      Top    (full width): Pipeline flow — Montage → Gemini 2.0 Flash → Structured output
      Bottom (two panels): Weight scatter plot  |  Gemini 2.0 vs FatSecret comparison
    """
    print("\nGenerating Grant Figure (Constellate Pipeline)...")

    # ── helpers ───────────────────────────────────────────────────
    def _clean(df_in):
        d = df_in.copy()
        for c in ['ai_calories', 'ai_weight', 'true_calories', 'true_weight',
                   'ingredient_jaccard']:
            if c in d.columns:
                d[c] = pd.to_numeric(d[c], errors='coerce')
        return d

    def _metrics(d):
        dc = d.dropna(subset=['true_calories', 'ai_calories']).copy()
        dc = dc[(dc['true_calories'] > 0) & (dc['ai_calories'] > 0)]
        dc = dc[(dc['true_calories'] <= 1500) & (dc['ai_calories'] <= 1500)]
        dw = d.dropna(subset=['true_weight', 'ai_weight']).copy()
        dw = dw[(dw['true_weight'] > 0) & (dw['ai_weight'] > 0)]
        dw = dw[(dw['true_weight'] <= 1200) & (dw['ai_weight'] <= 1200)]
        cal_ccc = calculate_ccc(dc['true_calories'].values,
                                dc['ai_calories'].values) if len(dc) else 0
        wt_ccc  = calculate_ccc(dw['true_weight'].values,
                                dw['ai_weight'].values) if len(dw) else 0
        cal_mae = float(np.mean(np.abs(
            dc['ai_calories'].values - dc['true_calories'].values))) if len(dc) else 0
        wt_mae  = float(np.mean(np.abs(
            dw['ai_weight'].values - dw['true_weight'].values))) if len(dw) else 0
        jac = float(d['ingredient_jaccard'].dropna().mean()) \
            if 'ingredient_jaccard' in d.columns and d['ingredient_jaccard'].notna().any() else 0
        w25 = float(np.mean(
            np.abs(dc['ai_calories'].values - dc['true_calories'].values)
            / dc['true_calories'].values <= 0.25
        ) * 100) if len(dc) else 0
        return dict(cal_ccc=cal_ccc, wt_ccc=wt_ccc, cal_mae=cal_mae,
                    wt_mae=wt_mae, jaccard=jac, within25=w25,
                    df_cal=dc, df_wt=dw)

    # ── Load Gemini 2.0 Flash ────────────────────────────────────
    gem_path = os.path.join(RESULTS_DIR, 'gemini_2.0_flash_P3_full3229.csv')
    if not os.path.exists(gem_path):
        print(f"  CSV not found: {gem_path}")
        return
    df_gem = _clean(pd.read_csv(gem_path))
    m_gem = _metrics(df_gem)

    # ── Load FatSecret ───────────────────────────────────────────
    fs_path = os.path.join(RESULTS_DIR, 'fatsecret_full3229.csv')
    if not os.path.exists(fs_path):
        print(f"  FatSecret CSV not found: {fs_path}")
        return
    df_fs = _clean(pd.read_csv(fs_path))
    m_fs = _metrics(df_fs)

    print(f"  Gemini 2.0 — Cal CCC={m_gem['cal_ccc']:.3f}, Wt CCC={m_gem['wt_ccc']:.3f}, "
          f"Jaccard={m_gem['jaccard']:.3f}, +/-25%={m_gem['within25']:.1f}%")
    print(f"  FatSecret  — Cal CCC={m_fs['cal_ccc']:.3f}, Wt CCC={m_fs['wt_ccc']:.3f}, "
          f"Jaccard={m_fs['jaccard']:.3f}, +/-25%={m_fs['within25']:.1f}%")

    # ── Build 1x4 montage — dishes with 3+ ingredients ──────────
    # Filter for dishes with 3+ pipe-separated ingredients
    multi_ing = df_gem[
        df_gem['true_ingredients'].notna() &
        (df_gem['true_ingredients'].str.count(r'\|') >= 2)   # 2 pipes = 3+ items
    ]['dish_id'].astype(str).tolist()
    random.shuffle(multi_ing)

    img_paths = []
    for did in multi_ing:
        rgb = os.path.join(NUTRITION5K_DIR, did, 'rgb.png')
        if os.path.exists(rgb):
            img_paths.append(rgb)
        if len(img_paths) >= 3:
            break

    cell = 200
    pad_px = 5
    m_cols, m_rows = 3, 1
    m_w = m_cols * cell + (m_cols + 1) * pad_px
    m_h = m_rows * cell + (m_rows + 1) * pad_px
    montage = Image.new('RGB', (m_w, m_h), (255, 255, 255))
    for idx, ip in enumerate(img_paths):
        c_idx = idx % m_cols
        try:
            im = Image.open(ip).convert('RGB')
            w, h = im.size
            mn = min(w, h)
            im = im.crop(((w - mn) // 2, (h - mn) // 2,
                           (w + mn) // 2, (h + mn) // 2))
            im = im.resize((cell, cell), Image.LANCZOS)
            x0 = pad_px + c_idx * (cell + pad_px)
            y0 = pad_px
            montage.paste(im, (x0, y0))
        except Exception:
            pass

    # ── Color palette ────────────────────────────────────────────
    BLUE      = '#4285F4'
    GREEN     = '#34A853'
    PURPLE    = '#9B59B6'
    BG_BLUE   = '#E8F0FE'
    BG_GREEN  = '#E6F4EA'
    BG_GRAY   = '#F8F9FA'
    DARK      = '#202124'
    SECONDARY = '#5F6368'
    BORDER    = '#DADCE0'

    # ── Figure layout ────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 10.5), facecolor='white')
    gs_main = GridSpec(2, 2, figure=fig,
                       height_ratios=[0.38, 0.62],
                       wspace=0.24, hspace=0.35)

    # ==========================================================
    # PANEL A — Pipeline flow (top row, full width)
    # ==========================================================
    ax_a = fig.add_subplot(gs_main[0, :])
    ax_a.set_xlim(0, 1)
    ax_a.set_ylim(0, 1)
    ax_a.axis('off')
    # Title — same fs(11) as B and C
    ax_a.set_title('A.  Nutrition5k Food Recognition Pipeline Validation Study',
                    fontsize=fs(11), loc='left', pad=15)

    # Box geometry — generous margins from figure edges
    B1_X, B1_Y, B1_W, B1_H = 0.03, 0.05, 0.28, 0.85
    B2_X, B2_Y, B2_W, B2_H = 0.37, 0.05, 0.24, 0.85
    B3_X, B3_Y, B3_W, B3_H = 0.67, 0.05, 0.28, 0.85

    # ---- Box 1: Montage ----
    box1 = mpatches.FancyBboxPatch(
        (B1_X, B1_Y), B1_W, B1_H,
        boxstyle="round,pad=0.02", facecolor=BG_GRAY,
        edgecolor=BORDER, linewidth=lw(1.5),
        transform=ax_a.transAxes, zorder=1)
    ax_a.add_patch(box1)

    # Two-line label
    ax_a.text(B1_X + B1_W / 2, B1_Y + B1_H - 0.04,
              'Input: 3,226 Nutrition5k\ncafeteria images',
              transform=ax_a.transAxes, fontsize=fs(10),
              ha='center', va='top', color=SECONDARY, linespacing=1.3)

    # 1x4 montage inset — wide and centered vertically
    inset_img = ax_a.inset_axes([B1_X + 0.02, B1_Y + 0.08,
                                  B1_W - 0.04, B1_H - 0.32])
    inset_img.imshow(montage)
    inset_img.axis('off')

    # ---- Arrow 1 ----
    ax_a.annotate('', xy=(B2_X - 0.015, 0.47),
                  xytext=(B1_X + B1_W + 0.015, 0.47),
                  xycoords='axes fraction', textcoords='axes fraction',
                  arrowprops=dict(arrowstyle='->', color=BLUE,
                                  lw=lw(2.5), mutation_scale=22))

    # ---- Box 2: Model ----
    box2 = mpatches.FancyBboxPatch(
        (B2_X, B2_Y), B2_W, B2_H,
        boxstyle="round,pad=0.02", facecolor=BG_BLUE,
        edgecolor=BLUE, linewidth=lw(2),
        transform=ax_a.transAxes, zorder=1)
    ax_a.add_patch(box2)

    # Model name — single line
    ax_a.text(B2_X + B2_W / 2, B2_Y + B2_H / 2 + 0.16,
              'Gemini 2.0 Flash',
              transform=ax_a.transAxes, fontsize=fs(13),
              ha='center', va='center', color=BLUE)

    # Description with more spacing below name
    ax_a.text(B2_X + B2_W / 2, B2_Y + B2_H / 2 - 0.04,
              'Vision-language model\npaired with special prompt',
              transform=ax_a.transAxes, fontsize=fs(9),
              ha='center', va='center', color=SECONDARY,
              fontstyle='italic', linespacing=1.4)

    # Cost badge
    ax_a.text(B2_X + B2_W / 2, B2_Y + 0.10,
              '$0.10 / 1,000 images',
              transform=ax_a.transAxes, fontsize=fs(9),
              ha='center', va='center', color=GREEN,
              bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                        edgecolor=GREEN, linewidth=lw(1)))

    # ---- Arrow 2 ----
    ax_a.annotate('', xy=(B3_X - 0.015, 0.47),
                  xytext=(B2_X + B2_W + 0.015, 0.47),
                  xycoords='axes fraction', textcoords='axes fraction',
                  arrowprops=dict(arrowstyle='->', color=GREEN,
                                  lw=lw(2.5), mutation_scale=22))

    # ---- Box 3: Structured Output (narrower) ----
    box3 = mpatches.FancyBboxPatch(
        (B3_X, B3_Y), B3_W, B3_H,
        boxstyle="round,pad=0.02", facecolor=BG_GREEN,
        edgecolor=GREEN, linewidth=lw(2),
        transform=ax_a.transAxes, zorder=1)
    ax_a.add_patch(box3)

    ax_a.text(B3_X + B3_W / 2, B3_Y + B3_H - 0.04,
              'Structured Output',
              transform=ax_a.transAxes, fontsize=fs(10),
              ha='center', va='top', color='#1B5E20')

    json_output = (
        '{\n'
        ' "items": [\n'
        '  {"chicken",\n'
        '   165 kcal, 120g},\n'
        '  {"rice",\n'
        '   130 kcal, 100g}\n'
        ' ]\n'
        '}'
    )
    ax_a.text(B3_X + B3_W / 2, B3_Y + B3_H / 2 - 0.06,
              json_output,
              transform=ax_a.transAxes, fontsize=fs(8.5),
              ha='center', va='center', color='#1B5E20',
              fontfamily='monospace', linespacing=1.3)

    # ==========================================================
    # PANEL B — Weight Scatter Plot (bottom-left)
    # ==========================================================
    dw = m_gem['df_wt']
    ax_b = fig.add_subplot(gs_main[1, 0])
    ax_b.set_title('B.  Weight Estimation Accuracy',
                    fontsize=fs(11), loc='left', pad=10)

    x_vals = dw['true_weight'].values
    y_vals = dw['ai_weight'].values

    ax_b.scatter(x_vals, y_vals, c=BLUE, s=5, alpha=0.35,
                 linewidths=0, rasterized=True)
    ax_b.plot([0, 1200], [0, 1200], color='#333', linestyle='--',
              alpha=0.6, linewidth=lw(1))

    ax_b.set_xlim(0, 1200)
    ax_b.set_ylim(0, 1200)
    ax_b.set_xlabel('True Weight (g)')
    ax_b.set_ylabel('Predicted Weight (g)')
    ax_b.set_aspect('equal')

    stats_box = (
        f'CCC = {m_gem["wt_ccc"]:.3f}\n'
        f'MAE = {m_gem["wt_mae"]:.0f} g\n'
        f'n = {len(dw):,}'
    )
    ax_b.text(0.04, 0.96, stats_box, transform=ax_b.transAxes,
              fontsize=fs(9), va='top', ha='left',
              bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                        alpha=0.92, edgecolor='#E0E0E0', linewidth=lw(0.8)))

    ax_b.grid(alpha=0.15, linestyle='-')
    ax_b.set_axisbelow(True)
    # (no legend needed)

    # ==========================================================
    # PANEL C — Gemini 2.0 Flash vs FatSecret comparison
    # ==========================================================
    ax_c = fig.add_subplot(gs_main[1, 1])
    ax_c.set_title('C.  Comparison vs. FatSecret (Commercial API)',
                    fontsize=fs(11), loc='left', pad=10)

    metric_labels = ['Calorie\nCCC', 'Weight\nCCC', 'Ingredient\nJaccard']
    gem_vals = [m_gem['cal_ccc'], m_gem['wt_ccc'], m_gem['jaccard']]
    fs_vals  = [m_fs['cal_ccc'],  m_fs['wt_ccc'],  m_fs['jaccard']]

    y_pos = np.arange(len(metric_labels))
    bar_h = 0.35

    bars_gem = ax_c.barh(y_pos - bar_h / 2, gem_vals, height=bar_h,
                          color=BLUE, alpha=0.85,
                          label='Constellate (Gemini 2.0 Flash)',
                          edgecolor='white', linewidth=lw(0.5))
    bars_fs  = ax_c.barh(y_pos + bar_h / 2, fs_vals, height=bar_h,
                          color=PURPLE, alpha=0.70, label='FatSecret API',
                          edgecolor='white', linewidth=lw(0.5))

    # Value labels
    for bars, vals in [(bars_gem, gem_vals), (bars_fs, fs_vals)]:
        for bar, val in zip(bars, vals):
            x_pos = max(val + 0.015, 0.04)
            ax_c.text(x_pos, bar.get_y() + bar.get_height() / 2,
                      f'{val:.3f}', va='center', ha='left',
                      fontsize=fs(9), color=DARK)

    ax_c.set_yticks(y_pos)
    ax_c.set_yticklabels(metric_labels, fontsize=fs(10))
    ax_c.set_xlim(0, 1.05)
    ax_c.set_xlabel('Score', labelpad=12)
    ax_c.invert_yaxis()
    ax_c.grid(axis='x', alpha=0.15, linestyle='-')
    ax_c.set_axisbelow(True)
    ax_c.spines['top'].set_visible(False)
    ax_c.spines['right'].set_visible(False)

    # Legend below the bars, not overlapping them
    ax_c.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18),
                fontsize=fs(10), frameon=False, ncol=2)

    # ── Save ─────────────────────────────────────────────────────
    plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    save_fig(fig, os.path.join(OUTPUT_DIR, 'fig_grant_constellate.png'),
             facecolor='white')
    plt.close()
    print("  Saved fig_grant_constellate.png")


def main():
    print("="*60)
    print("GENERATING ALL PAPER FIGURES")
    print("="*60)
    print(f"Output: {OUTPUT_DIR}")
    
    print("\nLoading data...")
    data = load_data()
    
    fig1_overview()
    fig2_prompt_ablation()
    fig3_scatter_calories(data)
    fig4_scatter_weight(data)
    fig5_model_comparison(data)
    fig6_violin_radar(data)
    fig7_complexity_heatmap(data)
    fig8_jaccard_scatter(data)
    fig9_low_jaccard_montage(data)
    fig10_confusion_matrix()
    fig11_cross_dataset_montages()
    fig12_component_count_comparison(data)
    fig12_component_count_boxplot()
    fig_grant_constellate()
    
    print("\n" + "="*60)
    print("ALL FIGURES GENERATED")
    print(f"Output: {OUTPUT_DIR}")
    print("="*60)


if __name__ == '__main__':
    main()
