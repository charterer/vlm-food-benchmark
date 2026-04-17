"""Prompt variants for cross-provider evaluation.

Keeps the task spec (identify items, estimate grams+calories, return JSON)
consistent across providers. P1-P5 are the ablation variants in the paper;
P6 is an extended app-style prompt not evaluated in the paper.
"""

from typing import Optional


CANONICAL_JSON_SCHEMA_EXAMPLE = """{
  "general_description": "Brief description of the meal",
  "items": [
    {
      "name": "ingredient name",
      "estimated_calories": 123,
      "estimated_weight_grams": 45
    }
  ]
}"""

APP_JSON_SCHEMA_EXAMPLE = """{
  "general_description": "Brief description of the meal",
  "items": [
    {
      "name": "ingredient name",
      "estimated_calories": 123,
      "estimated_weight_grams": 45,
      "confidence": 0.8,
      "assumptions": "Brief assumptions if container size / thickness / occlusion is unclear",
      "hidden_calorie_risk": "low",
      "possible_hidden_items": ["oil", "butter", "sauce"]
    }
  ]
}"""


PROMPT_P1_MINIMAL = """Analyze this food image.
List all visible food items with estimated calories and weight in grams.

Return ONLY valid JSON (no markdown, no extra text) matching this structure:
""" + CANONICAL_JSON_SCHEMA_EXAMPLE

PROMPT_P2_PORTION_FIRST = """You are an expert nutritionist analyzing a food photo.

IMPORTANT: Estimate in this order:
1. First, identify each visible food item
2. Then, estimate the WEIGHT IN GRAMS for each item based on visual portion size
3. Finally, compute calories from those gram estimates

Do NOT estimate calories first and back-fit grams. Grams come first.

Typical portion reference (for sanity-check only):
- Meat/fish: 100-150g
- Vegetables: 50-100g
- Rice/pasta: 150-200g cooked
- Bread slice: 30-40g
- Egg: ~50g each

Return ONLY valid JSON (no markdown, no extra text) matching this structure:
""" + CANONICAL_JSON_SCHEMA_EXAMPLE

PROMPT_P3_ANTI_SNAP = """You are a clinical nutrition assistant. Analyze this meal photo.

CRITICAL INSTRUCTIONS:
1. Estimate the ACTUAL PORTION SIZE visible in the photo - NOT a standard serving
2. Do NOT snap to typical/reference portions
3. Use standard servings ONLY as a sanity-check bound if your estimate is wildly implausible
4. If the photo shows a smaller or larger amount than typical, estimate what you SEE

For each visible food item, estimate:
- Weight in grams (the actual visible amount)
- Calories (based on the actual visible portion)

Return ONLY valid JSON (no markdown, no extra text) matching this structure:
""" + CANONICAL_JSON_SCHEMA_EXAMPLE

PROMPT_P4_CONSTRAINTS = """You are an expert nutritionist analyzing a food photo.

For each distinct food item visible:
1. Identify the food item
2. Estimate its weight in grams
3. Estimate its calories

PLAUSIBILITY CONSTRAINTS (check your estimates):
- All gram values must be positive
- Total meal weight is typically 50-1200g unless clearly a very large serving
- If any single item exceeds 600g, verify it looks that large
- If total calories exceed 2000, double-check portion sizes
- Calories should be consistent with grams (e.g., ~1-3 kcal/g for most foods, ~7-9 for oils/nuts)

Return ONLY valid JSON (no markdown, no extra text) matching this structure:
""" + CANONICAL_JSON_SCHEMA_EXAMPLE

PROMPT_P5_SELF_CHECK = """You are an expert nutritionist analyzing a food photo.

For each distinct food item visible:
1. Identify the food item
2. Estimate its weight in grams based on visual portion size
3. Estimate its calories based on the portion

BEFORE OUTPUTTING: Do a quick self-check:
- Did I miss any visible items?
- Are my gram estimates plausible for what's shown?
- Are calories consistent with grams?
- If anything looks wrong, fix it now.

Return ONLY your final corrected JSON (no markdown, no extra text) matching this structure:
""" + CANONICAL_JSON_SCHEMA_EXAMPLE

PROMPT_P6_APP = """You are a clinical nutrition assistant. Analyze this meal photo and estimate:
1. What food items are visible
2. The ACTUAL PORTION SIZE visible in the photo (not a standard serving)
3. Calories and weight (grams) for the VISIBLE AMOUNT ONLY

CRITICAL: Do NOT use standard reference portions.
Use typical serving sizes ONLY as a sanity-check bound if your estimate is wildly implausible; do not snap to a typical serving when the photo clearly shows a smaller/larger amount.

For each item, include confidence (0-1) and assumptions if container size / thickness / occlusion is unclear.
Also include hidden_calorie_risk (low/medium/high) and possible_hidden_items (e.g., oil, butter, sauce) when relevant.

Return ONLY valid JSON (no markdown, no extra text) matching this structure:
""" + APP_JSON_SCHEMA_EXAMPLE


PROMPT_VARIANTS = {
    "P1": PROMPT_P1_MINIMAL,
    "P2": PROMPT_P2_PORTION_FIRST,
    "P3": PROMPT_P3_ANTI_SNAP,
    "P4": PROMPT_P4_CONSTRAINTS,
    "P5": PROMPT_P5_SELF_CHECK,
    "P6": PROMPT_P6_APP,
}

DEFAULT_VARIANT = "P2"


def build_canonical_prompt(user_hint: Optional[str] = None, variant: Optional[str] = None) -> str:
    variant = (variant or DEFAULT_VARIANT).upper()
    if variant not in PROMPT_VARIANTS:
        raise ValueError(f"Unknown prompt variant: {variant}. Choose from: {list(PROMPT_VARIANTS)}")
    base = PROMPT_VARIANTS[variant]
    if user_hint:
        return f"User hint: {user_hint}\n\n{base}\n\nAnalyze the image:"
    return base + "\n\nAnalyze the image:"


def get_variant_description(variant: str) -> str:
    descriptions = {
        "P1": "Minimal - bare instruction, no guidance",
        "P2": "Portion-first - estimate grams before calories (decomposition)",
        "P3": "Anti-snap - explicit instruction to estimate actual visible portion",
        "P4": "Constraints - explicit plausibility bounds",
        "P5": "Self-check - single-pass verification before output",
        "P6": "App prompt - includes anti-snap + confidence/assumptions + hidden calorie risk",
    }
    return descriptions.get(variant.upper(), "Unknown variant")
