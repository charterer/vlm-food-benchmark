import json
import os
import random
import time
from io import BytesIO
from typing import Any, Dict, Optional

import google.generativeai as genai
from PIL import Image

from .base import FoodAnalyzer
from .prompting import build_canonical_prompt

DEFAULT_MODEL = os.environ.get("GEMINI_FLASH_MODEL", "gemini-2.0-flash")
MAX_PHOTO_DIMENSION = 1200


def _downsample(image: Image.Image, max_dim: int) -> Image.Image:
    if image.width > max_dim or image.height > max_dim:
        image.thumbnail((max_dim, max_dim))
    return image


NUTRITION_ESTIMATION_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "general_description": {"type": "STRING"},
        "items": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "name": {"type": "STRING"},
                    "estimated_calories": {"type": "NUMBER"},
                    "estimated_weight_grams": {"type": "NUMBER"},
                    "confidence": {"type": "NUMBER"},
                    "assumptions": {"type": "STRING"},
                    "hidden_calorie_risk": {"type": "STRING"},
                    "possible_hidden_items": {"type": "ARRAY", "items": {"type": "STRING"}},
                },
                "required": ["name", "estimated_calories", "estimated_weight_grams"],
            },
        },
    },
    "required": ["items", "general_description"],
}


class GeminiAnalyzer(FoodAnalyzer):
    def __init__(self, api_key: str, model_name: str = DEFAULT_MODEL, prompt_variant: Optional[str] = None):
        if not api_key:
            raise ValueError("Gemini API key is required")
        genai.configure(api_key=api_key)
        self.model_name = model_name
        self.api_key = api_key
        self.prompt_variant = prompt_variant

    def analyze(self, image_bytes: bytes, user_hint: Optional[str] = None) -> Dict[str, Any]:
        image_blob = _downsample(Image.open(BytesIO(image_bytes)), MAX_PHOTO_DIMENSION)

        prompt_text = build_canonical_prompt(user_hint=user_hint, variant=self.prompt_variant)

        generation_config = genai.types.GenerationConfig(
            response_mime_type="application/json",
            response_schema=NUTRITION_ESTIMATION_SCHEMA,
        )
        model = genai.GenerativeModel(self.model_name, generation_config=generation_config)

        max_attempts = 6
        last_err = None
        for attempt in range(1, max_attempts + 1):
            try:
                response = model.generate_content([prompt_text, image_blob])
                if not response.text:
                    raise ValueError("Empty response from Gemini")
                analysis_data = json.loads(response.text)
                break
            except Exception as e:
                last_err = e
                err_msg = str(e).lower()
                if "429" in str(e) or "quota" in err_msg or "rate" in err_msg or "resource" in err_msg:
                    sleep_s = min(120.0, 15.0 * (2.0 ** (attempt - 1))) + random.random() * 3.0
                    print(f"[GeminiAnalyzer] Rate limit. Retry {attempt}/{max_attempts} after {sleep_s:.1f}s")
                    time.sleep(sleep_s)
                    continue
                print(f"[GeminiAnalyzer] Error: {e}")
                return {"analysis": {"items": [], "general_description": f"Error: {e}", "estimated_calories": 0}}
        else:
            print(f"[GeminiAnalyzer] Failed after {max_attempts} attempts: {last_err}")
            return {"analysis": {"items": [], "general_description": f"Error: {last_err}", "estimated_calories": 0}}

        items = analysis_data.get("items", [])
        total_cals = 0.0
        total_weight = 0.0
        cleaned_items = []
        for i in items:
            if isinstance(i, dict):
                cal = i.get("estimated_calories", 0) or 0
                wt = i.get("estimated_weight_grams", 0) or 0
                try:
                    total_cals += float(cal)
                    total_weight += float(wt)
                except (ValueError, TypeError):
                    pass
                cleaned_items.append(i)
            elif isinstance(i, str):
                cleaned_items.append({"name": i, "estimated_calories": 0, "estimated_weight_grams": 0})

        return {
            "analysis": {
                "general_description": analysis_data.get("general_description", ""),
                "estimated_calories": total_cals,
                "estimated_weight": total_weight,
                "items": cleaned_items,
            }
        }
