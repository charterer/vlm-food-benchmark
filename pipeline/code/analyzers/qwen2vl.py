"""Qwen2-VL analyzer (open-source VLM, runs locally)."""

import io
import json
import re
import time
from typing import Dict, Any, Optional

from .base import FoodAnalyzer
from .prompting import build_canonical_prompt


class Qwen2VLAnalyzer(FoodAnalyzer):
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-7B-Instruct",
        device: str = "auto",
        prompt_variant: Optional[str] = None,
    ):
        self.model_name = model_name
        self.device = device
        self.prompt_variant = prompt_variant
        self.model = None
        self.processor = None
        self._loaded = False

    @staticmethod
    def _patch_torch_compiler() -> None:
        # Some torch builds (e.g. 2.2.x) ship torch.compiler without is_compiling,
        # which newer transformers stacks call during generation.
        try:
            import torch
        except Exception:
            return
        if not hasattr(torch, "compiler") or hasattr(torch.compiler, "is_compiling"):
            return

        def _is_compiling() -> bool:
            try:
                if hasattr(torch, "_dynamo") and hasattr(torch._dynamo, "is_compiling"):
                    return bool(torch._dynamo.is_compiling())
            except Exception:
                pass
            return False

        try:
            torch.compiler.is_compiling = _is_compiling  # type: ignore[attr-defined]
        except Exception:
            pass

    def _ensure_loaded(self):
        if self._loaded:
            return

        import torch
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

        self._patch_torch_compiler()

        print(f"Loading Qwen2-VL model: {self.model_name}")
        start = time.time()

        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            low_cpu_mem_usage=True,
        )
        if self.device == "cpu":
            self.model = self.model.to("cpu")

        self._loaded = True
        print(f"Model loaded in {time.time() - start:.1f}s on {self.device}")

    def analyze(self, image_bytes: bytes, user_hint: Optional[str] = None) -> Dict[str, Any]:
        import torch
        from PIL import Image

        self._patch_torch_compiler()
        self._ensure_loaded()
        start_time = time.time()

        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": build_canonical_prompt(user_hint=user_hint, variant=self.prompt_variant)},
                    ],
                }
            ]

            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(
                text=[text],
                images=[image],
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                output_ids = self.model.generate(**inputs, max_new_tokens=512, do_sample=False)

            generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
            response = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

            analysis_data = self._parse_response(response)
            elapsed = time.time() - start_time

            items = []
            for item in analysis_data.get("items", []):
                if isinstance(item, dict):
                    items.append({
                        "name": item.get("name", "unknown"),
                        "estimated_calories": item.get("estimated_calories", item.get("calories", 0)),
                        "estimated_weight_grams": item.get("estimated_weight_grams", item.get("weight_grams", 0)),
                    })

            return {
                "analysis": {
                    "general_description": analysis_data.get("general_description", analysis_data.get("description", response[:200])),
                    "estimated_calories": analysis_data.get("estimated_calories", analysis_data.get("total_calories", 0)),
                    "estimated_weight": analysis_data.get("estimated_weight", analysis_data.get("total_weight_grams", 0)),
                    "items": items,
                },
                "model": self.model_name,
                "inference_time_seconds": elapsed,
                "raw_response": response[:500],
            }

        except Exception as e:
            return {
                "analysis": {
                    "general_description": f"Error: {e}",
                    "estimated_calories": 0,
                    "estimated_weight": 0,
                    "items": [],
                },
                "model": self.model_name,
                "inference_time_seconds": time.time() - start_time,
                "error": str(e),
            }

    def _parse_response(self, response: str) -> Dict[str, Any]:
        m = re.search(r"\{[\s\S]*\}", response)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass

        cal = re.search(r"(\d+)\s*(?:calories|kcal|cal)", response, re.IGNORECASE)
        wt = re.search(r"(\d+)\s*(?:grams?|g\b)", response, re.IGNORECASE)
        return {
            "general_description": response[:200] if response else "Unable to parse response",
            "items": [],
            "estimated_calories": int(cal.group(1)) if cal else 0,
            "estimated_weight": int(wt.group(1)) if wt else 0,
        }
