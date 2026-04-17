"""FatSecret image recognition adapter (OAuth 2.0 client credentials)."""

import base64
import json
import os
import time
from typing import Any, Dict, Optional

import requests

from .base import FoodAnalyzer


class FatSecretAnalyzer(FoodAnalyzer):
    TOKEN_URL = "https://oauth.fatsecret.com/connect/token"
    IMAGE_RECOGNITION_URL = "https://platform.fatsecret.com/rest/image-recognition/v2"

    def __init__(self, client_id: str, client_secret: str):
        if not client_id or not client_secret:
            raise ValueError("FatSecret client_id and client_secret are required")
        self.client_id = client_id
        self.client_secret = client_secret
        self.session = requests.Session()
        self._access_token: Optional[str] = None
        self._token_expires_at = 0.0
        # Optional throttle: FATSECRET_MIN_INTERVAL_SECONDS=2 enforces >=2s between calls.
        try:
            self._min_interval_s = float(os.environ.get("FATSECRET_MIN_INTERVAL_SECONDS", "0") or 0)
        except Exception:
            self._min_interval_s = 0.0
        self._last_call_ts = 0.0

    def _get_access_token(self) -> str:
        if self._access_token and time.time() < self._token_expires_at - 60:
            return self._access_token

        resp = self.session.post(
            self.TOKEN_URL,
            data={"grant_type": "client_credentials", "scope": "image-recognition"},
            auth=(self.client_id, self.client_secret),
            timeout=30,
        )
        resp.raise_for_status()
        token_data = resp.json()
        self._access_token = token_data["access_token"]
        expires_in = token_data.get("expires_in", 86400)
        self._token_expires_at = time.time() + expires_in
        return self._access_token

    def analyze(self, image_bytes: bytes, user_hint: Optional[str] = None) -> Dict[str, Any]:
        start_time = time.time()
        food_data: Optional[Dict[str, Any]] = None

        try:
            if self._min_interval_s > 0:
                wait_s = (self._last_call_ts + self._min_interval_s) - time.time()
                if wait_s > 0:
                    time.sleep(wait_s)

            prepared_bytes = self._prepare_image_bytes_for_b64(image_bytes)
            access_token = self._get_access_token()

            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            }
            payload: Dict[str, Any] = {
                "image_b64": base64.b64encode(prepared_bytes).decode("utf-8"),
                "include_food_data": True,
                "region": "US",
                "language": "en",
            }

            max_attempts = int(os.environ.get("FATSECRET_MAX_ATTEMPTS", "3") or 3)
            last_err: Optional[Exception] = None
            for attempt in range(1, max_attempts + 1):
                try:
                    resp = self.session.post(
                        self.IMAGE_RECOGNITION_URL,
                        headers=headers,
                        json=payload,
                        timeout=90,
                    )
                    self._last_call_ts = time.time()
                    resp.raise_for_status()
                    food_data = resp.json()

                    if isinstance(food_data, dict) and "error" in food_data:
                        error_msg = food_data["error"].get("message", "Unknown error")
                        if "too many actions" in str(error_msg).lower():
                            raise RuntimeError(f"FatSecret throttled: {error_msg}")
                        raise ValueError(f"FatSecret API error: {error_msg}")

                    break
                except Exception as e:
                    last_err = e
                    msg = str(e).lower()
                    retryable = (
                        "too many actions" in msg
                        or "throttle" in msg
                        or "rate" in msg
                        or "429" in msg
                        or "timed out" in msg
                        or "timeout" in msg
                        or "connection aborted" in msg
                        or "remote end closed" in msg
                    )
                    if not retryable or attempt == max_attempts:
                        raise
                    time.sleep(min(60.0, 5.0 * (2.0 ** (attempt - 1))))

            if food_data is None:
                raise RuntimeError(f"FatSecret request failed after {max_attempts} attempts: {last_err}")

        except Exception as e:
            return {
                "analysis": {
                    "general_description": f"Error: {e}",
                    "estimated_calories": 0.0,
                    "estimated_weight": 0.0,
                    "items": [],
                },
                "error": str(e),
                "inference_time_seconds": time.time() - start_time,
            }

        items: list = []
        total_calories = 0.0
        total_weight = 0.0

        food_response = food_data.get("food_response", []) if isinstance(food_data, dict) else []
        if isinstance(food_response, dict):
            food_response = [food_response]

        for entry in food_response:
            if not isinstance(entry, dict):
                continue

            name = entry.get("food_entry_name") or entry.get("food_name") or "Unknown"
            eaten = entry.get("eaten") or {}
            tnc = eaten.get("total_nutritional_content") or {}

            try:
                item_cal = float(tnc.get("calories") or 0)
            except (TypeError, ValueError):
                item_cal = 0.0
            try:
                item_wt = float(eaten.get("total_metric_amount") or 0)
            except (TypeError, ValueError):
                item_wt = 0.0

            total_calories += item_cal
            total_weight += item_wt

            items.append({
                "name": str(name),
                "estimated_calories": round(item_cal, 1),
                "estimated_weight_grams": round(item_wt, 1),
            })

        if items:
            names = [item["name"] for item in items]
            description = ", ".join(names[:3])
            if len(items) > 3:
                description += f" and {len(items) - 3} more items"
        else:
            description = "No food items recognized"

        return {
            "analysis": {
                "general_description": description,
                "estimated_calories": round(total_calories, 1),
                "estimated_weight": round(total_weight, 1),
                "items": items,
            },
            "raw_response": json.dumps(food_data)[:1000] if food_data else None,
            "inference_time_seconds": time.time() - start_time,
        }

    @staticmethod
    def _prepare_image_bytes_for_b64(image_bytes: bytes) -> bytes:
        # FatSecret image_b64 must stay under 999,982 chars; base64 expands ~4/3x.
        # Target ~700k bytes before encoding.
        if len(image_bytes) <= 650_000:
            return image_bytes

        try:
            from io import BytesIO
            from PIL import Image
        except Exception:
            return image_bytes

        try:
            img = Image.open(BytesIO(image_bytes)).convert("RGB")
            out = image_bytes
            for max_dim, quality in [(1024, 85), (896, 80), (768, 75), (640, 70), (512, 65)]:
                tmp = img.copy()
                tmp.thumbnail((max_dim, max_dim))
                buf = BytesIO()
                tmp.save(buf, format="JPEG", quality=quality, optimize=True)
                out = buf.getvalue()
                if len(out) <= 700_000:
                    return out
            return out
        except Exception:
            return image_bytes
