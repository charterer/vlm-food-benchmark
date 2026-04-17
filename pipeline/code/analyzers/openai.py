from typing import Dict, Any, Optional
import base64
import json
import re
import random
import time
import requests
from .base import FoodAnalyzer
from .prompting import build_canonical_prompt

class OpenAIAnalyzer(FoodAnalyzer):
    def __init__(self, api_key: str, model_name: str = "gpt-4o", prompt_variant: Optional[str] = None):
        self.api_key = api_key
        self.model_name = model_name
        self.prompt_variant = prompt_variant
        self.url = "https://api.openai.com/v1/chat/completions"

    def analyze(self, image_bytes: bytes, user_hint: Optional[str] = None) -> Dict[str, Any]:
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        system_prompt = build_canonical_prompt(user_hint=user_hint, variant=self.prompt_variant)

        user_content = [
            {"type": "text", "text": "Analyze this meal."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
        ]

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "response_format": {"type": "json_object"},
        }

        # GPT-5 / reasoning models use max_completion_tokens instead of max_tokens.
        if self.model_name.startswith(("gpt-5", "o1", "o3", "o4")):
            payload["max_completion_tokens"] = 1000
        else:
            payload["max_tokens"] = 1000

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        try:
            # Retry on rate limits / transient issues with aggressive backoff
            max_attempts = 8
            last_err: Optional[Exception] = None
            for attempt in range(1, max_attempts + 1):
                try:
                    response = requests.post(self.url, headers=headers, json=payload, timeout=120)
                    # Handle rate limiting explicitly
                    if response.status_code == 429:
                        retry_after = response.headers.get("Retry-After")
                        # More aggressive backoff: start at 10s, double each time
                        base = float(retry_after) if retry_after else min(120.0, 10.0 * (2.0 ** (attempt - 1)))
                        sleep_s = base + random.random() * 2.0
                        print(f"[OpenAIAnalyzer] 429 rate limit. Sleeping {sleep_s:.1f}s before retry {attempt}/{max_attempts}")
                        time.sleep(sleep_s)
                        raise RuntimeError(f"429 Too Many Requests (sleep {sleep_s:.1f}s)")
                    # Retry on transient server-side failures
                    if 500 <= response.status_code < 600:
                        raise RuntimeError(f"{response.status_code} Server Error: {response.text[:500]}")
                    response.raise_for_status()
                    try:
                        result = response.json()
                    except Exception as e:
                        # Sometimes the API returns an empty/non-JSON body (network/gateway). Treat as retryable.
                        raise RuntimeError(f"Invalid JSON response body (status {response.status_code})") from e

                    try:
                        content = result["choices"][0]["message"]["content"]
                    except Exception as e:
                        raise RuntimeError(f"Unexpected OpenAI response schema: {str(result)[:500]}") from e

                    if not isinstance(content, str) or not content.strip():
                        raise RuntimeError("Empty OpenAI message content")

                    try:
                        data = json.loads(content)
                    except Exception:
                        # Defensive: some models occasionally return extra text; attempt to extract the first JSON object.
                        match = re.search(r"\{[\s\S]*\}", content)
                        if match:
                            data = json.loads(match.group(0))
                        else:
                            raise

                    # Normalize keys if GPT varies them
                    items = data.get('items', [])

                    # Calculate totals with robustness check
                    total_cals = 0.0
                    total_wt = 0.0
                    cleaned_items = []
                    for i in items:
                        if isinstance(i, dict):
                            cal = i.get('estimated_calories', 0) or 0
                            wt = i.get('estimated_weight_grams', 0) or 0
                            try:
                                total_cals += float(cal)
                            except (ValueError, TypeError):
                                pass
                            try:
                                total_wt += float(wt)
                            except (ValueError, TypeError):
                                pass
                            cleaned_items.append(i)
                        elif isinstance(i, str):
                            cleaned_items.append({"name": i, "estimated_calories": 0, "estimated_weight_grams": 0})

                    return {
                        "analysis": {
                            "general_description": data.get("general_description", "Analyzed by OpenAI"),
                            "estimated_calories": total_cals,
                            "estimated_weight": total_wt,
                            "items": cleaned_items,
                        }
                    }
                except Exception as e:
                    last_err = e
                    msg = str(e)
                    # Retry on rate limits, transient transport issues, or decode problems
                    if (
                        "429" in msg
                        or "Too Many Requests" in msg
                        or "timed out" in msg.lower()
                        or "timeout" in msg.lower()
                        or "rate" in msg.lower()
                        or "connection aborted" in msg.lower()
                        or "remote end closed" in msg.lower()
                        or "invalid json response body" in msg.lower()
                        or "empty openai message content" in msg.lower()
                        or "expecting value" in msg.lower()
                        or "server error" in msg.lower()
                    ):
                        # More aggressive exponential backoff + jitter
                        sleep_s = min(120.0, 10.0 * (2.0 ** (attempt - 1))) + random.random() * 2.0
                        print(f"[OpenAIAnalyzer] Retry {attempt}/{max_attempts} after error: {e}. Sleeping {sleep_s:.1f}s")
                        time.sleep(sleep_s)
                        continue
                    raise

            # If we exhausted retries
            raise RuntimeError(f"OpenAI request failed after {max_attempts} attempts: {last_err}")

        except Exception as e:
            print(f"[OpenAIAnalyzer] Error: {e}")
            return {
                "analysis": {
                    "items": [],
                    "general_description": f"Error: {str(e)}",
                    "estimated_calories": 0,
                    "estimated_weight": 0,
                }
            }


