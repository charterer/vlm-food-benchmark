from typing import Dict, Any, Optional
import requests
import base64
import json
import re
import random
import time
from .base import FoodAnalyzer
from .prompting import build_canonical_prompt


class AnthropicAnalyzer(FoodAnalyzer):
    def __init__(self, api_key: str, model_name: str = "claude-sonnet-4-20250514", prompt_variant: Optional[str] = None):
        self.api_key = api_key
        self.model_name = model_name
        self.prompt_variant = prompt_variant
        self.url = "https://api.anthropic.com/v1/messages"

    def analyze(self, image_bytes: bytes, user_hint: Optional[str] = None) -> Dict[str, Any]:
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        media_type = "image/png" if image_bytes[:8] == b'\x89PNG\r\n\x1a\n' else "image/jpeg"
        system_prompt = build_canonical_prompt(user_hint=user_hint, variant=self.prompt_variant)

        payload = {
            "model": self.model_name,
            "max_tokens": 1024,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": base64_image
                            }
                        },
                        {
                            "type": "text",
                            "text": system_prompt
                        }
                    ]
                }
            ]
        }

        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01"
        }

        try:
            max_attempts = 8
            last_err: Optional[Exception] = None
            for attempt in range(1, max_attempts + 1):
                try:
                    response = requests.post(self.url, headers=headers, json=payload, timeout=120)

                    if response.status_code == 429:
                        retry_after = response.headers.get("Retry-After")
                        base = float(retry_after) if retry_after else min(120.0, 10.0 * (2.0 ** (attempt - 1)))
                        sleep_s = base + random.random() * 2.0
                        print(f"[AnthropicAnalyzer] 429 rate limit. Sleeping {sleep_s:.1f}s before retry {attempt}/{max_attempts}")
                        time.sleep(sleep_s)
                        raise RuntimeError(f"429 Too Many Requests (sleep {sleep_s:.1f}s)")

                    if 500 <= response.status_code < 600:
                        raise RuntimeError(f"{response.status_code} Server Error: {response.text[:500]}")

                    response.raise_for_status()

                    try:
                        result = response.json()
                    except Exception as e:
                        raise RuntimeError(f"Invalid JSON response body (status {response.status_code})") from e

                    # Extract text content from Claude's response
                    content = result.get('content', [])
                    text_content = ""
                    for block in content:
                        if block.get('type') == 'text':
                            text_content = block.get('text', '')
                            break

                    if not text_content.strip():
                        raise RuntimeError("Empty Claude message content")

                    # Parse JSON from response (Claude might include markdown code blocks)
                    clean = text_content.strip()
                    if '```json' in clean:
                        clean = clean.split('```json')[1].split('```')[0]
                    elif '```' in clean:
                        clean = clean.split('```')[1].split('```')[0]

                    try:
                        data = json.loads(clean)
                    except Exception:
                        match = re.search(r"\{[\s\S]*\}", clean)
                        if match:
                            data = json.loads(match.group(0))
                        else:
                            raise

                    # Normalize and calculate totals
                    items = data.get('items', [])
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
                            "general_description": data.get("general_description", "Analyzed by Claude"),
                            "estimated_calories": total_cals,
                            "estimated_weight": total_wt,
                            "items": cleaned_items
                        }
                    }
                except Exception as e:
                    last_err = e
                    msg = str(e)
                    if (
                        "429" in msg
                        or "Too Many Requests" in msg
                        or "timed out" in msg.lower()
                        or "timeout" in msg.lower()
                        or "rate" in msg.lower()
                        or "connection aborted" in msg.lower()
                        or "remote end closed" in msg.lower()
                        or "invalid json response body" in msg.lower()
                        or "empty claude message content" in msg.lower()
                        or "expecting value" in msg.lower()
                        or "server error" in msg.lower()
                    ):
                        sleep_s = min(120.0, 10.0 * (2.0 ** (attempt - 1))) + random.random() * 2.0
                        print(f"[AnthropicAnalyzer] Retry {attempt}/{max_attempts} after error: {e}. Sleeping {sleep_s:.1f}s")
                        time.sleep(sleep_s)
                        continue
                    raise

            raise RuntimeError(f"Anthropic request failed after {max_attempts} attempts: {last_err}")

        except Exception as e:
            print(f"[AnthropicAnalyzer] Error: {e}")
            return {
                "analysis": {
                    "items": [],
                    "general_description": f"Error: {str(e)}",
                    "estimated_calories": 0,
                    "estimated_weight": 0,
                }
            }
