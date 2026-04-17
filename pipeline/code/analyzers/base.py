from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class FoodAnalyzer(ABC):
    @abstractmethod
    def analyze(self, image_bytes: bytes, user_hint: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze an image and return a standardized dictionary:
        {
            "analysis": {
                "general_description": str,
                "estimated_calories": float,
                "items": [
                    {
                        "name": str,
                        "estimated_calories": float,
                        "estimated_weight_grams": float
                    },
                    ...
                ]
            }
        }
        """
        pass


