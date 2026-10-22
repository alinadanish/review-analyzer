import json
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types
from typing import Literal
from pydantic import BaseModel, Field

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY is missing")

client = genai.Client(api_key= api_key)

class AnalysisResult(BaseModel):
    label: Literal["positive", "negative", "neutral"]
    score: int =Field(ge=-1, le=1)
    theme: str

def analyze_with_gemini(review: str) -> dict:
    prompt =f"""
    Analyze this product review. Return JSON with:
    - label: positive, negative, or neutral
    - score: an integer from -1 (very negative) to 1 (very positive)
    - theme: the main topic, such as quality, delivery, price, or general

    Review: {review!r}
    """

    response = client.models.generate_content(
        model="gemini-flash-lite-latest",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json"
        )
    )

    if response.text is None:
        raise ValueError("Gemini returned an empty response")

    result = AnalysisResult.model_validate(json.loads(response.text))
    return result.model_dump()