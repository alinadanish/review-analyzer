import json
import os
import logging

from dotenv import load_dotenv
from google import genai
from google.genai import types
from typing import Literal
from pydantic import BaseModel, Field

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY is missing")

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite-preview")
GEMINI_FALLBACK_MODELS = [
    model.strip()
    for model in os.getenv(
        "GEMINI_FALLBACK_MODELS",
        "gemini-3-flash-preview,gemini-3.5-flash",
    ).split(",")
    if model.strip()
]

client = genai.Client(
    api_key=api_key,
    http_options=types.HttpOptions(
        timeout=15_000,
        retry_options=types.HttpRetryOptions(
            attempts=1,
            initial_delay=1,
            max_delay=1,
            exp_base=1,
            http_status_codes=[500, 502, 503, 504],
        ),
    ),
)

class AnalysisResult(BaseModel):
    label: Literal["positive", "negative", "neutral"]
    score: int =Field(ge=-1, le=1)
    theme: str


def should_try_fallback(error: Exception) -> bool:
    status = getattr(error, "code", None)
    message = str(error).lower()
    return (
        status in {429, 503, 504}
        or "resource_exhausted" in message
        or "high demand" in message
        or "overloaded" in message
        or "temporarily unavailable" in message
    )


def ordered_models() -> list[str]:
    return list(dict.fromkeys([GEMINI_MODEL, *GEMINI_FALLBACK_MODELS]))


def generate_content_sync(*, contents, config=None):
    last_error = None
    for model_name in ordered_models():
        try:
            return client.models.generate_content(
                model=model_name,
                contents=contents,
                config=config,
            )
        except Exception as error:
            last_error = error
            logging.warning("Gemini model %s failed: %s", model_name, type(error).__name__)
            if not should_try_fallback(error):
                raise
    if last_error is not None:
        raise last_error


async def generate_content_async(*, contents, config=None):
    last_error = None
    for model_name in ordered_models():
        try:
            return await client.aio.models.generate_content(
                model=model_name,
                contents=contents,
                config=config,
            )
        except Exception as error:
            last_error = error
            logging.warning("Gemini model %s failed: %s", model_name, type(error).__name__)
            if not should_try_fallback(error):
                raise
    if last_error is not None:
        raise last_error


def build_analysis_prompt(review: str) -> str:    return f"""
    Analyze this product review. Return JSON with:
    - label: positive, negative, or neutral
    - score: an integer from -1 (very negative) to 1 (very positive)
    - theme: the main topic, such as quality, delivery, price, or general

    Review: {review!r}
    """


def analysis_config() -> types.GenerateContentConfig:
    return types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=AnalysisResult,
    )


async def analyze_with_gemini_async(review: str) -> dict:
    response = await generate_content_async(
        contents=build_analysis_prompt(review),
        config=analysis_config(),
    )

    if response.text is None:
        raise ValueError("Gemini returned an empty response")

    result = AnalysisResult.model_validate(json.loads(response.text))
    return result.model_dump()


def analyze_with_gemini(review: str) -> dict:
    response = generate_content_sync(
        contents=build_analysis_prompt(review),
        config=analysis_config(),
    )

    if response.text is None:
        raise ValueError("Gemini returned an empty response")

    result = AnalysisResult.model_validate(json.loads(response.text))
    return result.model_dump()