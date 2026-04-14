import json
import os
from typing import Any, Dict, Optional

try:
    from gigachat import GigaChat
except ModuleNotFoundError as exc:
    GigaChat = None
    _GIGACHAT_IMPORT_ERROR = exc
else:
    _GIGACHAT_IMPORT_ERROR = None


def _strip_markdown_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]

    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]

    return cleaned.strip()


def parse_product_params_gigachat(text: str, credentials: Optional[str] = None) -> Dict[str, Any]:
    """Parse structural product parameters from free text via GigaChat."""
    if GigaChat is None:
        raise ModuleNotFoundError(
            "Optional dependency 'gigachat' is required for LLM parsing. "
            "Install it or avoid calling parse_product_params_gigachat()."
        ) from _GIGACHAT_IMPORT_ERROR

    if credentials is None:
        credentials = os.getenv("GIGACHAT_CREDENTIALS")

    if not credentials:
        raise ValueError(
            "GigaChat credentials not provided. Set GIGACHAT_CREDENTIALS env var."
        )

    prompt = f"""
You are a financial analyst. Read the product description and return STRICT JSON only.
No markdown, no comments, no extra text.

Required JSON shape:
{{
  "meta": {{
    "isin": "string or null",
    "type": "string, e.g. PhoenixAutocall"
  }},
  "params": {{
    "nominal": number,
    "currency": "string, e.g. RUB",
    "underlyings": ["TICKER1", "TICKER2"],
    "feature_type": "string, e.g. standard or hybrid_avg_worst",
    "coupon_rate": number,
    "participation_rate": "optional number, e.g. 1.5",
    "strike": "optional number in performance terms, e.g. 1.0",
    "barriers": {{
      "autocall": number,
      "coupon": number,
      "capital_protection": number
    }}
  }}
}}

Description:
{text}
"""

    with GigaChat(credentials=credentials, verify_ssl_certs=False) as giga:
        response = giga.chat(prompt)
        content = response.choices[0].message.content

    try:
        return json.loads(_strip_markdown_fences(content))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Failed to parse JSON from GigaChat response. Response: {content}"
        ) from exc
