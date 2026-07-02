"""Shared helpers for nodes that call an LLM and expect a JSON verdict back."""

import json
import re

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def strip_fences(text: str) -> str:
    return _FENCE_RE.sub("", text.strip()).strip()


def invoke_json(llm, prompt: str):
    """Invoke the LLM and parse its response as JSON.

    Returns (parsed_dict, raw_response) so callers can still pull token usage
    off the raw response. Raises json.JSONDecodeError on a malformed reply --
    callers are expected to catch it and fall back to a deterministic result.
    """
    response = llm.invoke(prompt)
    raw = strip_fences(response.content)
    return json.loads(raw), response
