# ============================================================
# llm.py - Shared helper for talking to Gemini safely
# ============================================================
# The free tier has rate limits. When we send many requests
# quickly we can get:
#   - 429 RESOURCE_EXHAUSTED  (too many requests, slow down)
#   - 503 UNAVAILABLE         (model temporarily busy)
#
# Instead of crashing, this helper:
#   1. Retries automatically with a growing wait time.
#   2. Respects the "retry in X seconds" hint Google sends.
#   3. Falls back to a lighter model (its own separate quota)
#      if the main model stays unavailable.
# ============================================================

import re
import time

from google.genai import errors

# Main model, plus a backup that has its own separate quota.
# We use flash-lite as the primary because the regular flash model's
# free-tier daily quota is very small and gets used up quickly.
PRIMARY_MODEL  = "gemini-2.5-flash-lite"
FALLBACK_MODEL = "gemini-2.5-flash"


def _suggested_delay(exc, default):
    """
    Pull the "retryDelay": "38s" hint out of the error if present.
    Returns a number of seconds to wait.
    """
    match = re.search(r"retryDelay'?:?\s*'?(\d+)s", str(exc))
    if match:
        # Add 1s of safety margin so we don't retry a hair too early.
        return int(match.group(1)) + 1
    return default


def generate_text(client, prompt, max_attempts=6):
    """
    Send `prompt` to Gemini and return the response text.

    Handles rate limits (429) and temporary outages (503) by waiting
    and retrying, and by falling back to a lighter model.
    """
    delay = 4  # seconds; grows after each failed round
    last_exc = None

    for attempt in range(1, max_attempts + 1):
        # Prefer the main model; switch to the lighter one on later tries.
        model = PRIMARY_MODEL if attempt <= 2 else FALLBACK_MODEL

        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
            )
            return response.text

        except errors.ClientError as exc:
            # 429 = rate limited. Anything else (e.g. bad request) is a
            # real error we should not silently retry.
            if exc.code != 429:
                raise
            last_exc = exc
            wait = _suggested_delay(exc, default=delay)
            print(f"  [LLM] Rate limited (429). Waiting {wait}s, then retrying "
                  f"(attempt {attempt}/{max_attempts})...")
            time.sleep(wait)
            delay = min(delay * 2, 60)

        except errors.ServerError as exc:
            # 503 = model busy. Wait a bit and try again.
            last_exc = exc
            wait = _suggested_delay(exc, default=delay)
            print(f"  [LLM] Model busy ({exc.code}). Waiting {wait}s, then retrying "
                  f"(attempt {attempt}/{max_attempts})...")
            time.sleep(wait)
            delay = min(delay * 2, 60)

    # If we get here, every attempt failed.
    raise last_exc
