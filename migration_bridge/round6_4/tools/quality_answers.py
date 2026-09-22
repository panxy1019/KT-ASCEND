"""Frozen-template answer extraction; never executes generated expressions/code."""
from decimal import Decimal
import re
import unicodedata

NUMBER = r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"


def parse(benchmark, text):
    text = unicodedata.normalize("NFKC", text).strip()
    if benchmark == "gsm8k":
        # Require an explicitly marked final answer at end, not a number from
        # an intermediate equation. Dataset gold uses this same #### marker.
        match = re.search(r"####\s*(" + NUMBER + r")\s*$", text)
        return str(Decimal(match.group(1).replace(",", "")).normalize()) if match else None
    if benchmark == "cmmlu":
        match = re.fullmatch(r"([ABCD])[.。]?", text)
        return match.group(1) if match else None
    raise ValueError("UNKNOWN_BENCHMARK")


def score(benchmark, text, answer, truncated=False):
    predicted = parse(benchmark, text)
    gold = parse(benchmark, "#### " + answer if benchmark == "gsm8k" else answer)
    if gold is None:
        raise ValueError("INVALID_GOLD")
    return dict(prediction=predicted, correct=not truncated and predicted == gold,
                invalid=predicted is None, truncated=bool(truncated))
