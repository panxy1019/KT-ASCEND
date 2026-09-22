"""Pre-primary zero-shot numeric extraction, no code execution or gold access.

Explicit #### when supplied, otherwise last numeric substring in the generated
answer. This is the conventional flexible numeric extraction for a base-model
zero-shot CoT, not strict instruction-following certification. Truncations are
still wrong. CMMLU stops on newline and accepts exactly one option letter.
"""
from decimal import Decimal
import re
import unicodedata
from quality_answers import parse as strict_parse

NUMBER = r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"


def parse(benchmark,text):
    text=unicodedata.normalize("NFKC",text).strip()
    if benchmark == "cmmlu":
        return strict_parse(benchmark,text)
    if benchmark != "gsm8k":
        raise ValueError("UNKNOWN_BENCHMARK")
    if "####" in text:
        return strict_parse(benchmark,text)
    matches=list(re.finditer(NUMBER,text))
    return str(Decimal(matches[-1].group().replace(",","")).normalize()) if matches else None


def score(benchmark,text,answer,truncated=False):
    predicted=parse(benchmark,text)
    gold=str(Decimal(answer.replace(",","")).normalize()) if benchmark == "gsm8k" else strict_parse(benchmark,answer)
    if gold is None:
        raise ValueError("INVALID_GOLD")
    return dict(prediction=predicted,correct=not truncated and predicted==gold,
                invalid=predicted is None,truncated=bool(truncated))
