"""Offline one-sided binomial bounds. Statistical units must be independent.

No model data, network calls, or experiment defaults are embedded here.
Prompt-block aggregation avoids treating within-prompt positions as IID trials;
it does not itself establish independence between prompts.
"""
from __future__ import annotations

import math


def binomial_cdf(k: int, n: int, p: float) -> float:
    if not isinstance(n, int) or n < 0 or not math.isfinite(p) or not 0 <= p <= 1:
        raise ValueError("invalid binomial parameters")
    if not isinstance(k, int):
        raise ValueError("k must be an integer")
    if k < 0:
        return 0.0
    if k >= n or p == 0:
        return 1.0
    if p == 1:
        return 0.0
    terms = [
        math.lgamma(n + 1) - math.lgamma(j + 1) - math.lgamma(n - j + 1)
        + j * math.log(p) + (n - j) * math.log1p(-p)
        for j in range(k + 1)
    ]
    peak = max(terms)
    return min(1.0, math.exp(peak) * math.fsum(math.exp(t - peak) for t in terms))


def upper_confidence_bound(exceedances: int, n: int, alpha: float) -> float:
    """One-sided Clopper-Pearson upper bound, confidence 1-alpha."""
    if (not isinstance(n, int) or n <= 0 or not isinstance(exceedances, int)
            or not 0 <= exceedances <= n or not 0 < alpha < 1):
        raise ValueError("invalid confidence-bound parameters")
    if exceedances == n:
        return 1.0
    if exceedances == 0:
        return -math.expm1(math.log(alpha) / n)
    lo, hi = 0.0, 1.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if binomial_cdf(exceedances, n, mid) > alpha:
            lo = mid
        else:
            hi = mid
    return hi


def tolerance_rank(n: int, coverage: float, alpha: float) -> int:
    """Smallest 1-based order statistic with confidence >=1-alpha.

    For continuous IID samples, P[F(X_(k)) >= coverage] =
    P[Binomial(n, coverage) <= k-1]. Atoms make the bound conservative.
    Insufficient sample size is an error, never an infinite usable reserve.
    """
    if not isinstance(n, int) or n <= 0 or not 0 < coverage < 1 or not 0 < alpha < 1:
        raise ValueError("invalid tolerance parameters")
    for rank in range(1, n + 1):
        if binomial_cdf(rank - 1, n, coverage) >= 1 - alpha:
            return rank
    raise ValueError("INSUFFICIENT_INDEPENDENT_CALIBRATION_UNITS")


def derive_threshold(values: list[float], coverage: float, alpha: float) -> dict:
    if not values or any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("expected nonempty finite nonnegative risk values")
    rank = tolerance_rank(len(values), coverage, alpha)
    return {"n": len(values), "rank_1based": rank,
            "threshold": sorted(values)[rank - 1], "coverage": coverage,
            "confidence": 1 - alpha}


def summarize_tail(values: list[float], threshold: float, alpha: float,
                   risk_budget: float) -> dict:
    if (not values or any(not math.isfinite(v) or v < 0 for v in values)
            or not math.isfinite(threshold) or threshold < 0
            or not 0 < risk_budget < 1):
        raise ValueError("invalid heldout risk values or bound")
    count = sum(value > threshold for value in values)
    upper = upper_confidence_bound(count, len(values), alpha)
    return {"n": len(values), "exceedances": count,
            "rate": count / len(values), "upper_confidence_bound": upper,
            "confidence": 1 - alpha, "risk_budget": risk_budget,
            "pass": upper <= risk_budget}


def prompt_block_maxima(rows: list[dict], metric: str,
                        expected_positions: dict[str, int]) -> list[float]:
    """Fail closed on missing/duplicate positions; never silently filter rows."""
    expected = {(pid, index) for pid, n in expected_positions.items() for index in range(n)}
    if not expected or any(not isinstance(n, int) or n <= 0 for n in expected_positions.values()):
        raise ValueError("invalid expected positions")
    seen = set()
    blocks = {pid: [] for pid in expected_positions}
    for row in rows:
        key = (row["prompt_id"], row["token_index"])
        if key not in expected or key in seen:
            raise ValueError("unexpected or duplicate position")
        seen.add(key)
        value = float(row[metric])
        if not math.isfinite(value) or value < 0:
            raise ValueError("non-finite or negative risk")
        blocks[key[0]].append(value)
    if seen != expected:
        raise ValueError("missing positions")
    return [max(blocks[pid]) for pid in sorted(blocks)]
