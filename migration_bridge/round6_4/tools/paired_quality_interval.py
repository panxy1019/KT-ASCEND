"""Conservative exact paired difference bound; no independent-arm approximation.

Let gain/loss be mutually exclusive paired outcomes. Binomial marginal exact
bounds each spend alpha/2; union bound gives P(delta >= lower) >= 1-alpha,
without assuming gain and loss independent. Bootstrap is supplemental only.
"""
import numpy as np
from scipy.stats import beta


def bounds(n, gains, losses, alpha=0.025):
    if any(type(v) is not int for v in (n, gains, losses)) or n <= 0:
        raise ValueError("INVALID_COUNTS")
    if min(gains, losses) < 0 or gains + losses > n or not 0 < alpha < 1:
        raise ValueError("INVALID_COUNTS_OR_ALPHA")
    tail = alpha / 2

    def lower(k):
        return 0.0 if k == 0 else float(beta.ppf(tail, k, n - k + 1))

    def upper(k):
        return 1.0 if k == n else float(beta.ppf(1 - tail, k + 1, n - k))

    return dict(delta=(gains - losses) / n,
                lower=lower(gains) - upper(losses),
                upper=upper(gains) - lower(losses),
                one_sided_confidence=1-alpha,
                method="paired_discordance_exact_marginals_bonferroni")


def evaluate(off, on):
    if len(off) != len(on) or not off or any(type(x) is not bool for x in [*off, *on]):
        raise ValueError("EXPECTED_ALIGNED_BOOLEAN_PAIRS")
    n = len(off)
    gains = sum(b and not a for a, b in zip(off, on))
    losses = sum(a and not b for a, b in zip(off, on))
    result = bounds(n, gains, losses)
    rng = np.random.default_rng(6404)
    samples = rng.multinomial(n, [gains/n, losses/n, 1-(gains+losses)/n], size=20000)
    bootstrap = (samples[:, 0] - samples[:, 1]) / n
    result.update(n=n, correct_off=sum(off), correct_on=sum(on), gains=gains, losses=losses,
                  bootstrap_percentile_95=np.quantile(bootstrap, [.025, .975]).tolist(),
                  bootstrap_replicates=20000, seed=6404, margin=0.02,
                  status="NONINFERIOR" if result["lower"] > -0.02 else
                         "REGRESSION_BEYOND_MARGIN" if result["upper"] < -0.02 else "INCONCLUSIVE")
    return result
