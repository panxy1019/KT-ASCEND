"""Prospective assumed-discordance scenarios, never model-result-driven power."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.stats import beta


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    rng = np.random.default_rng(6404)
    rows = []
    for n in (1319, 2048):
        for discordance in (.01, .05, .10, .20):
            for delta in (-.01, 0, .005):
                gains, losses, _ = rng.multinomial(n, [(discordance+delta)/2,
                    (discordance-delta)/2, 1-discordance], size=20000).T
                lower_gain = np.where(gains == 0, 0, beta.ppf(.0125, gains, n-gains+1))
                upper_loss = np.where(losses == n, 1, beta.ppf(.9875, losses+1, n-losses))
                rate = float(np.mean(lower_gain-upper_loss > -.02))
                rows.append(dict(n=n, assumed_discordance=discordance, assumed_delta=delta,
                                 probability_of_noninferiority_pass=rate,
                                 monte_carlo_standard_error=(rate*(1-rate)/20000)**.5))
    report = dict(status="PROSPECTIVE_SIMULATION_NOT_MODEL_EVIDENCE", seed=6404,
                  replicates=20000, assumptions="IID paired trinomial benchmark sampling model",
                  method="paired_discordance_exact_marginals_bonferroni", rows=rows)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(rows), flush=True)


if __name__ == "__main__":
    main()
