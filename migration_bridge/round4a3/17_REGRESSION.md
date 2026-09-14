# Regression

## Tool regressions added

- exclusive sweep lock prevents concurrent dump interleaving;
- serving-token-to-captured-logit consistency check;
- stable-token contract validator;
- near-tie membership validator;
- deterministic same-path/prefix tool;
- free-generation first-divergence classifier;
- paired quality A/B scorer.

All Python tools pass `py_compile`. The Q/H sampling-pass invariant passed for
all accepted 2688 mode-position captures (Q 576x2 plus H 768x2).

## Not run

Round2A/Round2B/Round2C/Round3, SGLang KT EP, P1 lifecycle, stable/near-tie
fixture promotion, and downstream quality regressions were not run after the
held-out stop. Production code and the SGLang child tree have zero Round 4A.3
changes.
