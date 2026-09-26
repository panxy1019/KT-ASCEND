# Round10 unprofiled request comparison

| Configuration | Scope | Samples (s) | Median (s) | Ratio to H0 | Disposition |
|---|---|---:|---:|---:|---|
| H0 v95 Hybrid129 | 64-token ACK/EOS | 5.194823, 5.200895, 5.181578, 5.181313 | 5.188200 | 1.000000 | Frozen baseline |
| HFINAL v107 | same fixed configuration | 5.409330, 5.832713 | 5.621022 | 1.083424 | Quality/reuse regression PASS; no speed claim |
| D1 v102 Dynamic | 64-token request including synchronous state proof/migration | 24.130848, 23.166609 | 23.648729 | 4.558176 | Correct transaction, default OFF |

Hybrid129 is a path selection qualification. It does not demonstrate Layerwise prepare acceleration. v107 and v95 use the same enabled configuration, so their observed difference is repeat dispersion rather than a promoted implementation delta.
