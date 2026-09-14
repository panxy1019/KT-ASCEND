# Regression

Executed before the full P1 gate:

- backend captured-input determinism: pass for LLAMAFILE/OpenBLAS/BLIS1/ATLAS/ACL
- OpenBLAS integrated output vs isolated output: pass on 12 real rows
- Round 4A P1 local C0–C4: pass
- sequential vs overlap: exact pass
- 1000-forward single hash: pass
- SGLang KT EP registered Ascend routing test: pass (`1 passed`)

Not executed after P1 failed:

- Round2A, Round2B, Round2C and Round3 full regression campaigns
- P2/P3 regression

The experimental production adapter was removed, leaving zero production diff;
therefore no rejected backend code is carried into earlier rounds. The new
probe tools are non-production and syntax-checked.
