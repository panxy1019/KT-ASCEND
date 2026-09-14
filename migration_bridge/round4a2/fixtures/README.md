# Captured Fixture Policy

Raw Round 4A.1 production captures are intentionally not duplicated in Git.
They remain immutable A3 artifacts under the recorded capture directories.
The committed evidence JSON contains, for every consumed row, capture filename,
pass/token/expert identity, X dtype/shape/SHA256, and all weight hashes. This is
sufficient to detect operand drift and to re-export the same fixture from the
frozen captures.

The two near-tie top-20 responses are committed under `../evidence/` as the
small numerical regression fixtures. Future runs must preserve the documented
margin signs and token choices unless the numerical acceptance decision changes.
