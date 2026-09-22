# HD2 input freeze

128 new authored prompts, same category counts and 64-position protocol as QD2.
No HD2 model inference has occurred. Input metadata is not held-out output.

Manifest: `corpora/frozen-corpora/hd2.json`.
Canonical payload SHA:
`eb4b59b76840174498e7c615fc26c2c93d76dfae2a3b6ff5bc417c3b4d461539`.
Serialized file SHA256:
`b22cd79c82769bb25bf1da30e39046f311071f8b5e1a537fef66f99ae462ba3a`.

The shared `pair_freeze.json` binds both manifests from one offline invocation.
Tokenizer SHA:
`41f3bf64213da8c012d8bd0871a58a1fdf70463e8f08f110ddbb1082f529f669`.
All input lengths leave room for 64 positions within context512.
History is generated only after the relevant inference gate opens and bound
by a separate hash; no invented baseline token history is embedded here.

Do not run HD2 until QD2-derived candidate commit and retained runtime gates
are complete. No HD2-based prompt replacement, threshold choice, deletion,
confidence change or candidate-universe change is allowed.
The workload/independence limitations in the QD2 freeze document also apply.
