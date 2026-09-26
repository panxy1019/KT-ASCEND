# Round10 T4 Dynamic disposition

Status: **PASS_TRANSACTION_NO_NET_BENEFIT_DEFAULT_OFF**.

The one allowed Hybrid-to-Dynamic trigger wiring reuses the logical route IDs already captured at the real KTEP method boundary. It publishes only after the first output token, requires all 48 layers and exact `tokens × Top-10` conservation, and then invokes the existing synchronous NZ29 migration transaction. Explicit cancellation drains and discards observations without placement publication.

The v102 full-model process completed five generations and one abort with no retry. Four successful requests preserved live KV, GDN, QSA and PLE state byte-for-byte across migration. Commit generations were 1, 2, 3 and 4. The cancel request published no placement, returned its lease, and the following 27-token request produced the frozen `391` answer in the same process.

Performance rejects enablement. H0 v95 64-token request measurements had median 5.188200 s; D1 v102 had median 23.648729 s, or 4.558176× H0. The first migration copied 1431 slots and took 68.381127 s. Even the two repeated inputs copied 118 and 86 slots and took 24.130848 s and 23.166609 s. These clocks include the synchronous live-state proof and migration on the request critical path.

D1 remains default OFF. This is a current Hybrid lifecycle qualification of transaction correctness, not a speedup and not a reason to alter the frozen Layerwise/NZ update path. Raw evidence archive SHA-256 is `c73b5d17f3ffccf93201670e624abd8a993a6fe89fc5a35a166bea50e89c043c`.
