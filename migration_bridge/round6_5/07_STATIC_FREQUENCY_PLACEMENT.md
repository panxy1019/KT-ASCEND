# Static Frequency — NOT_RUN_PREDECESSOR_FAILURE

DEV/HOLDOUT texts and offline input tokenIDs were frozen before routing analysis.
32prompts,16per split,4per English/Chinese/numeric/structured family.
Manifest rawSHA256:
`6aa4f561b7bc3c8bc31c78dbae151b38bd94ecdf14c8143026d0105bc099a187`.

Prepared tools use DEV count-descending/global-ID-ascending ties and compare
uniform/frequency on identical HOLDOUT reference histories. Lookahead is
excluded. Local unit tests ensure changing HOLDOUT cannot change the selected
policy and extra/duplicate forwards reject. Target complete tool unit suite:
28/28PASS;this is not model/frequency qualification.

No scheduling routes were collected,no hotness fitted,no frequency profile
selected or promoted. Locality is **UNKNOWN**,not weak or strong. Preserved
entry criteria:event reduction≥10% or whole-model fully-NPU rate+5pp.
