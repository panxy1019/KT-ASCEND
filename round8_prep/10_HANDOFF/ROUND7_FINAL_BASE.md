# Round7 closed; Round8 entry authorized

Final Round7 commit: `30de6a234284bfa25d196037ed692d9584855ddf`.
Status COMPLETE_WITH_USER_APPROVED_SCOPE_REDUCTION, not comprehensive formal
performance qualification. SelectedUniformR8C64,DynamicOFF,DeferredOFF.
D1CMMLU174/256invalid0. Short12cells only2resourceeligible;Medium/Long remainder
explicitly skipped. Do not revive cancelled campaigns.

Round8 prep remains on independent branch, no wholesale merge of historical
stack. Use Round7 final production artifact identity as reference; its local
commit is not identical to remote installed parent+child+overlays. Existing
runtime sources are mounted read-only in isolatedRound8container pending
minimal Qwen adapter selection.

Remote host .22; `/home/admin/kt-round8-qwen35` is the new writable root.
Container `kt-round8-qwen35-prep`, CPU128-191/NUMA4,5,NPU2,192GiB limit,
swap limit equals memory. All prior Round7 mounts read-only; independent logs,
builds/env/model staging must go under `/workspace/round8`.

Initial container attempt inherited image ENTRYPOINT=/bin/sleep and accidentally
supplied redundant /bin/sleep inCmd,exit1 before model imports. Preserved as
`kt-round8-qwen35-prep-entrypoint-failed-v1`; corrected with explicit entrypoint
andCmd infinity. No production/runtimefix, no oldcontainer modified.
