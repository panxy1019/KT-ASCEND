# Native TP8 supplemental workflow

Updated: 2026-09-27 UTC

Status: `PASS_ROUND12A_NATIVE_TP8_BASELINE_ABBA`.

v180/v181/v182/v183 completed the frozen A/B/B/A identical-implementation repeat sequence on physical0-7. Each process passed a fresh 2x60-second admission, real-device ServerArgs/source identity checks, all-eight-rank Expert/GDN/old-QSA evidence, four exact requests, HBM/Host guards, guard exit ACK, container removal, full local SHA archive and targeted checkpoint cache retirement.

Process E2E medians are 3.428952584, 3.371347328, 3.489020653 and 3.325310193 seconds. The comparator passes its 5% timing-stability gate with max/min 1.049231636. All twelve measurements returned `[3897, 248046]`; cross-run per-device HBM peak spread is 0-2 MiB and Host peak max/min is 1.000052514.

The three no-device same-image attempts are retained as preflight-design failures. Frozen SGLang requires a real NPU for `is_npu()`, the Qwen native extension, and `torch.npu.mem_get_info()`. After the two-revision limit, no third fake-device shim was added; the real-device preflight already present before `run_server` passed in all four budgeted processes.

This baseline uses old QSA SHA `76e9f34066f0b42f42545427967c0b789d5c32031dc5c085e230efaf95e3c54c`. It does not form a causal KT/Native speedup comparison with the promoted-QSA main ABBA. Main and Expert weights/compute are on eight NPUs, while the PLE checkpoint row table remains Host-backed.
