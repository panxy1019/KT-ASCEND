# Transformers / dependency boundary

Config serialization declares `4.57.0.dev0`; this is NOT an install requirement. Frozen KT child pyproject instead declares `transformers-kt==5.6.0.post3`, torch2.9.1. Recorded Round7 native extension uses torch2.9.0; do not pip-install over it.

Two isolated candidates only: (A) preserve Round7 CANN/torch_npu/native ABI and backport minimum Qwen model/GDN compatibility; (B) use an independently pinned compatible KT/torch_npu stack. Selection requires `AutoConfig`, tokenizer/chat-template, Qwen model construction and NPU GDN operator smoke, with full pip freeze and binary hashes.

Current desktop prep venv contains stdlib tools only. No Transformers/PyTorch model construction executed. Dependency lock and Ascend wheel build remain NOT_RUN. Do not label prep tests as model construction tests.
