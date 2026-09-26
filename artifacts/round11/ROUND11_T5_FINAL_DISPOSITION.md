# Round11 T5 final disposition

T5 is complete as `COMPLETE_QSA_COMPONENT_ONLY_NO_PROMOTION`.

Two independent bounded profile processes ranked QSA as the real B-route hotspot: its median attributable share was 53.87%, versus 19.07% for GDN. PLE consumer cost was 0.278% and PLE exposed wait was 0.0177%, so PLE overlap was not reopened.

The single allowed QSA candidate, query-aware vectorized sparse GQA, passed the v144 real-device component gate. Its maximum relative L2 error was `5.645496551606974e-05`; the prefill component ratios at contexts 32/64/128 were `0.05691/0.05675/0.05649`, while Decode was neutral at `1.00506`. Allocation returned to baseline and the candidate remained default OFF.

Request-level promotion did not complete. v145 failed before container creation because the launcher still bound the original QSA source hash. v146 repaired that launcher boundary and loaded the model, but the runtime `context_reference_binding.py` independently reapplied the original-QSA gate; the server exited before readiness and before any request. The original failures are preserved. This was an integration contract failure, not a candidate numerical failure. T5's three-model-start budget was exhausted by the two profile processes and v146, so no additional model retry was made.

The final provider is the original HFINAL QSA implementation. The candidate disposition is `COMPONENT_ONLY_NO_PROMOTION`; Round11 makes no request-level QSA speedup claim.
