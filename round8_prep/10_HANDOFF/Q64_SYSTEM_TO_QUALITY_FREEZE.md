# Q64 system closure → fresh quality reference

2026-09-19. 32distinctprompt/64response screen completed; independent raw audit
confirmed ordered exactID coverage,repeat equality,finite output logprobs,
nonempty text and protocolSHA7e07b05a802256200fc10982062c5ce3a59632f116f9cc12443fc0488c99fdac.
RawarchiveSHA22450380194dd2c314c3f0c5e37cca2bcedd2a181d49086a9286794495301027.

## Additional production operator evidence

Earlier middle/late evidence was CPU production vsCPUcanonical, not NPU grouped.
Added actual `UnquantizedFusedMoEMethod.forward_npu`, same deployed implementation,
BF16 weights and routing, FP32 partial output, NPU format29/FRACTAL_NZ asserted.
Eight selected real logical experts, compact8slot NPU tensor, C64LLAMAFILE,
same operands/weights/routes; canonicalCPU BF16-visible reference also recorded.
Each layer has12cases: top8 qlen1/8/64/512 plus8single-expert onehot controls.
CPU10repeats andNPU3repeats exact,allfinite,CPU/NPU andCPU/canonical relL2≤0.01.

| Layer | Max CPU-vs-production-NPU relativeL2 | Cases |
|---|---:|---:|
|0|0.0016931676|12|
|20|0.0017112474|12|
|39|0.0017120768|12|

Synthetic operands and compact8slot layout, not exhaustive256expert/NPU64layout
certification. Initial v1 probe accidentally leftinternalformatOFF; retain as
ND diagnostic only. v2explicitly follows deployed init_npu andassertsformat29.
No production modifications. RawarchiveSHA
c95edb581e3901eabb822e304d9ebd14437a9822009e311a41d486e9c80fe61f.

## Quality protocol frozen BEFORE scoring

Original unchanged CMMLU256 sourceSHA
bca1e1ec41da188e51f176813e6d916120eaab26b17a3dc63a99ed712cf9bf04.
All ordereditemIDs,answers,subject,fewshotquestiontext retained. OldDSV2 inputIDs
cannot be used with Qwen. Re-encode using pinnedQwen tokenizer/chattemplate,
enable_thinking=False,greedy,oneoutputtoken. Score highest ofA/B/C/D candidate
logprobs atfirstoutputposition;tiesA/B/C/D order,invalidrequired0.
Token lengths225–722, no truncation or sample substitution.

NewmanifestSHA365f557c0746db1e02ddee4ec66c7afb5e72eeb17e8dd8e0a47d1ed2b7af1b9d,
stored`qwen-cmmlu-manifest-v1.json`. This is a freshQ64 engineeringreference,
not all-NPU orDSV2 comparison. Q32 future criterion stays observed≥−5pp vsQ64.
Quality runner requires successfulsystemscreen+threewayv2, manifestSHA/commitack
before firstscoredrequest. No outcome-dependent prompt orscoring changes.

Q64 score must be audited for coverage,invalids andsanity beforeQ32promotion.
Diagnostic hooks remain ON for thisuntimedqualityrun; performanceNOTQUALIFIED.
