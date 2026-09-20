# Full-model system gates and non-atomic telemetry repair

Trialv3 on110.123.0.3physical15 passed the first real HTTP prefill/decode,
compact32 across four prompts, active streaming abort followed by exact output
ID equality on state reuse, and four fixed known-answer canaries (391,北京,
ORANGE,and the requested JSON object). No benchmark score or performance claim.

Observed short-workload peaks: owned process39850MiB, board48708MiB, torch
allocated40705135616bytes, reserved41502638080bytes; cgroup peak334439145472bytes
and failcnt0. Training process remained6124MiB in these samples. A prior complete
identity receipt and live144file sizes/metadata hashes were rebound at startup;
this is not another whole360GB hash pass.

Actual4096-token resource boundary did not execute in trialv3: f02 reserved
slots caused an HTTP400. Report32 explains corrected internal4098context and
4100token pool for the same actual4096 input+output workload. State payload
still fits321MiB. Do not relabel short-request peaks as boundary qualification.

Trialv4 was stopped by its own guard during model construction, not OOM.
A real sample read board28592MiB, owned10864MiB, training6124MiB. Board minus
owned yielded17728MiB, falsely suggesting nonowned exceeded15GiB. The previous
sample had training6124MiB too. npu-smi process and board counters update at
different times during fast allocation and release, so subtraction is not an
atomic measurement of training growth. Raw samples and the stopped container
are retained.

The repaired guard records system_baseline=board-training while the owned
container is still CREATED. The current verified baseline is2845MiB. Training
budget checks use training_process+baseline>15360MiB; the live subtraction is
retained as a diagnostic, clearly distinct from that estimate. Absolute board
>56832MiB and owned>41472MiB checks remain immediate, as do unknown device PID,
telemetry failure and host available memory<64GiB. Changing system overhead
is still covered by the absolute board threshold. This does not make sampling
atomic or promise a hardware reservation; the39.5GiB PyTorch cap remains a
separate process-local limit, not a cap on native/CANN allocations.

Regression first failed with the missing decision helper, then4 tests passed:
actual asynchronous sample is accepted, and real board/owned/training-budget
exceedances still stop immediately. Trialv5 now retries under this guard and
corrected internal guard-slot profile with memory observer ON. Full-model
boundary, clean restart/observer comparison and CMMLU reference remain pending.

CMMLU256 is frozen, not scored. Original ordered items/sourceSHA are unchanged;
Qwen3.8 original tokenizer/chat template is freshly applied with thinkingOFF,
225–722prompttokens. ManifestSHA88efae7d68e7edf644add12909db89712dc001046b8624801dbb8ce7a8804858
is committed at e66726f2f6a07aad85a974a15e0aba7604a20852 before any scoring.
No earlier model qualification or scores are inherited.
