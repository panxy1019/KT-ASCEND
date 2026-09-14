# Host Callback Ordering

## Contract tested

The CANN callback is an ordering marker on an existing stream. It performs host-only work and returns. Source audit and the compiled probe confirm that the callback contains no `aclrtMemcpy*`, device operation launch, stream/device synchronization, allocation/free, or PyTorch call.

## A3 results

| Case | Result |
|---|---|
| single stream: device write → callback → later device read | counter 1, errors 0, output 1 |
| stress | 10,000 launches, counter 10,000, errors 0 |
| two independent streams | 1,000 callbacks per stream, exact counts, errors 0 |
| callback source restriction audit | PASS |

The tests establish per-stream ordering and independent progress for two streams. They do not claim a global order between streams. Callback thread identity is runtime-managed and is not assumed by production code.

Focused result: `4 passed in 7.62s`.

Evidence: `host-callback-ordering-final.log`; fixture source: `kt-kernel/test/fixtures/ascend/host_callback_probe.cpp`.
