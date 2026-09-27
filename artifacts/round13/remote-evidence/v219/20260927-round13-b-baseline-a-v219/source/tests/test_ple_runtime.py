"""Exact-input async PLE consumer ownership, independent of device imports."""
import unittest
from types import SimpleNamespace as NS
from migration_bridge.round9.runtime import ple_runtime


class Tensor:
    def __init__(self, values): self.values = values
    def detach(self): return self
    def cpu(self): return self
    def tolist(self): return self.values


class Rows:
    shape = (2, 2, 3)
    def flatten(self, start):
        assert start == 1
        return 'flat-real-device-rows'


class Ring:
    row_width = 3
    def __init__(self): self.events = []; self.ticket = object(); self.fallback = False
    def submit_exact(self, *args, **kw):
        self.events.append(('submit', args, kw))
        return None if self.fallback else self.ticket
    def wait(self, ticket):
        assert ticket is self.ticket
        self.events.append(('wait',)); return Rows()
    def release(self, ticket):
        assert ticket is self.ticket
        self.events.append(('release',))
    def cancel_and_drain(self, ticket):
        assert ticket is self.ticket
        self.events.append(('cancel_and_drain',))


class PLEConsumerTests(unittest.TestCase):
    def test_wait_failure_and_missing_consumer_drain_ticket_before_poison(self):
        for kind in ('wait', 'boundary'):
            runtime, ring = self.ready(); payload = self.prepare(runtime).flatten(1)
            if kind == 'wait':
                def fail_wait(ticket):
                    ring.events.append(('wait_failed',))
                    raise OSError('wait failed')
                ring.wait = fail_wait
            with self.subTest(kind=kind), self.assertRaisesRegex(
                    (OSError, RuntimeError), 'wait failed|consumer did not drain'):
                if kind == 'wait': runtime.consume(payload, lambda rows: rows)
                else: runtime.end()
            self.assertEqual(ring.events[-1], ('cancel_and_drain',))
            self.assertEqual(runtime.state, 'BROKEN_RELOAD_REQUIRED')

    def test_cleanup_failure_preserves_wait_failure_as_primary(self):
        runtime, ring = self.ready(); payload = self.prepare(runtime).flatten(1)
        def fail_wait(ticket): raise OSError('primary wait failure')
        def fail_drain(ticket): raise RuntimeError('secondary drain failure')
        ring.wait = fail_wait; ring.cancel_and_drain = fail_drain
        with self.assertRaisesRegex(OSError, 'primary wait failure') as caught:
            runtime.consume(payload, lambda rows: rows)
        self.assertIsInstance(caught.exception.__cause__, RuntimeError)
        self.assertIn('secondary drain failure', str(caught.exception.__cause__))

    def test_sync_payload_is_shape_checked_and_batch_stamped(self):
        runtime, ring = self.ready(); ring.fallback = True
        wrong = NS(shape=(1, 99), flatten=lambda start: 'wrong')
        with self.assertRaisesRegex(RuntimeError, 'shape'):
            runtime.prepare(
                Tensor([7, 8]), Tensor([4]), Tensor([[0, 1], [2, 2]]),
                NS(gather_rows=lambda rows: wrong),
            )

        runtime, ring = self.ready(); ring.fallback = True
        old = self.prepare(runtime).flatten(1)
        runtime.consume(old, lambda rows: rows); runtime.end()
        runtime.begin('next', 4, [7, 8], 4, phase='sampled_decode')
        current = self.prepare(runtime).flatten(1)
        with self.assertRaisesRegex(ValueError, 'stale'):
            runtime.consume(old, lambda rows: rows)
        self.assertIsNot(old, current)
        self.assertEqual(runtime.state, 'BROKEN_RELOAD_REQUIRED')

    def test_bound_consumer_preserves_keyword_call_and_original_argument_names(self):
        runtime, ring = self.ready(); payload = self.prepare(runtime).flatten(1)
        calls = []
        class PLE:
            def run(self, *, hidden_states, embeddings, state_slot, forward_batch):
                calls.append((hidden_states, embeddings, state_slot, forward_batch))
                return 'original-output'
        ple = PLE(); ple_runtime.bind_ple_consumer(ple, runtime)
        result = ple.run(
            hidden_states='hidden', embeddings=payload,
            state_slot='state', forward_batch='batch',
        )
        self.assertEqual(result, 'original-output')
        self.assertEqual(calls, [
            ('hidden', 'flat-real-device-rows', 'state', 'batch'),
        ])
        self.assertEqual(ring.events[-1], ('release',))

    def ready(self):
        ring = Ring(); runtime = ple_runtime.PLEConsumerRuntime(ring)
        runtime.begin('r', 3, [7, 8], 4, phase='prefill_input')
        return runtime, ring

    def prepare(self, runtime):
        return runtime.prepare(Tensor([7, 8]), Tensor([4]), Tensor([[0, 1], [2, 2]]),
                               NS(gather_rows=lambda rows: Rows()))

    def test_gather_starts_before_consumer_and_releases_after_enqueued_use(self):
        runtime, ring = self.ready(); value = self.prepare(runtime)
        self.assertEqual(value.shape, (2, 2, 3))
        self.assertEqual([e[0] for e in ring.events], ['submit'])
        flattened = value.flatten(1)
        def use(rows):
            self.assertEqual(rows, 'flat-real-device-rows')
            self.assertEqual([e[0] for e in ring.events], ['submit', 'wait'])
            return 'ple-output'
        self.assertEqual(runtime.consume(flattened, use), 'ple-output')
        self.assertEqual([e[0] for e in ring.events], ['submit', 'wait', 'release'])
        runtime.end()

    def test_mismatched_input_slot_duplicate_or_stale_payload_fail_closed(self):
        for kind in ('ids', 'slot', 'duplicate', 'stale'):
            runtime, ring = self.ready()
            with self.subTest(kind=kind), self.assertRaises((ValueError, RuntimeError)):
                if kind == 'ids': runtime.prepare(Tensor([8, 7]), Tensor([4]), Tensor([[0, 1], [2, 2]]), None)
                elif kind == 'slot': runtime.prepare(Tensor([7, 8]), Tensor([5]), Tensor([[0, 1], [2, 2]]), None)
                else:
                    payload = self.prepare(runtime).flatten(1)
                    if kind == 'duplicate': self.prepare(runtime)
                    else:
                        runtime.consume(payload, lambda x: x); runtime.end()
                        runtime.begin('new', 3, [7, 8], 4, phase='sampled_decode')
                        runtime.consume(payload, lambda x: x)

    def test_consumer_failure_drains_ticket_and_poisoned_runtime_rejects_reuse(self):
        runtime, ring = self.ready(); payload = self.prepare(runtime).flatten(1)
        def fail(value): raise OSError('consumer failed after enqueue')
        with self.assertRaisesRegex(OSError, 'consumer failed'): runtime.consume(payload, fail)
        self.assertEqual(ring.events[-1], ('release',))
        with self.assertRaisesRegex(RuntimeError, 'BROKEN'): runtime.end()

    def test_sync_fallback_and_missing_consumer(self):
        runtime, ring = self.ready(); ring.fallback = True
        value = self.prepare(runtime)
        self.assertEqual(runtime.consume(value.flatten(1), lambda x: x), 'flat-real-device-rows')
        runtime.end()
        self.assertEqual([e[0] for e in ring.events], ['submit'])
        runtime, _ = self.ready(); self.prepare(runtime)
        with self.assertRaisesRegex(RuntimeError, 'consumer'): runtime.end()

    def test_invalid_phase_and_unbounded_input_rejected_before_submission(self):
        runtime = ple_runtime.PLEConsumerRuntime(Ring())
        for values, phase in [([], 'prefill_input'), (list(range(33)), 'prefill_input'),
                               ([7], 'future_decode')]:
            with self.assertRaises(ValueError): runtime.begin('r', 0, values, 4, phase=phase)


if __name__ == '__main__': unittest.main()
