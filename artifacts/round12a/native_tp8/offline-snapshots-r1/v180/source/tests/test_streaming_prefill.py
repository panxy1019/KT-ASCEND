import unittest

from migration_bridge.round9.runtime import streaming_prefill as api


class StreamingPrefillTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.runtime = api.StreamingPrefillCoordinator(
            threshold_tokens=4, chunk_tokens=3, event=self.events.append)
        self.request = object()

    def mark_planes(self):
        self.runtime.mark_expert(layers=48)
        self.runtime.mark_ple(async_engaged=True, drained=True)
        self.runtime.mark_context(layers=12, hbm_release=False)

    def test_contiguous_chunks_share_epochs_and_complete_all_planes(self):
        for start, count in ((0, 3), (3, 3), (6, 2)):
            self.assertTrue(self.runtime.begin(
                "r", self.request, placement_epoch=2, allocation_epoch=5,
                total=8, start=start, count=count))
            self.mark_planes()
            self.runtime.finish_chunk()
        self.assertEqual([event["status"] for event in self.events], [
            "STREAMING_CHUNK_BEGIN", "STREAMING_CHUNK_DRAINED",
            "STREAMING_CHUNK_BEGIN", "STREAMING_CHUNK_DRAINED",
            "STREAMING_CHUNK_BEGIN", "STREAMING_CHUNK_DRAINED",
            "STREAMING_PREFILL_COMPLETE",
        ])
        complete = self.events[-1]
        self.assertEqual((complete["chunks"], complete["tokens"]), (3, 8))
        self.runtime.request_end(self.request)
        self.assertEqual(self.runtime.state, "READY")

    def test_hybrid_below_threshold_does_not_create_streaming_state(self):
        self.assertFalse(self.runtime.begin(
            "short", object(), placement_epoch=0, allocation_epoch=0,
            total=3, start=0, count=3))
        self.assertEqual(self.runtime.state, "READY")
        self.assertEqual(self.events, [])

    def test_gap_missing_plane_epoch_switch_and_abort_fail_closed(self):
        self.runtime.begin("r", self.request, placement_epoch=2, allocation_epoch=5,
                           total=8, start=0, count=3)
        self.runtime.mark_expert(layers=48)
        with self.assertRaisesRegex(RuntimeError, "planes"):
            self.runtime.finish_chunk()
        self.assertEqual(self.runtime.state, "BROKEN_RELOAD_REQUIRED")

        runtime = api.StreamingPrefillCoordinator(4, chunk_tokens=3)
        runtime.begin("r", self.request, placement_epoch=2, allocation_epoch=5,
                      total=8, start=0, count=3)
        runtime.mark_expert(layers=48); runtime.mark_ple(async_engaged=False, drained=True)
        runtime.mark_context(layers=12, hbm_release=False); runtime.finish_chunk()
        with self.assertRaisesRegex(ValueError, "contiguous"):
            runtime.begin("r", self.request, placement_epoch=2, allocation_epoch=5,
                          total=8, start=4, count=3)

        runtime = api.StreamingPrefillCoordinator(4, chunk_tokens=3, event=self.events.append)
        runtime.begin("r", self.request, placement_epoch=2, allocation_epoch=5,
                      total=8, start=0, count=3)
        runtime.abort(self.request)
        self.assertEqual(runtime.state, "READY")
        self.assertEqual(self.events[-1]["status"], "STREAMING_ABORTED")


if __name__ == "__main__":
    unittest.main()
