import unittest
from analyze_stability_memory import summarize


class MemoryTests(unittest.TestCase):
    def data(self,grow=False):
        report=dict(status="STABILITY_RUNTIME_PASS_PENDING_HBM_REVIEW",requests=18,
                    generated_tokens=1152,pending_drained=True,
                    memory_samples=[dict(time_unix=i*10,summed_owned_rss_bytes=(100+i if grow else 100)) for i in range(19)])
        samples=[dict(status="SAMPLED",time_unix=i*10,owned_hbm_mb_by_pid={"1":100},other_process_hbm_mb=0) for i in range(19)]
        return report,samples

    def test_plateau_is_descriptive_not_zero_leak(self):
        result=summarize(*self.data())
        self.assertFalse(result["all_cycle_ends_grow_review_required"])
        self.assertIn("not a proof",result["observation_limit"])

    def test_growth_requires_review(self):
        self.assertTrue(summarize(*self.data(True))["all_cycle_ends_grow_review_required"])

    def test_monitor_failure_cannot_pass(self):
        report,samples=self.data()
        samples[2]["status"]="MONITOR_ERROR"
        with self.assertRaisesRegex(ValueError,"MONITOR_ERROR"):
            summarize(report,samples)

    def test_missing_hbm_cannot_pass(self):
        report,samples=self.data()
        with self.assertRaisesRegex(ValueError,"INSUFFICIENT"):
            summarize(report,[])


if __name__ == "__main__":
    unittest.main()
