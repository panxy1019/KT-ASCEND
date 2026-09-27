import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROUND9 = Path(__file__).parents[1]
AGGREGATOR = ROUND9 / "tools/aggregate_cmmlu_reference.py"
DRIVER = ROUND9 / "tests/full_model_cmmlu.py"
MANIFEST = (
    ROUND9
    / "evidence/physical15/20260920-full-model-v3/cmmlu-frozen-v2/manifest.json"
)
PROFILE_SHA = "2a88116fb6a67f31149494104106857a88ba4fcf4078a6b2606b3d1fe64e69c4"


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False) + "\n")


class AggregateCMMLUTests(unittest.TestCase):
    def fixture(self, root):
        manifest = json.loads(MANIFEST.read_text())
        manifest_bytes = MANIFEST.read_bytes()
        manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
        driver_sha = hashlib.sha256(DRIVER.read_bytes()).hexdigest()
        implementation = "a" * 64
        image = "sha256:" + "b" * 64
        slices = []
        for run_index, start in enumerate((0, 128)):
            run = root / f"run-{run_index}"
            slice_root = run / f"cmmlu-{start}"
            slice_root.mkdir(parents=True)
            (slice_root / "manifest.json").write_bytes(manifest_bytes)
            write(
                slice_root / "freeze-ack.json",
                {
                    "manifest_sha256": manifest_sha,
                    "local_commit": "e66726f2f6a07aad85a974a15e0aba7604a20852",
                },
            )
            admission = {
                "status": "STATIC_BOUND_READY_FOR_MONITORED_FULL_MODEL_RUN",
                "implementation_sha256": implementation,
                "profile_sha256": PROFILE_SHA,
            }
            container_id = f"container-{run_index}"
            container = {
                "Id": container_id,
                "Image": image,
                "Config": {"Env": ["ROUND9_MEMORY_OBSERVER=0"]},
            }
            restart = {
                "status": "PASS_RESTART4",
                "container_id": container_id,
                "implementation_sha256": implementation,
                "profile_sha256": PROFILE_SHA,
                "protocol_sha256": "c" * 64,
            }
            write(run / "preflight/admission.json", admission)
            write(run / "container-created.json", container)
            restart_path = run / "observer-off-v1/summary.json"
            write(restart_path, restart)
            restart_sha = hashlib.sha256(restart_path.read_bytes()).hexdigest()
            runtime = {
                "implementation_sha256": implementation,
                "profile_sha256": PROFILE_SHA,
                "image": image,
                "observer": False,
                "admission_status": admission["status"],
            }
            write(
                slice_root / "summary.json",
                {
                    "status": "REFERENCE_SLICE_SCORED_PENDING_AUDIT",
                    "manifest_sha256": manifest_sha,
                    "freeze_ack": json.loads(
                        (slice_root / "freeze-ack.json").read_text()
                    ),
                    "start": start,
                    "count": 128,
                    "invalid": 0,
                    "driver_sha256": driver_sha,
                    "runtime_identity": runtime,
                    "restart_artifact_sha256": restart_sha,
                    "container_id": container_id,
                },
            )
            with (slice_root / "raw.jsonl").open("w") as raw:
                for row in manifest["rows"][start : start + 128]:
                    scores = [
                        [0.0 if letter == row["answer"] else -1.0, token]
                        for letter, token in zip("ABCD", row["candidate_ids"])
                    ]
                    request = {
                        "rid": "r9-cmmlu-" + row["id"],
                        "input_ids": row["input_ids"],
                        "sampling_params": {
                            "temperature": 0,
                            "max_new_tokens": 1,
                            "stop": row["stop"],
                        },
                        "return_logprob": True,
                        "logprob_start_len": -1,
                        "top_logprobs_num": 0,
                        "token_ids_logprob": row["candidate_ids"],
                    }
                    response = {
                        "meta_info": {
                            "id": request["rid"],
                            "completion_tokens": 1,
                            "output_token_ids_logprobs": [scores],
                        }
                    }
                    raw.write(
                        json.dumps(
                            {"id": row["id"], "request": request, "response": response},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            slices.append(slice_root)
        return slices

    def aggregate(self, root, slices):
        output = root / "aggregate.json"
        process = subprocess.run(
            [
                sys.executable,
                str(AGGREGATOR),
                "--manifest",
                str(MANIFEST),
                "--slices",
                *(str(path) for path in slices),
                "--output",
                str(output),
            ],
            text=True,
            capture_output=True,
        )
        return process, output

    def test_two_real_frozen_128_slices_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            process, output = self.aggregate(root, self.fixture(root))
            self.assertEqual(process.returncode, 0, process.stderr)
            result = json.loads(output.read_text())
            self.assertEqual(result["ordered_ranges"], [[0, 128], [128, 256]])
            self.assertEqual(result["count"], 256)
            self.assertEqual(result["correct"], 256)

    def test_tampered_restart_container_or_admission_is_rejected(self):
        cases = {
            "restart": lambda run: write(
                run / "observer-off-v1/summary.json", {"status": "FAIL"}
            ),
            "container": lambda run: write(
                run / "container-created.json",
                {
                    "Id": "tampered-container",
                    "Image": "sha256:" + "b" * 64,
                    "Config": {"Env": ["ROUND9_MEMORY_OBSERVER=0"]},
                },
            ),
            "admission": lambda run: write(
                run / "preflight/admission.json",
                {
                    "status": "STATIC_BOUND_READY_FOR_MONITORED_FULL_MODEL_RUN",
                    "implementation_sha256": "d" * 64,
                    "profile_sha256": PROFILE_SHA,
                },
            ),
        }
        for name, mutate in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                slices = self.fixture(root)
                mutate(slices[0].parent)
                process, output = self.aggregate(root, slices)
                self.assertNotEqual(process.returncode, 0)
                self.assertFalse(output.exists())

    def test_slice_shape_and_distinct_restarts_are_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            slices = self.fixture(root)
            process, _ = self.aggregate(root, slices[:1])
            self.assertNotEqual(process.returncode, 0)
            summary = json.loads((slices[1] / "summary.json").read_text())
            summary["container_id"] = "container-0"
            write(slices[1] / "summary.json", summary)
            process, _ = self.aggregate(root, slices)
            self.assertNotEqual(process.returncode, 0)


if __name__ == "__main__":
    unittest.main()
