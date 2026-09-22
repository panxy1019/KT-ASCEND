import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


def sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False,sort_keys=True,separators=(",", ":")).encode()).hexdigest()


class FrozenCorpusTests(unittest.TestCase):
    def test_joint_freeze(self):
        pair = json.loads((ROOT/"corpora/frozen-corpora/pair_freeze.json").read_text())
        seen = set()
        for name in ("qd2", "hd2"):
            value = json.loads((ROOT/f"corpora/frozen-corpora/{name}.json").read_text())
            stored = value.pop("sha256")
            self.assertEqual(sha(value), stored)
            self.assertEqual(pair[name],stored)
            self.assertEqual(len(value["prompts"]),128)
            self.assertEqual(value["protocol"]["positions"],64)
            for prompt in value["prompts"]:
                self.assertEqual(sha(prompt["input_ids"]), prompt["input_ids_sha256"])
                self.assertEqual(hashlib.sha256(prompt["text"].encode()).hexdigest(),prompt["prompt_sha256"])
                self.assertNotIn(prompt["text"],seen)
                seen.add(prompt["text"])
                self.assertLessEqual(len(prompt["input_ids"])+64,512)

    def test_smoke_is_separate(self):
        smoke=json.loads((ROOT/"corpora/smoke_source.json").read_text())
        self.assertFalse(smoke["qualification_allowed"])
        texts={p["text"] for name in ("qd2","hd2") for p in json.loads((ROOT/f"corpora/frozen-corpora/{name}.json").read_text())["prompts"]}
        self.assertFalse(texts.intersection(smoke["prompts"]))


if __name__ == "__main__":
    unittest.main()
