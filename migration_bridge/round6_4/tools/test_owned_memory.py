import unittest
from monitor_owned_memory import descendants, proc_memory


class MemoryTests(unittest.TestCase):
    def test_excludes_neighbor_process(self):
        rows = {10:(1,"guard"), 11:(10,"server"), 12:(11,"worker"), 20:(1,"other")}
        self.assertEqual(descendants(rows,[10]), {10,11,12})
        self.assertEqual(descendants(rows,[]), set())

    def test_actual_npu_smi_schema(self):
        text = "\tProcess id:450013  Process name:sglangschedul     Process memory(MB):30792\n\tProcess id:448693  Process name:python Process memory(MB):112\n"
        self.assertEqual(proc_memory(text), {450013:30792,448693:112})


if __name__ == "__main__":
    unittest.main()
