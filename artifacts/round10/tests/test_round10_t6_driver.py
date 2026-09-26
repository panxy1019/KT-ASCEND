import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import round10_t6_driver as driver


class T6DriverTests(unittest.TestCase):
    def test_uses_exactly_first_eight_final_rows(self):
        fake={"status":"PASS_ROUND10_T1_PROCESS","generation_attempts":8,"abort_controls":1}
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(driver.t1,"run",return_value=fake) as run:
            out=Path(tmp);(out/"round10-t1-results.json").write_text("{}")
            result=driver.run(out,Path("fixture.json"))
            self.assertEqual(run.call_args.args[0],"F")
            schedule=driver.t1.build_schedule(json.loads((ROOT/"round10-t1-fixture.json").read_text()),"R")[:8]
            self.assertEqual(len(schedule),8)
            self.assertEqual(sum(x["phase"]=="cancel" for x in schedule),1)
            self.assertEqual(result["status"],"PASS_ROUND10_T6_PROCESS")
            self.assertTrue((out/"round10-t6-results.json").exists())


if __name__=="__main__":unittest.main()
