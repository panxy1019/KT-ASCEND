import unittest
from monitor_resources import kb_fields


class ResourceParsing(unittest.TestCase):
    def test_bytes(self):
        self.assertEqual(kb_fields('RssAnon:    12 kB\nRssFile: 7 kB\n'),
                         {'RssAnon':12288,'RssFile':7168})

    def test_missing_not_zero(self):
        self.assertNotIn('Pss',kb_fields('VmRSS: 0 kB\n'))

    def test_meminfo_names(self):
        self.assertEqual(kb_fields('Active(file): 2 kB\nHugePages_Total: 0\n'),
                         {'Active(file)':2048})


if __name__=='__main__':unittest.main()
