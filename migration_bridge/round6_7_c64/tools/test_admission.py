import unittest
from unittest.mock import patch
from serial_session import admission

class AdmissionTest(unittest.TestCase):
    def check(self,swap='0',active=False,reserve=True):
        reads=[0]
        def read(path):
            if str(path)=='/proc/meminfo':
                return f'MemTotal: 1000000000 kB\nMemAvailable: {900000000 if reserve else 1} kB\nSwapTotal: 8388604 kB\nSwapFree: 8388348 kB\n'
            if str(path)=='/sys/fs/cgroup/memory.swap.current': return swap
            reads[0]+=1
            return f'pswpin 0\npswpout {reads[0] if active else 0}\n'
        with patch('pathlib.Path.read_text',read),patch('serial_session.time.sleep'):
            return admission()
    def test_idle_host_swap(self): self.assertEqual(self.check()['cgroup_swap_bytes'],0)
    def test_container_swap(self):
        with self.assertRaisesRegex(ValueError,'CONTAINER_SWAP'): self.check(swap='4096')
    def test_active_host_swap(self):
        with self.assertRaisesRegex(ValueError,'ACTIVE_HOST'): self.check(active=True)
    def test_reserve(self):
        with self.assertRaisesRegex(ValueError,'MEMORY_RESERVE'): self.check(reserve=False)

if __name__=='__main__': unittest.main()
