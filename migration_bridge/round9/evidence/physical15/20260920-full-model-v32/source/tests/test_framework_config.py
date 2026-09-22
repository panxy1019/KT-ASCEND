import importlib
import unittest


class FrameworkConfigTests(unittest.TestCase):
    def test_explicit_version_capacity_and_default_off(self):
        api = importlib.import_module('migration_bridge.round9.runtime.framework_config')
        self.assertIsNone(api.parse_framework(None))
        config = dict(schema=1, resident_capacity=64, route_collection=False, frequency=None)
        self.assertEqual(api.parse_framework(config), config)
        for bad in ({**config, 'schema': 2}, {**config, 'resident_capacity': 63},
                    {**config, 'route_collection': 1}, {**config, 'unknown': True},
                    {**config, 'frequency': {'path': 'x', 'sha256': 'x'}}):
            with self.assertRaises(ValueError): api.parse_framework(bad)
        self.assertEqual(api.parse_framework({**config, 'resident_capacity': 32})['resident_capacity'], 32)

    def test_buffer_identity_has_separate_request_placement_allocation_lifetimes(self):
        api = importlib.import_module('migration_bridge.round9.runtime.framework_config')
        old = api.BufferIdentity('r', 2, 7)
        self.assertNotEqual(old, api.BufferIdentity('r', 2, 8))
        self.assertNotEqual(old, api.BufferIdentity('r', 3, 7))
        self.assertNotEqual(old, api.BufferIdentity('s', 2, 7))
        for values in (('', 0, 0), ('r', -1, 0), ('r', 0, -1), ('r', True, 0)):
            with self.assertRaises(ValueError): api.BufferIdentity(*values)

    def test_version2_explicit_layerwise_contract_and_off(self):
        api = importlib.import_module('migration_bridge.round9.runtime.framework_config')
        value = dict(schema=2, resident_capacity=32, route_collection=False,
                     frequency=None, prefill=None)
        self.assertEqual(api.parse_framework(value), value)
        layerwise = dict(mode='LAYERWISE', threshold_tokens=8,
                         device_gate=dict(path='/evidence/gate.json', sha256='a' * 64))
        enabled = value | {'prefill': layerwise}
        self.assertEqual(api.parse_framework(enabled), enabled)
        for bad in (layerwise | {'mode': 'STREAMING'},
                    layerwise | {'threshold_tokens': True},
                    layerwise | {'threshold_tokens': 0},
                    layerwise | {'device_gate': None},
                    layerwise | {'unknown': True}):
            with self.assertRaises(ValueError): api.parse_framework(value | {'prefill': bad})
        with self.assertRaises(ValueError):
            api.parse_framework(enabled | {'frequency': dict(path='/frequency.json', sha256='b' * 64)})


if __name__ == '__main__': unittest.main()
