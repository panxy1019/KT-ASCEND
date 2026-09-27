import importlib
import unittest


class FrameworkConfigTests(unittest.TestCase):
    def test_version4_joint_budget_needs_real_async_ple_and_explicit_boolean(self):
        api=importlib.import_module('migration_bridge.round9.runtime.framework_config')
        base=dict(schema=4,resident_capacity=32,route_collection=False,frequency=None,
                  prefill=None,ple=None,joint_host_budget=False,joint_device_gate=None)
        self.assertEqual(api.parse_framework(base),base)
        with self.assertRaises(ValueError):api.parse_framework(base | {'joint_host_budget':True})
        with self.assertRaises(ValueError):api.parse_framework(base | {'joint_host_budget':1})
        enabled=base | {'joint_host_budget':True,'ple':dict(mode='ASYNC',device_gate=dict(path='/g',sha256='a'*64)),
                        'joint_device_gate':dict(path='/joint',sha256='b'*64)}
        self.assertEqual(api.parse_framework(enabled),enabled)
        with self.assertRaises(ValueError):api.parse_framework(enabled | {'resident_capacity':64})
        with self.assertRaises(ValueError):api.parse_framework(enabled | {'joint_device_gate':None})
        with self.assertRaises(ValueError):api.parse_framework(base | {'joint_device_gate':enabled['joint_device_gate']})

    def test_version3_async_ple_requires_explicit_receipt_and_defaults_off(self):
        api = importlib.import_module('migration_bridge.round9.runtime.framework_config')
        value = dict(schema=3,resident_capacity=32,route_collection=False,
                     frequency=None,prefill=None,ple=None)
        self.assertEqual(api.parse_framework(value),value)
        spec=dict(mode='ASYNC',device_gate=dict(path='/gate.json',sha256='a'*64))
        self.assertEqual(api.parse_framework(value | {'ple':spec})['ple'],spec)
        for bad in (True,{},spec | {'mode':'PREDICT_DECODE'},spec | {'device_gate':None}):
            with self.assertRaises(ValueError):api.parse_framework(value | {'ple':bad})

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
        self.assertEqual(api.parse_framework(value | {'prefill': layerwise | {'update_residents': True}})['prefill']['update_residents'], True)
        for bad in (layerwise | {'mode': 'STREAMING'},
                    layerwise | {'threshold_tokens': True},
                    layerwise | {'threshold_tokens': 0},
                    layerwise | {'device_gate': None},
                    layerwise | {'update_residents': 1},
                    layerwise | {'unknown': True}):
            with self.assertRaises(ValueError): api.parse_framework(value | {'prefill': bad})
        with self.assertRaises(ValueError):
            api.parse_framework(enabled | {'frequency': dict(path='/frequency.json', sha256='b' * 64)})


if __name__ == '__main__': unittest.main()
