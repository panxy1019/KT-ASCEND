import unittest
from dynamic_telemetry import event
from dynamic_route_summary import correlate_updates


class LightTelemetry(unittest.TestCase):
    def test_host_scope_and_copied_slots(self):
        row=event(1,[0,8],[8,9],128,.02,{'copy_experts_weights_bf16':.01})
        self.assertEqual(row['migration_bytes'],128)
        self.assertEqual(row['selected'],[8,9])
        self.assertIn('NOT_WEIGHT_CERTIFICATION',row['scope'])
        result=correlate_updates([row],[dict(id='q',started_unix=row['time_unix']-1,
            ended_unix=row['time_unix']+1)])
        self.assertEqual(result['actual_trigger_rate'],1)

    def test_slot_drift_rejected(self):
        with self.assertRaises(ValueError):event(1,[0,8],[8],64,.02,{})
        with self.assertRaises(ValueError):event(1,[0,8],[8,8],128,.02,{})

    def test_zero_actual_triggers_is_valid_quality_result(self):
        result=correlate_updates([],[dict(id='q',started_unix=1,ended_unix=2)])
        self.assertEqual(result['actual_trigger_rate'],0)
