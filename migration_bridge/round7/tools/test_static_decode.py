import unittest
from analyze_static_decode import summarize_prefill
from test_dynamic_routes import rows


class PrefillSeparation(unittest.TestCase):
    def test_exact_prefill_keys_exclude_decode_and_tail(self):
        result=summarize_prefill(rows('1',3)+rows('2',4)+rows('2',5),{(1,3)})
        self.assertEqual(result['assignments'],156)
        self.assertEqual(result['model_token_rows'],1)
        self.assertEqual(result['cpu_hits'],104)

    def test_duplicate_prefill_layer(self):
        with self.assertRaisesRegex(ValueError,'DUPLICATE'):summarize_prefill(rows('1',3)+rows('1',3)[:1],{(1,3)})

    def test_missing_layer(self):
        with self.assertRaisesRegex(ValueError,'LAYER_COMPLETENESS'):summarize_prefill(rows('1',3)[:-1],{(1,3)})

    def test_wrong_route_shape(self):
        data=rows('1',3);data[0]['assignments']=7
        with self.assertRaisesRegex(ValueError,'ROUTE_SHAPE'):summarize_prefill(data,{(1,3)})

    def test_missing_prefill_forward(self):
        with self.assertRaisesRegex(ValueError,'MISSING_PREFILL'):summarize_prefill(rows('1',3),{(1,3),(1,6)})


if __name__=='__main__':unittest.main()
