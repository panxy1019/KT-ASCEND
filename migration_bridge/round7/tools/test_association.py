import unittest
from associate_forwards import associate


class Association(unittest.TestCase):
    def test_overlap_unused_final_is_attributed_not_fitted(self):
        requests=[dict(input_ids=[11,12],output_ids=[7,8])]
        steps=[dict(input_ids=[11,12],positions=[0,1]),
               dict(input_ids=[7],positions=[2]),dict(input_ids=[8],positions=[3])]
        result=associate(steps,requests)
        self.assertEqual(result['kept_indices'],[0,1])
        self.assertEqual(result['unused_final_token_indices'],[2])

    def test_unknown_and_missing_rejected(self):
        request=[dict(input_ids=[11,12],output_ids=[7,8])]
        with self.assertRaises(ValueError):associate([dict(input_ids=[99],positions=[0])],request)
        with self.assertRaises(ValueError):associate([dict(input_ids=[11,12],positions=[0,1])],request)


if __name__=='__main__':unittest.main()
