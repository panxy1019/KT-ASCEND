import copy
import unittest
from analyze_scheduling import analyze,positions


def fixture():
    requests=[];records=[]
    for index,split in enumerate(('dev','holdout')):
        requests.append(dict(sequence_start=2*index,budget=1,
            prompt=dict(id=split,split=split,input_ids=[1,2])))
        for offset in range(2):
            for layer in range(1,27):
                ids=list(range(6)) if offset==0 else list(range(58,64))
                records.append(dict(sequence=2*index+offset,layer=layer,
                    ids=[ids]*(2 if offset==0 else 1),weights=[[1/6]*6]*(2 if offset==0 else 1)))
    return records,requests


class Scheduling(unittest.TestCase):
    def test_lookahead_excluded(self):
        records,requests=fixture()
        result=analyze(records,requests,8)
        self.assertEqual(result['counts'][1],[2]*6+[0]*58)
        self.assertEqual(result['holdout']['frequency']['positions'],2)

    def test_holdout_never_changes_policy(self):
        records,requests=fixture();changed=copy.deepcopy(records)
        for row in changed:
            if row['sequence']==2:row['ids']=[[58,59,60,61,62,63]]*2
        self.assertEqual(analyze(records,requests,8)['frequency'],analyze(changed,requests,8)['frequency'])

    def test_extra_forward_rejected(self):
        records,requests=fixture();records.append(dict(sequence=99,layer=1,ids=[],weights=[]))
        with self.assertRaises(ValueError):list(positions(records,requests))

    def test_duplicate_rejected(self):
        records,requests=fixture();records.append(records[0])
        with self.assertRaises(ValueError):list(positions(records,requests))


if __name__=='__main__':unittest.main()
