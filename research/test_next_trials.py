import unittest
import numpy as np,pandas as pd
from try_improvements import associate

class IdentityTrialTests(unittest.TestCase):
    def scene(self):
        rng=np.random.default_rng(4);points=[];colors=[];ids=[];rows=[]
        for frame in range(3):
            for obj in range(2):
                oid=2*frame+obj;p=rng.normal(0,.015,(180,3))+[obj,0,0]
                points.extend(p);colors.extend(np.tile([.4,.3,.2],(180,1)));ids.extend([oid]*180)
                rows.append({'frame':frame,'observation_id':oid,'semantic_label':'chair' if frame==0 else 'sofa'})
        return pd.DataFrame(rows),{'points':np.array(points),'colors':np.array(colors),'observation_ids':np.array(ids)}
    def test_label_flip_does_not_split_geometry(self):
        rows,arrays=self.scene();assigned,tracks=associate(rows,arrays)
        self.assertEqual(len(tracks),2)
        rows['semantic_label']='arbitrary changed label';other,_=associate(rows,arrays)
        self.assertEqual(assigned.track_id.tolist(),other.track_id.tolist())
    def test_distinct_same_class_instances_stay_separate(self):
        rows,arrays=self.scene();assigned,tracks=associate(rows,arrays)
        self.assertTrue(all(len(t['rows'])==3 for t in tracks))
        self.assertFalse(assigned.duplicated(['frame','track_id']).any())
if __name__=='__main__':unittest.main()
