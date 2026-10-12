"""Evaluate tracked masks with the existing uncertainty-aware motion model."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'.packages'))
import numpy as np,pandas as pd
import room_upgrade as upgrade, motion_map as motion
OUT=ROOT/'outputs/sam_trial'
model,K,poses,R,t=upgrade.load_geometry()
observations=pd.read_csv(ROOT/'outputs/room_refined/detections_reviewed.csv')
hybrid=dict(np.load(OUT/'hybrid_masks.npz'))
sofa=dict(np.load(OUT/'sam31_masks.npz'));beanbag=dict(np.load(OUT/'sam31_beanbag_masks.npz'))
records=[]
for a in [400,450]:
    g=observations[(observations.frame==a)&observations.semantic_label.isin(['sofa','beanbag'])].copy()
    g['vlm_label']=g.semantic_label
    for kind in ['box','sam3_refinement','sam31_tracking']:
        masks=None
        if kind!='box':
            masks={}
            for idx,row in g.iterrows():
                if kind=='sam3_refinement':mask=hybrid[str(int(row.observation_id))]
                else:mask=(beanbag if row.semantic_label=='beanbag' else sofa)[f'{a-400}_0']
                masks[idx]=motion.cv2.erode(mask.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
        result=motion.measure_pair(a,a+20,g,model,K,R,t,instance_masks=masks)
        for item,(_,row) in zip(result,g.iterrows()):
            records.append({**item,'region':kind,'observation_id':int(row.observation_id),'class_name':row.semantic_label})
pd.DataFrame(records).to_csv(OUT/'sam_motion_comparison.csv',index=False)
print(pd.DataFrame(records)[['class_name','region','valid_pixels','score','fraction_flagged','ego_refinement_supported']].to_string(index=False))
