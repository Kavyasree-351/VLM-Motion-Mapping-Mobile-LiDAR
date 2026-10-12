"""Refine measured object surfaces with gated SAM masks; preserve geometry."""
from pathlib import Path
import sys,json,shutil
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.packages'))
import numpy as np,pandas as pd,cv2
from scipy.spatial import cKDTree
import room_upgrade as upgrade, room_views as views, improve_mapping as mapping
OUT=ROOT/'outputs/sam_room'; BASE=ROOT/'outputs/room_refined'; TRIAL=ROOT/'outputs/sam_trial'
OUT.mkdir(exist_ok=True)

def main():
    observations=pd.read_csv(BASE/'detections_visual.csv'); baseline=dict(np.load(BASE/'mask_observations_visual.npz'))
    masks=dict(np.load(TRIAL/'hybrid_masks.npz'))
    model,K,poses,R,t=upgrade.load_geometry()
    cloud=views.o3d.io.read_point_cloud(str(mapping.OUT/'calibrated_supported_cloud.ply'))
    points=np.asarray(cloud.points);colors=np.asarray(cloud.colors); tree=cKDTree(points)
    piles=[];color_piles=[];ids=[];audit=[]
    for frame,group in observations.groupby('frame'):
        _,rgb,depth,valid=mapping.load_frame(int(frame),model)
        xyz=mapping.backproject(depth,K)@R[int(frame)].T+t[int(frame)]
        for row in group.itertuples():
            oid=int(row.observation_id); where=baseline['observation_ids']==oid
            old=baseline['points'][where]; old_colors=baseline['colors'][where]
            p=old;c=old_colors;accepted=False; reason='no_matching_sam_mask';support=None;shift=None;ratio=None
            old_support=float(np.mean(tree.query(old)[0]<.14))
            if str(oid) in masks:
                mask=cv2.erode(masks[str(oid)].astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)&valid
                if mask.sum()>=25:
                    candidate=xyz[mask];candidate_colors=rgb[mask]/255.
                    center=np.median(candidate,axis=0);inside=np.linalg.norm(candidate-center,axis=1)<np.percentile(np.linalg.norm(candidate-center,axis=1),95)
                    candidate=candidate[inside];candidate_colors=candidate_colors[inside]
                    support=float(np.mean(tree.query(candidate)[0]<.14));shift=float(np.linalg.norm(np.median(candidate,axis=0)-np.median(old,axis=0)))
                    ratio=float(len(candidate)/max(row.mask_depth_pixels,1))
                    accepted=support>=old_support-.05 and shift<.25 and .4<ratio<2.5
                    reason='accepted' if accepted else 'geometry_or_area_gate'
                    if accepted:
                        step=max(1,len(candidate)//2000);p=candidate[::step];c=candidate_colors[::step]
                        observations.loc[observations.observation_id==oid,['world_x','world_y','world_z']]=np.median(candidate,axis=0)
                        observations.loc[observations.observation_id==oid,'mask_depth_pixels']=len(candidate)
            piles.append(p);color_piles.append(c);ids.append(np.full(len(p),oid,np.int32))
            audit.append(dict(observation_id=oid,frame=int(frame),class_name=row.class_name,accepted=accepted,reason=reason,baseline_cloud_support=old_support,sam_cloud_support=support,center_shift_pose_units=shift,valid_area_ratio=ratio))
    arrays=dict(points=np.vstack(piles),colors=np.vstack(color_piles),observation_ids=np.concatenate(ids))
    np.savez_compressed(OUT/'mask_observations_visual.npz',**arrays);observations.to_csv(OUT/'detections_visual.csv',index=False)
    pd.DataFrame(audit).to_csv(OUT/'sam_fusion_audit.csv',index=False)
    # Same proposal boxes and labels: reuse CLIP review, updating geometric data.
    reviewed=pd.read_csv(BASE/'detections_reviewed.csv')
    for column in ['world_x','world_y','world_z','mask_depth_pixels']:
        reviewed[column]=observations[column]
    reviewed.to_csv(OUT/'detections_reviewed.csv',index=False)
    shutil.copy2(BASE/'room_geometry.json',OUT/'room_geometry.json')
    geometry=json.loads((OUT/'room_geometry.json').read_text());A=np.array(geometry['plan_rotation_matrix']);floor=geometry['floor_world_y'];ceiling=geometry['ceiling_world_y'];bounds=np.array(geometry['plan_bounds']);lines=geometry['wall_segments_world']
    upgrade.OUT=OUT;views.OUT=OUT
    objects,object_arrays=views.fuse_objects(observations,arrays,A,floor,points)
    views.draw_plan(points,colors,A,floor,poses,bounds,lines,objects,mask_description='YOLOE-26 + gated SAM 3 masks')
    mesh=views.o3d.io.read_triangle_mesh(str(BASE/'room_textured_mesh.ply'))
    if not len(mesh.vertices):raise RuntimeError('Existing measured room mesh missing')
    views.render_3d(mesh,points,colors,objects,object_arrays,A,floor,bounds,ceiling)
    summary=dict(proposals=len(observations),sam_masks=len(masks),accepted_refinements=sum(a['accepted'] for a in audit),fallbacks=sum(not a['accepted'] for a in audit),objects=len(objects),classes=[o['class_name'] for o in objects],geometry='Existing measured mesh and room alignment; object masks refined only',limitations='Geometry gates are consistency proxies, not annotated mask accuracy. Same-room visual references and inferred depth scale remain.')
    (OUT/'sam_room_manifest.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
