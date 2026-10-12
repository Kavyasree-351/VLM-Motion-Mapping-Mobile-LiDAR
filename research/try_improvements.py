"""Isolated geometry-first identity and actual segmentation-mask motion trials."""
from pathlib import Path
import sys,json,time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.packages'))
import room_upgrade as upgrade, room_views, motion_map as motion
import numpy as np,pandas as pd,cv2,torch
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLOE
OUT=ROOT/'outputs/next_trials';OUT.mkdir(exist_ok=True)

def dump(name,data):
    (OUT/name).write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8')

def coverage(a,b,radius=.09):
    if not len(a) or not len(b):return 0.
    return float(min(np.mean(cKDTree(a).query(b)[0]<radius),np.mean(cKDTree(b).query(a)[0]<radius)))

def associate(observations,arrays):
    """Stable IDs use supported 3D overlap and RGB; label never gates matching."""
    tracks=[];assignments=[]
    for frame,group in observations.sort_values(['frame','observation_id']).groupby('frame'):
        candidates=[]
        for _,row in group.iterrows():
            where=arrays['observation_ids']==int(row.observation_id)
            p=arrays['points'][where][::3];c=arrays['colors'][where][::3]
            if len(p)<30:continue
            candidates.append((row,p,c,np.median(p,axis=0),np.median(c,axis=0)))
        costs=np.full((len(candidates),len(tracks)),1e3)
        for i,(row,p,c,center,color) in enumerate(candidates):
            for j,track in enumerate(tracks):
                if np.linalg.norm(center-track['center'])>.75:continue
                overlap=coverage(p,track['points'])
                color_gap=np.linalg.norm(color-track['color'])
                if overlap>=.22 and color_gap<.48:
                    costs[i,j]=1-overlap+.25*color_gap
        matched={}
        if costs.size:
            ii,jj=linear_sum_assignment(costs)
            matched={int(i):int(j) for i,j in zip(ii,jj) if costs[i,j]<1.}
        for i,(row,p,c,center,color) in enumerate(candidates):
            score=None
            if i in matched:
                j=matched[i];track=tracks[j];score=float(costs[i,j])
                joined=np.vstack([track['points'],p]);vox=np.floor(joined/.045).astype(int)
                _,unique=np.unique(vox,axis=0,return_index=True)
                track['points']=joined[unique][::max(1,len(unique)//2500)]
                track['center']=np.median(track['points'],axis=0)
                track['color']=.8*track['color']+.2*color
            else:
                j=len(tracks);track={'points':p,'center':center,'color':color,'rows':[]};tracks.append(track)
            track['rows'].append(int(row.observation_id))
            assignments.append({'observation_id':int(row.observation_id),'frame':int(frame),'track_id':f'instance_{j:03d}','association_cost':score})
    result=pd.DataFrame(assignments)
    return result,tracks

def extract_masks(observations):
    path=OUT/'motion_masks.npz'
    frames=np.sort(observations.frame.unique())[::3]
    if path.exists():
        saved=dict(np.load(path));return saved,frames
    detector=YOLOE(str(upgrade.MODELS/'yoloe-26l-room-visual-seg.pt'));torch.set_num_threads(4)
    masks={}
    for k,frame in enumerate(frames):
        result=detector.predict(str(upgrade.DATA/'rgb'/f'{int(frame):06d}.png'),imgsz=960,conf=.18,device='cpu',verbose=False,retina_masks=True)[0]
        if result.masks is None:continue
        boxes=result.boxes.xyxy.cpu().numpy();seg=result.masks.data.cpu().numpy()
        for rowid,row in observations[observations.frame==frame].iterrows():
            box=np.array([row.x0,row.y0,row.x1,row.y1]);inter=np.maximum(0,np.minimum(box[2:],boxes[:,2:])-np.maximum(box[:2],boxes[:,:2])).prod(axis=1)
            union=(box[2:]-box[:2]).prod()+(boxes[:,2:]-boxes[:,:2]).prod(axis=1)-inter
            i=int(np.argmax(inter/np.maximum(union,1)));iou=inter[i]/max(union[i],1)
            if iou<.8:continue
            mask=cv2.resize(seg[i],(192,256),interpolation=cv2.INTER_NEAREST)>.5
            masks[str(int(row.observation_id))]=cv2.erode(mask.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
        print('Actual motion masks',k+1,'/',len(frames),flush=True)
    np.savez_compressed(path,**masks)
    return masks,frames

def main():
    start=time.perf_counter();cv2.setNumThreads(4);torch.set_num_threads(4)
    observations,arrays,selected=upgrade.detect();observations=room_views.semantic_review(observations)
    good=observations[(observations.semantic_status!='disputed')&(observations.class_name!='person')&(observations.confidence>=.22)].copy()
    assigned,tracks=associate(good,arrays);assigned.to_csv(OUT/'identity_assignments.csv',index=False)
    joined=good.merge(assigned,on=['observation_id','frame'])
    records=[]
    for trackid,g in joined.groupby('track_id'):
        labels=g.groupby('semantic_label').confidence.sum().sort_values(ascending=False)
        records.append({'id':trackid,'class_name':str(labels.index[0]),'alternative_labels':{str(k):float(v) for k,v in labels.items()},'observations':len(g),'frames':g.frame.astype(int).tolist(),'status':'repeated_candidate' if len(g)>=3 else 'unconfirmed','label_is_probability':False})
    dump('identity_tracks.json',records)
    print('Tracks',len(records),'repeated',sum(r['observations']>=3 for r in records),flush=True)
    masks,frames=extract_masks(good)
    model,K,poses,R,t=upgrade.load_geometry();rows=[]
    for frame,g in joined[joined.frame.isin(frames)].groupby('frame'):
        g=g[g.observation_id.astype(str).isin(masks)].copy()
        if g.empty:continue
        g['vlm_label']=g.semantic_label
        byrow={idx:masks[str(int(row.observation_id))] for idx,row in g.iterrows()}
        a=int(frame);b=min(a+20,len(poses)-1)
        for region in ['eroded_box','instance_mask']:
            measured=motion.measure_pair(a,b,g,model,K,R,t,byrow if region=='instance_mask' else None)
            for result,(_,row) in zip(measured,g.iterrows()):
                result.update({'track_id':row.track_id,'observation_id':int(row.observation_id),'region':region});rows.append(result)
        print('Paired box/mask motion',a,flush=True)
    df=pd.DataFrame(rows);df.to_csv(OUT/'motion_box_mask_comparison.csv',index=False)
    states=[]
    for record in records:
        g=df[(df.track_id==record['id'])&(df.region=='instance_mask')&df.ego_refinement_supported&(df.valid_pixels>=30)]
        score=float(g.score.median()) if len(g) else None;reliability=float(g.reliability.median()) if len(g) else None
        votes=int(((g.score>2.8)&(g.reliability>=.35)).sum())
        state='motion_unknown' if len(g)<2 else ('observed_moving_candidate' if votes>=2 and score>2.8 and reliability>=.35 else ('not_observed_moving' if score<=2.8 else 'motion_unknown'))
        states.append({**record,'eligible_pairs':len(g),'median_motion_score':score,'median_reliability':reliability,'moving_votes':votes,'motion_state':state})
    dump('mask_motion_tracks.json',states)
    summary={'trial':'geometry-first association and matched box/mask motion','candidate_tracks':len(records),'repeated_tracks':sum(r['observations']>=3 for r in records),'motion_frames':len(frames),'actual_masks':len(masks),'states':{s:sum(x['motion_state']==s for x in states) for s in sorted({x['motion_state'] for x in states})},'motion_by_region':{},'runtime_seconds':time.perf_counter()-start,'limits':'Class-agnostic conservative overlap association is not full SuperMap/OVI-MAP reproduction; fixed geometry matching can fragment true moving objects. No identity or motion ground truth. Diagnostic flag fraction is not false-positive rate.'}
    for region,g in df.groupby('region'):
        eligible=g[g.ego_refinement_supported&(g.valid_pixels>=30)]
        summary['motion_by_region'][region]={'observations':len(g),'eligible':len(eligible),'median_valid_pixels':float(eligible.valid_pixels.median()),'mean_flag_fraction':float(eligible.fraction_flagged.mean()),'median_score':float(eligible.score.median())}
    dump('trial_summary.json',summary);print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
