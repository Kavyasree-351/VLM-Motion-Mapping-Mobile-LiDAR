"""VLMM-inspired motion attributes, with export-calibration quality in the noise model.

This is a CPU adaptation using DIS flow and bounding-box regions, not a claimed
reproduction of VLMM's RAFT/instance-mask benchmark. Frame indices are not seconds.
Reference: https://arxiv.org/abs/2607.16173v2 (corrected residual covariance).
"""
from pathlib import Path
import sys,os,json,re
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.packages'))
os.environ['HF_HUB_OFFLINE']='1'
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from scipy.stats import chi2
from geometry_probe import DATA,cameras
import improve_mapping as mapping

OUT=mapping.OUT

def point_covariance(u,v,z,K,R,sigma_depth,sigma_pixel=.6):
    u,v,z=np.broadcast_arrays(u,v,z)
    J=np.zeros((*z.shape,3,3));J[...,0,0]=z/K[0,0];J[...,1,1]=z/K[1,1]
    J[...,0,2]=(u-K[0,2])/K[0,0];J[...,1,2]=(v-K[1,2])/K[1,1];J[...,2,2]=1
    variance=np.zeros((*z.shape,3));variance[...,:2]=np.asarray(sigma_pixel)[...,None]**2;variance[...,2]=np.asarray(sigma_depth)**2
    local=(J*variance[...,None,:])@J.swapaxes(-1,-2)
    return R@local@R.T

def residual_covariance(cov_a,cov_b,refinement_R,pose_floor=.012):
    # v2 correction: transform endpoint A, not B.
    return cov_b+refinement_R@cov_a@refinement_R.T+np.eye(3)*pose_floor**2

def motion_statistics(residual,covariance):
    if len(residual)<30:return {'valid_pixels':len(residual),'score':None,'fraction_flagged':None,'coherence':None,'reliability':0.,'median_displacement':None}
    d=np.sqrt(np.maximum(0,np.einsum('ni,ni->n',residual,np.linalg.solve(covariance,residual[...,None])[...,0])))
    norm=np.linalg.norm(residual,axis=1)
    coherence=float(np.linalg.norm(np.mean(residual/np.maximum(norm[:,None],1e-8),axis=0)))
    fraction=float(np.mean(d>np.sqrt(chi2.ppf(.95,3))))
    return {'valid_pixels':len(residual),'score':float(np.percentile(d,75)),'fraction_flagged':fraction,'coherence':coherence,'reliability':fraction*coherence,'median_displacement':float(np.median(norm))}

def rigid_fit(a,b):
    ca=a.mean(axis=0);cb=b.mean(axis=0);U,_,Vt=np.linalg.svd((a-ca).T@(b-cb))
    R=Vt.T@np.diag([1,1,np.linalg.det(Vt.T@U.T)])@U.T
    return R,cb-R@ca

def robust_ego_fit(a,b):
    if len(a)<100:return np.eye(3),np.zeros(3),False
    keep=np.ones(len(a),bool);R=np.eye(3);t=np.zeros(3)
    for _ in range(6):
        R,t=rigid_fit(a[keep],b[keep]);err=np.linalg.norm(b-(a@R.T+t),axis=1)
        keep=err<max(.04,float(np.percentile(err,65)))
        if keep.sum()<50:return np.eye(3),np.zeros(3),False
    valid=Rotation.from_matrix(R).magnitude()<np.deg2rad(3) and np.linalg.norm(t)<.20
    return (R,t,True) if valid else (np.eye(3),np.zeros(3),False)

def measure_pair(a,b,boxes,model,K,R,t,instance_masks=None):
    raw1,rgb1,z1,valid1=mapping.load_frame(a,model);raw2,rgb2,z2,valid2=mapping.load_frame(b,model)
    gray1=cv2.cvtColor(rgb1,cv2.COLOR_RGB2GRAY);gray2=cv2.cvtColor(rgb2,cv2.COLOR_RGB2GRAY)
    flow=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    ab=flow.calc(gray1,gray2,None);ba=flow.calc(gray2,gray1,None)
    v,u=np.indices(z1.shape,dtype=np.float32);uu=u+ab[:,:,0];vv=v+ab[:,:,1]
    reverse=cv2.remap(ba,uu,vv,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=1e3)
    fb=np.linalg.norm(ab+reverse,axis=2)
    target=cv2.remap(z2,uu,vv,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    targetvalid=cv2.remap(valid2.astype(np.uint8),uu,vv,cv2.INTER_NEAREST)>0
    # A texture check rejects optically unobservable white-wall correspondences.
    texture=cv2.cornerMinEigenVal(gray1,3,3)
    valid=valid1&targetvalid&(uu>=1)&(uu<191)&(vv>=1)&(vv<255)&(fb<1.2)&(texture>1e-5)
    local1=mapping.backproject(z1,K);local2=np.stack([(uu-K[0,2])*target/K[0,0],(vv-K[1,2])*target/K[1,1],target],axis=-1)
    world1=local1@R[a].T+t[a];world2=local2@R[b].T+t[b]
    static=valid.copy()
    for rowid,box in boxes.iterrows():
        if instance_masks is not None:
            static[instance_masks[rowid]]=False
            continue
        x0,x1=np.clip(np.array([box.x0,box.x1])*192/720,0,192).astype(int);y0,y1=np.clip(np.array([box.y0,box.y1])*256/960,0,256).astype(int)
        static[y0:y1,x0:x1]=False
    candidates=np.flatnonzero(static.ravel())
    if len(candidates)>4000:candidates=candidates[::int(np.ceil(len(candidates)/4000))]
    refinement,shift,ego_valid=robust_ego_fit(world1.reshape(-1,3)[candidates],world2.reshape(-1,3)[candidates])
    residual=world2-(world1@refinement.T+shift)
    sensor1=.012+.004*z1**2;sensor2=.012+.004*target**2
    # Unlike raw sensor-depth uncertainty, export decoding adds a shared,
    # frame-level error source. Use empirical spread conservatively at each ray;
    # do not turn correlated pixels into independent posterior votes.
    sigma1=np.sqrt(sensor1**2+mapping.calibration_spread(model,a)**2)
    sigma2=np.sqrt(sensor2**2+mapping.calibration_spread(model,b)**2)
    c1=point_covariance(u,v,z1,K,R[a],sigma1,.6+.25*fb)
    c2=point_covariance(uu,vv,target,K,R[b],sigma2,.6+.25*fb)
    cov=residual_covariance(c1,c2,refinement)
    cov_sensor=residual_covariance(point_covariance(u,v,z1,K,R[a],sensor1),point_covariance(uu,vv,target,K,R[b],sensor2),refinement)
    measurements=[]
    for rowid,box in boxes.iterrows():
        x0,x1=np.clip(np.array([box.x0,box.x1])*192/720,0,192).astype(int);y0,y1=np.clip(np.array([box.y0,box.y1])*256/960,0,256).astype(int)
        dx=max(1,int(.18*(x1-x0)));dy=max(1,int(.18*(y1-y0)))
        if instance_masks is None:
            mask=np.zeros_like(valid);mask[y0+dy:y1-dy,x0+dx:x1-dx]=True
        else:
            mask=instance_masks[rowid].copy()
        mask &= valid
        stats=motion_statistics(residual[mask],cov[mask]);sensorstats=motion_statistics(residual[mask],cov_sensor[mask])
        world_center=np.median(world1[mask],axis=0) if mask.any() else np.array([0.,0.,0.])
        measurements.append({'frame_a':a,'frame_b':b,'class_name':box.vlm_label,'ego_refinement_supported':bool(ego_valid),'world_x':float(world_center[0]),'world_z':float(world_center[2]),'sensor_only_score':sensorstats['score'],'sensor_only_fraction_flagged':sensorstats['fraction_flagged'],**stats})
    return measurements

def parse_motion_query(text):
    text=text.lower().strip()
    text=re.sub(r'[^\w\s]',' ',text)
    text=' '.join(text.split())
    if re.search(r'\b(now|currently|right now)\b',text):return {'intent':'needs_fresh_observation','semantic':''}
    if ' and ' in text or ' or ' in text:return {'intent':'unsupported_composition','semantic':''}
    patterns=[('immovability',r'not movable|immovable|permanently fixed'),('movability',r'could be moved|can be moved|could move|can move|movable|repositioned|portable'),('observed_static',r'not moving|did not move|stayed still|remained still'),('observed_motion',r'observed moving|was moving|were moving|has moved|moving|moved'),('reobserve',r'needs another observation|reobserve|observe again')]
    for intent,pattern in patterns:
        if re.search(pattern,text):
            semantic=re.sub(pattern,' ',text)
            semantic=re.sub(r'\b(what|which|things|objects|the|a|an|that|were|was|is|are|has|during|observed|frames|this|episode|i|could|can|should|be)\b',' ',semantic)
            return {'intent':intent,'semantic':' '.join(semantic.split())}
    return {'intent':'semantic','semantic':text}

def execute_attribute_query(text,elements,semantic_scores=None):
    parsed=parse_motion_query(text);intent=parsed['intent'];result={'query':text,**parsed,'matches':[],'scope':'recorded episode only'}
    if intent in ['needs_fresh_observation','unsupported_composition']:
        result['status']=intent;return result
    if parsed['semantic'] and semantic_scores is None:
        result['status']='needs_semantic_grounding';return result
    for k,e in enumerate(elements):
        if parsed['semantic'] and semantic_scores is not None and semantic_scores[k]<.23:continue
        selected=(intent=='semantic' or (intent=='movability' and e['movability_prior_score']>=.6) or (intent=='immovability' and e['movability_prior_score']<.4) or (intent=='observed_motion' and e['motion_class']=='observed_moving') or (intent=='observed_static' and e['motion_class']=='movable_not_observed_moving') or intent=='reobserve')
        if selected:result['matches'].append({'id':e['id'],'class_name':e['class_name'],'motion_class':e['motion_class'],'position':[e['world_x'],e['world_z']],'motion_reliability_score':e['motion_reliability_score'],'reobservation_priority':e['reobservation_priority']})
    if intent=='reobserve':result['matches'].sort(key=lambda e:e['reobservation_priority'],reverse=True)
    result['status']='matches' if result['matches'] else 'no_supported_match'
    return result

def main():
    cv2.setNumThreads(4)
    p=np.load(OUT/'depth_calibration.npz');model={k:p[k] for k in ['frames','coeff','inverse','residual_spread','anchor_counts']};K=p['K_half'].copy();K[:2,:]*=192/360
    poses=np.loadtxt(DATA/'poses.txt');R,t=cameras(poses,'c2w_opengl')
    boxes=pd.read_csv(OUT/'vlm_verified_boxes.csv');boxes=boxes[(boxes.vlm_status=='accepted')&(~boxes.class_name.isin(['person']))]
    rows=[]
    for a,g in boxes.groupby('frame'):
        a=int(a);b=min(a+20,len(poses)-1)
        if a==b:continue
        rows.extend(measure_pair(a,b,g,model,K,R,t))
        print('Motion pair',a,b,flush=True)
    observations=pd.DataFrame(rows);observations.to_csv(OUT/'motion_observations.csv',index=False)
    # Semantic affordance prior is a CLIP relative property score, not a physical
    # affordance measurement or a calibrated probability.
    import torch
    from transformers import CLIPModel,CLIPProcessor
    from verify_semantics import features
    import query_map as language
    torch.set_num_threads(4);records,embeddings=language.build_index()
    clip=CLIPModel.from_pretrained(ROOT/'.models/clip',local_files_only=True).eval();processor=CLIPProcessor.from_pretrained(ROOT/'.models/clip',local_files_only=True)
    with torch.inference_mode():
        prompts=['a movable piece of furniture that can be repositioned','a permanently fixed structural part of a building']
        f=features(clip.get_text_features(**processor(text=prompts,return_tensors='pt',padding=True)));f=f/f.norm(dim=-1,keepdim=True)
    prior_similarity=embeddings@f.cpu().numpy().T
    prior=1/(1+np.exp(-(prior_similarity[:,0]-prior_similarity[:,1])/.05))
    elements=[]
    for k,record in enumerate(records):
        g=observations[observations.class_name==record['class_name']].copy()
        distance=np.linalg.norm(g[['world_x','world_z']].to_numpy()-[record['world_x'],record['world_z']],axis=1)
        g=g[(distance<record['spread_radius']+.65)&g.ego_refinement_supported&(g.valid_pixels>=30)]
        score=float(g.score.median()) if len(g) else None;reliability=float(g.reliability.median()) if len(g) else 0.
        eligible=len(g);moving_votes=int(((g.score>2.8)&(g.reliability>=.35)).sum()) if eligible else 0
        if eligible<2:state='motion_unknown'
        elif moving_votes>=2 and score>2.8 and reliability>=.35:state='observed_moving'
        elif score<=2.8 and prior[k]>=.6:state='movable_not_observed_moving'
        elif score<=2.8 and prior[k]<.4:state='not_observed_moving_low_movability_prior'
        else:state='motion_unknown'
        element={**record,'id':f"{record['class_name']}_{record['cluster_id']}",'observed_motion_score':score,'motion_reliability_score':reliability,'movability_prior_score':float(prior[k]),'movability_prior_source':'CLIP relative comparison of movable-furniture vs fixed-structure prompts; heuristic, uncalibrated','motion_class':state,'eligible_motion_pairs':eligible,'moving_pair_votes':moving_votes,'last_seen_frame':int(g.frame_b.max()) if eligible else None,'motion_units':'Mahalanobis score; displacement in pose units; no velocity without timestamps','residual_covariance':'rotated endpoint sensor noise + empirical export-decoder residual spread + assumed pose floor','uncertainty_calibrated':False,'association_caveat':'label and spatial gate; not a full identity tracker','reobservation_priority':float(record['spread_radius']+(1 if state=='motion_unknown' else 0)+.25*(1-reliability))}
        elements.append(element)
    mapping.dump('vlmm_elements.json',elements)
    queries=['what could be moved?','which beanbag could be moved?','what was observed moving?','what was not moving?','what is moving now?','what needs another observation?']
    results=[]
    for q in queries:
        parsed=parse_motion_query(q);scores=None
        if parsed['semantic']:
            with torch.inference_mode():
                vector=features(clip.get_text_features(**processor(text=[f"a photo of {parsed['semantic']}."],return_tensors='pt',padding=True)))
                vector=vector/vector.norm(dim=-1,keepdim=True)
            scores=embeddings@vector[0].cpu().numpy()
        results.append(execute_attribute_query(q,elements,scores))
    mapping.dump('vlmm_queries.json',results)
    fig,axes=plt.subplots(1,2,figsize=(12,5));labels=[e['class_name'] for e in elements]
    axes[0].bar(labels,[e['movability_prior_score'] for e in elements],color='#2a807e');axes[0].set(title='Semantic movability score (not observed motion)',ylim=(0,1),ylabel='Uncalibrated relative CLIP score')
    axes[1].bar(labels,[e['observed_motion_score'] or 0 for e in elements],color='#b68843');axes[1].axhline(2.8,color='#b7372c',linestyle='--',label='Illustrative noise gate');axes[1].legend();axes[1].set(title='Camera-compensated motion evidence',ylabel='Median episode Mahalanobis score')
    fig.tight_layout();fig.savefig(OUT/'vlmm_motion_attributes.png',dpi=180);plt.close(fig)
    summary={'reference':'https://arxiv.org/abs/2607.16173v2','adaptation':'CPU DIS flow, eroded box regions, static-background trimmed rigid fit; not paper reproduction','objects':len(elements),'states':{state:sum(e['motion_class']==state for e in elements) for state in sorted(set(e['motion_class'] for e in elements))},'motion_observation_pairs':len(observations),'sensor_only_pixel_flag_fraction_mean':float(observations.sensor_only_fraction_flagged.dropna().mean()),'with_export_uncertainty_pixel_flag_fraction_mean':float(observations.fraction_flagged.dropna().mean()),'limitation':'Real sequence has no annotated moving-object ground truth; flag fractions are diagnostics, not false-positive rates or recall. Noise inflation alone can suppress true motion. Synthetic static and moving tests are separate.'}
    mapping.dump('vlmm_summary.json',summary);print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
