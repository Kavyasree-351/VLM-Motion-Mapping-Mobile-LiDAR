"""Paired metrics, support/occlusion trial, wall relations and dense patch queries."""
from pathlib import Path
import sys,json
print('Starting diagnostics',flush=True)
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'.packages'))
import room_upgrade as upgrade
import numpy as np,pandas as pd,cv2,torch
print('Mapping and numerical dependencies imported',flush=True)
from scipy.spatial import cKDTree
from PIL import Image
OUT=ROOT/'outputs/next_trials'
def dump(name,data):(OUT/name).write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8')

def support_trial():
    model,K,poses,R,t=upgrade.load_geometry()
    cloud=upgrade.o3d.io.read_point_cloud(str(upgrade.mapping.OUT/'calibrated_supported_cloud.ply'));tree=cKDTree(np.asarray(cloud.points))
    rows=[]
    for a in [450,750,1050,1500,1720,2000]:
        b=a+20;raw,rgb,z,valid=upgrade.mapping.load_frame(a,model);_,_,z2,v2=upgrade.mapping.load_frame(b,model)
        xyz=upgrade.mapping.backproject(z,K)@R[a].T+t[a];local=(xyz-t[b])@R[b]
        uv=local@K.T;uv=uv[:,:,:2]/np.maximum(uv[:,:,2:],1e-9)
        u,v=uv[:,:,0].astype(np.float32),uv[:,:,1].astype(np.float32)
        target=cv2.remap(z2,u,v,cv2.INTER_LINEAR);visible=cv2.remap(v2.astype(np.uint8),u,v,cv2.INTER_NEAREST)>0
        inside=(u>1)&(u<190)&(v>1)&(v<254)&(local[:,:,2]>0)
        err=abs(local[:,:,2]-target);rel=err/np.maximum(target,1e-5)
        distances=tree.query(xyz.reshape(-1,3))[0].reshape(z.shape)
        # Selection uses source uncertainty only; target disagreement is held out.
        threshold=min(.18,.11+upgrade.mapping.calibration_spread(model,a))
        base=valid&visible&inside
        for mode,limit in [('strict',.11),('adaptive',threshold)]:
            mask=base&(distances<limit)
            rows.append({'frame':a,'mode':mode,'source_gate':limit,'pixels':int(mask.sum()),'median_relative_target_disagreement':float(np.median(rel[mask])),'p90_relative_target_disagreement':float(np.percentile(rel[mask],90)),'occlusion_or_mismatch_fraction':float(np.mean(err[mask]>.1))})
    pd.DataFrame(rows).to_csv(OUT/'support_gate_comparison.csv',index=False)
    return rows

def wall_relations():
    geometry=json.loads((upgrade.OUT/'room_geometry.json').read_text());objects=json.loads((upgrade.OUT/'room_objects.json').read_text());A=np.array(geometry['plan_rotation_matrix'])
    relations=[]
    for obj in objects:
        if obj['class_name'] not in {'door','window','picture'}:continue
        center=np.array(obj['plan_center']);candidates=[]
        for k,line in enumerate(geometry['wall_segments_world']):
            endpoints=np.array(line['endpoints'])@A;d=endpoints[1]-endpoints[0];length=np.linalg.norm(d);d/=length
            delta=center-endpoints[0]
            s=np.dot(delta,d);distance=abs(d[0]*delta[1]-d[1]*delta[0])
            if -.1<=s<=length+.1:candidates.append((float(distance),k,float(s),float(length)))
        candidates.sort()
        distance,k,s,length=candidates[0] if candidates else (None,None,None,None)
        relations.append({'object_id':obj['id'],'class_name':obj['class_name'],'wall_id':k if distance is not None and distance<.25 else None,'wall_distance':distance,'status':'observed_wall_relation_candidate' if distance is not None and distance<.25 else 'unresolved','opening_verified':False})
    dump('wall_object_relations.json',relations)
    return relations

def patch_language():
    from transformers import CLIPModel,CLIPProcessor
    from verify_semantics import features
    torch.set_num_threads(4)
    model=CLIPModel.from_pretrained(ROOT/'.models/clip',local_files_only=True).eval();processor=CLIPProcessor.from_pretrained(ROOT/'.models/clip',local_files_only=True)
    records=[];crops=[]
    for frame in [450,750,1050,1500,1720,2000]:
        image=Image.open(upgrade.DATA/'rgb'/f'{frame:06d}.png').convert('RGB')
        for row in range(4):
            for col in range(3):
                box=[max(0,col*240-40),max(0,row*240-40),min(720,(col+1)*240+40),min(960,(row+1)*240+40)]
                crops.append(image.crop(box));records.append({'frame':frame,'pixel_box':box,'type':'image_patch_not_object'})
    vectors=[]
    with torch.inference_mode():
        for start in range(0,len(crops),12):
            f=features(model.get_image_features(**processor(images=crops[start:start+12],return_tensors='pt')));vectors.append((f/f.norm(dim=-1,keepdim=True)).cpu().numpy())
        queries=['a turquoise wall','a white ceiling','a wooden floor','an orange beanbag','a glass window','a refrigerator']
        f=features(model.get_text_features(**processor(text=[f'a photo of {q}' for q in queries],padding=True,return_tensors='pt')));text=(f/f.norm(dim=-1,keepdim=True)).cpu().numpy()
    vectors=np.vstack(vectors);scores=vectors@text.T;results=[]
    for k,query in enumerate(queries):
        best=np.argsort(scores[:,k])[-3:][::-1]
        results.append({'query':query,'status':'unverified_ranked_patches','matches':[{**records[i],'cosine':float(scores[i,k])} for i in best],'caveat':'Cosine rank does not establish object presence; refrigerator query must not be interpreted as a detected refrigerator.'})
    np.savez_compressed(OUT/'dense_patch_embeddings.npz',embeddings=vectors);dump('dense_patch_queries.json',results);dump('dense_patch_index.json',records)
    return {'patches':len(records),'frames':6,'memory_bytes':vectors.nbytes,'status':'coarse CPU patch-language feasibility; not full FUS3DMaps or dense voxel map'}

def main():
    torch.set_num_threads(4);cv2.setNumThreads(4)
    df=pd.read_csv(OUT/'motion_box_mask_comparison.csv');a=df[df.region=='eroded_box'].set_index('observation_id');b=df[df.region=='instance_mask'].set_index('observation_id')
    common=a.index.intersection(b.index);a=a.loc[common];b=b.loc[common]
    eligible=a.ego_refinement_supported&b.ego_refinement_supported&(a.valid_pixels>=30)&(b.valid_pixels>=30)
    paired={'same_eligible_observations':int(eligible.sum()),'box_mean_flag_fraction':float(a.loc[eligible,'fraction_flagged'].mean()),'mask_mean_flag_fraction':float(b.loc[eligible,'fraction_flagged'].mean()),'box_median_pixels':float(a.loc[eligible,'valid_pixels'].median()),'mask_median_pixels':float(b.loc[eligible,'valid_pixels'].median()),'caveat':'Paired diagnostic comparison. No true motion labels; lower flags alone do not prove improvement.'}
    print('Paired motion',paired,flush=True)
    support=support_trial();print('Support gates complete',flush=True)
    relations=wall_relations();print('Wall relations complete',flush=True)
    patches=patch_language();print('Language patches complete',flush=True)
    result={'paired_motion':paired,'support_trial':support,'wall_relations':relations,'patch_language':patches};dump('diagnostics.json',result);print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
