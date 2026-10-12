"""Small, isolated pose-conditioned DA3 trial; no predicted mesh replacement."""
from pathlib import Path
import sys,json,time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.packages'))
sys.path.insert(0,str(ROOT/'.models/da3-source/Depth-Anything-3-main/src'))
import numpy as np,torch,cv2
import room_upgrade as upgrade
from depth_anything_3.cfg import create_object
from omegaconf import OmegaConf
from safetensors.torch import load_file
from depth_anything_3.utils.io.input_processor import InputProcessor
from scipy.optimize import least_squares
OUT=ROOT/'outputs/next_trials';OUT.mkdir(exist_ok=True)

def main():
    torch.set_num_threads(4);start=time.perf_counter()
    config=json.loads((ROOT/'.models/da3-small/config.json').read_text())
    net=create_object(OmegaConf.create(config['config'])).eval()
    weights=load_file(str(ROOT/'.models/da3-small/model.safetensors'))
    status=net.load_state_dict({k.removeprefix('model.'):v for k,v in weights.items()},strict=False)
    # Recent source adds optional auxiliary training heads absent in this
    # released checkpoint. Inference uses the main head, whose weights must match.
    assert not status.unexpected_keys and all('output_conv2_aux' in k for k in status.missing_keys),status
    print('Unused auxiliary-head checkpoint differences:',status.missing_keys,flush=True)
    decoder,K,poses,R,t=upgrade.load_geometry()
    frames=np.array([430,450,470,1480,1500,1520])
    predictions=[]
    for group in frames.reshape(2,3):
        ext=np.repeat(np.eye(4)[None],len(group),axis=0)
        ext[:,:3,:3]=R[group].transpose(0,2,1);ext[:,:3,3]=-np.einsum('nij,nj->ni',ext[:,:3,:3],t[group])
        # Centre each batch on the first camera, retaining ARKit pose scale.
        ext=ext@np.linalg.inv(ext[0])
        intrinsic=np.repeat(K[None],len(group),axis=0);intrinsic[:,:2,:]*=720/192
        images=[str(upgrade.DATA/'rgb'/f'{int(f):06d}.png') for f in group]
        image,ex,intr=InputProcessor()(images,ext,intrinsic,294,'upper_bound_resize')
        with torch.inference_mode():output=net(image.unsqueeze(0),torch.as_tensor(ex).unsqueeze(0).float(),torch.as_tensor(intr).unsqueeze(0).float())
        depth=output['depth'].squeeze(0).cpu().numpy();confidence=output['depth_conf'].squeeze(0).cpu().numpy()
        for k,frame in enumerate(group):
            predictions.append((int(frame),cv2.resize(depth[k],(192,256)),cv2.resize(confidence[k],(192,256))))
        print('DA3 predicted',group.tolist(),depth.shape,flush=True)
    # Per-triplet scale/offset uses only neighbouring reference views, then
    # evaluated on the middle view. The target is calibrated LiDAR, not GT.
    results=[];saved={}
    for group in [predictions[:3],predictions[3:]]:
        trainx=[];trainy=[]
        for frame,depth,conf in [group[0],group[2]]:
            raw,rgb,z,valid=upgrade.mapping.load_frame(frame,decoder)
            keep=valid&np.isfinite(depth)&(depth>0)&(conf>=np.percentile(conf,25))
            trainx.extend(depth[keep][::20]);trainy.extend(z[keep][::20])
        x=np.asarray(trainx);y=np.asarray(trainy)
        fit=least_squares(lambda ab:ab[0]*x+ab[1]-y,[np.median(y/x),0],bounds=([1e-5,-2],[100,2]),loss='soft_l1',f_scale=.05)
        frame,depth,conf=group[1];raw,rgb,z,valid=upgrade.mapping.load_frame(frame,decoder)
        pred=depth*fit.x[0]+fit.x[1];keep=valid&np.isfinite(pred)&(pred>0)&(conf>=np.percentile(conf,25))
        relative=np.abs(pred[keep]-z[keep])/z[keep]
        results.append({'heldout_frame':frame,'alignment_reference_frames':[group[0][0],group[2][0]],'pixels':int(keep.sum()),'median_relative_disagreement':float(np.median(relative)),'p90_relative_disagreement':float(np.percentile(relative,90)),'aligned_scale':float(fit.x[0]),'aligned_offset':float(fit.x[1]),'predicted_positive_pixel_fraction':float(np.mean(pred>0))})
        for f,d,c in group:
            saved[f'depth_{f}']=d*fit.x[0]+fit.x[1];saved[f'confidence_{f}']=c
    consistency=[]
    for a,b in [(450,470),(1500,1520)]:
        _,_,za,va=upgrade.mapping.load_frame(a,decoder);_,_,zb,vb=upgrade.mapping.load_frame(b,decoder)
        candidates={}
        for label,z1,z2 in [('calibrated_export',za,zb),('DA3_aligned',saved[f'depth_{a}'],saved[f'depth_{b}'])]:
            xyz=upgrade.mapping.backproject(z1,K)@R[a].T+t[a];local=(xyz-t[b])@R[b];uv=local@K.T;uv=uv[:,:,:2]/np.maximum(uv[:,:,2:],1e-9)
            u,v=uv[:,:,0].astype(np.float32),uv[:,:,1].astype(np.float32)
            target=cv2.remap(z2,u,v,cv2.INTER_LINEAR);targetvalid=cv2.remap(vb.astype(np.uint8),u,v,cv2.INTER_NEAREST)>0
            keep=va&targetvalid&(u>1)&(u<190)&(v>1)&(v<254)&(local[:,:,2]>0)&(target>0)
            candidates[label]=(keep,abs(local[:,:,2]-target)/np.maximum(target,1e-5))
        common=candidates['calibrated_export'][0]&candidates['DA3_aligned'][0]
        consistency.append({'pair':[a,b],'matched_source_pixels':int(common.sum()),'median_reprojection_disagreement':{label:float(np.median(values[1][common])) for label,values in candidates.items()},'caveat':'In-batch consistency proxy, not holdout accuracy; model jointly sees these views and scale anchors.'})
    np.savez_compressed(OUT/'da3_aligned_depth.npz',**saved)
    summary={'model':'DA3-SMALL','frames':frames.tolist(),'device':'cpu','input':'RGB + repaired ARKit poses + inferred intrinsics; direct official network, float32, no Gaussian export','results':results,'in_batch_consistency':consistency,'runtime_seconds':time.perf_counter()-start,'evaluation':'Held-out middle-view agreement with calibrated exported LiDAR; not independent accuracy. No pixels added to final mesh. Six-frame feasibility experiment, not a full model benchmark. Only optional auxiliary ray-head parameters are absent in checkpoint; main depth/confidence parameters loaded; no ray-head pose inference.'}
    (OUT/'da3_trial.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
