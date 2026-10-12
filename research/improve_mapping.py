"""Geometry-conditioned recovery and view-balanced mapping of exported LiDAR.

Run: python research/improve_mapping.py
Camera calibration is inferred, not substituted for an original calibration file.
All accuracy metrics below are geometric consistency proxies, not ground truth.
"""
from pathlib import Path
import sys, os, json, time, hashlib
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.packages'))
os.environ['MPLCONFIGDIR'] = str(ROOT / '.matplotlib')
os.environ['YOLO_CONFIG_DIR'] = str(ROOT / '.yolo')
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy import sparse
from scipy.sparse.linalg import lsqr
from scipy.optimize import lsq_linear
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree
from scipy.ndimage import binary_dilation
import open3d as o3d
from geometry_probe import DATA, cameras, intrinsic, epipolar

OUT = ROOT / 'outputs/research_improved'
OUT.mkdir(parents=True, exist_ok=True)
cv2.setNumThreads(4)
RNG = np.random.default_rng(20261008)

def dump(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')

def irls(A, y, reg, prior, rounds=8):
    """Huber IRLS with sparse temporal regularization, O(samples + frames)."""
    w = np.ones(len(y)); p = prior.copy()
    for _ in range(rounds):
        Aw=A.multiply(np.sqrt(w)[:,None])
        M=sparse.vstack([Aw,reg],format='csr')
        yy=np.r_[y*np.sqrt(w),np.zeros(reg.shape[0])]
        # Both parameters are positive spans/endpoints; underconstrained anchors
        # cannot reverse near/far ordering or generate negative distances.
        result=lsq_linear(M,yy,bounds=(.025,8.),tol=1e-7,lsmr_tol='auto',max_iter=150)
        if not result.success:raise RuntimeError('Constrained depth solve failed: '+result.message)
        p=result.x
        residual=A@p-y
        w=np.minimum(1.,.08/np.maximum(np.abs(residual),1e-9))
    return p

def fit_depth(samples, inverse=False, selected=None):
    """Framewise affine depth/disparity recovered in pose translation units."""
    frames=np.arange(0,2091,20); n=len(frames)
    index=np.clip(np.rint(samples[:,0]/20).astype(int),0,n-1)
    raw=samples[:,4]/255.;z=samples[:,5]
    y=1/z if inverse else z
    scale=np.maximum(y,.2)
    if selected is None:selected=np.ones(len(samples),bool)
    ii=np.flatnonzero(selected); row=np.arange(len(ii))
    basis=1-raw if inverse else raw
    A=sparse.coo_matrix((np.r_[basis[ii]/scale[ii],1/scale[ii]],(np.r_[row,row],np.r_[2*index[ii],2*index[ii]+1])),shape=(len(ii),2*n)).tocsr()
    target=y[ii]/scale[ii]
    # Fixed regularization. Neighbour anchors are 20 frames apart; no test tuning.
    reg=sparse.lil_matrix((2*(n-1)+2*n,2*n))
    strength=.30
    for k in range(n-1):
        for q in range(2):reg[2*k+q,2*k+q]=-strength;reg[2*k+q,2*(k+1)+q]=strength
    for k in range(2*n):reg[2*(n-1)+k,k]=.001
    p=irls(A,target,reg.tocsr(),np.tile([1.,1.],n))
    coeff=p.reshape(n,2)
    if inverse:coeff=np.column_stack([-coeff[:,0],coeff[:,0]+coeff[:,1]])
    return {'frames':frames,'coeff':coeff,'inverse':inverse}

def predict_depth(model, frame, raw):
    a=np.interp(frame,model['frames'],model['coeff'][:,0])
    b=np.interp(frame,model['frames'],model['coeff'][:,1])
    value=a*np.asarray(raw)/255.+b
    if model['inverse']:
        value=np.divide(1.,value,out=np.full_like(value,np.nan,dtype=float),where=value>.03)
    return value

def attach_residual_spread(model,samples):
    """Empirical local residual spread. This is not a calibrated probability."""
    pred=predict_depth(model,samples[:,0],samples[:,4])
    residual=np.abs(pred-samples[:,5]);spread=[];counts=[]
    for f in model['frames']:
        near=(np.abs(samples[:,0]-f)<=40)&np.isfinite(residual)
        counts.append(int(near.sum()))
        spread.append(max(.025,float(np.percentile(residual[near],75))) if near.sum()>=15 else .4)
    model['residual_spread']=np.array(spread);model['anchor_counts']=np.array(counts)
    return model

def calibration_spread(model,frame):
    if 'residual_spread' not in model:return np.zeros_like(np.asarray(frame),dtype=float)+.1
    return np.interp(frame,model['frames'],model['residual_spread'])

def evaluate_depth(samples,model):
    z=samples[:,5];pred=predict_depth(model,samples[:,0],samples[:,4])
    valid=np.isfinite(pred)&(pred>0)
    err=np.abs(pred[valid]-z[valid])/z[valid]
    return {'samples':len(z),'valid_fraction':float(valid.mean()),'median_abs_relative_error':float(np.median(err)),'mean_abs_relative_error':float(np.mean(np.minimum(err,10))),'rmse_pose_units':float(np.sqrt(np.mean((pred[valid]-z[valid])**2))),'fraction_within_10_percent':float(np.mean(err<.1))}

def original_depth(frame,raw):
    d=np.load(DATA/'depth'/f'{int(frame):06d}.npy');v=d[np.isfinite(d)&(d>0)]
    low,high=np.percentile(v,[5,98])
    return .5+2*np.clip((raw-low)/max(high-low,1e-6),0,1)

def calibrate():
    pack=np.load(ROOT/'research/triangulated_samples.npz');s=pack['samples'];K=pack['K'];mode=str(pack['mode'])
    # Entire frame-pairs held out. Remove repeated image keypoints from training.
    hold=((np.minimum(s[:,0],s[:,1]).astype(int)//40)%5)==0
    keys=np.column_stack([s[:,0],np.rint(s[:,2]*2),np.rint(s[:,3]*2)]).astype(int)
    hold_keys={tuple(k) for k in keys[hold]}
    train=(~hold)&np.array([tuple(k) not in hold_keys for k in keys])
    results=[];models=[]
    for inverse in [False,True]:
        m=fit_depth(s,inverse,train);models.append(m)
        result=evaluate_depth(s[hold],m);result['method']='temporally_regularized_inverse_depth' if inverse else 'temporally_regularized_affine_depth';results.append(result)
    # Model family selected on TRAINING performance; test data never chooses it.
    training_scores=[evaluate_depth(s[train],m)['median_abs_relative_error'] for m in models]
    chosen=int(np.argmin(training_scores));model=models[chosen]
    originals=np.array([original_depth(int(f),np.array([d]))[0] for f,d in s[hold][:,[0,4]]])
    z=s[hold,5];err=np.abs(originals-z)/z
    baseline={'method':'original_framewise_pseudo_depth','samples':len(z),'valid_fraction':1.,'median_abs_relative_error':float(np.median(err)),'mean_abs_relative_error':float(err.mean()),'rmse_pose_units':float(np.sqrt(np.mean((originals-z)**2))),'fraction_within_10_percent':float(np.mean(err<.1))}
    # A single affine fit is a separate baseline: temporal calibration ablation.
    x=s[train,4]/255.;y=s[train,5];A=np.column_stack([x,np.ones(len(x))])
    p=np.linalg.lstsq(A,y,rcond=None)[0]
    for _ in range(8):
        w=np.minimum(1,.15/np.maximum(abs(A@p-y),1e-9));p=np.linalg.lstsq(A*np.sqrt(w[:,None]),y*np.sqrt(w),rcond=None)[0]
    global_model={'frames':np.array([0,2090]),'coeff':np.tile(p,(2,1)),'inverse':False}
    global_result=evaluate_depth(s[hold],global_model);global_result['method']='single_global_affine_depth'
    # Full production fit uses all observations AFTER independent evaluation.
    production=attach_residual_spread(fit_depth(s,bool(chosen)),s)
    np.savez_compressed(OUT/'depth_calibration.npz',frames=production['frames'],coeff=production['coeff'],inverse=production['inverse'],K_half=K,residual_spread=production['residual_spread'],anchor_counts=production['anchor_counts'])
    poses=np.loadtxt(DATA/'poses.txt');pairs=np.load(ROOT/'research/feature_pairs.npz')
    R,t=cameras(poses,mode);Rold,told=cameras(poses,'c2w_opencv');geometry={}
    for label,RR,tt,KK in [('original',Rold,told,intrinsic(375.)),('axis_fix_only',R,t,intrinsic(375.)),('axis_and_focal_fix',R,t,K)]:
        errors=[]
        for i,j in np.unique(pairs['pairs'],axis=0):
            if (i//40)%5!=0 or np.linalg.norm(t[i]-t[j])<.035:continue
            mask=(pairs['pairs'][:,0]==i)&(pairs['pairs'][:,1]==j)
            errors.extend(epipolar(pairs['uv1'][mask],pairs['uv2'][mask],RR[i],tt[i],RR[j],tt[j],KK))
        geometry[label]={'heldout_matches':len(errors),'median_epipolar_error_half_resolution_px':float(np.median(errors)),'p90_epipolar_error_half_resolution_px':float(np.percentile(errors,90))}
    # Same held-out RGB correspondences for geometric fusion comparison.
    # Evaluate before point filtering: discarding difficult points cannot improve it.
    pp=pairs['pairs'];holdpairs=(pp[:,0]//40)%5==0
    validpairs=np.linalg.norm(t[pp[:,0]]-t[pp[:,1]],axis=1)>.035
    ii=np.flatnonzero(holdpairs&validpairs);depth_cache={};fusion_errors={}
    for label,RR,tt,KK,decoder in [('original',Rold,told,intrinsic(375.),None),('axis_and_focal_fix_only',R,t,K,None),('geometry_and_depth_calibrated',R,t,K,model)]:
        distances=[];pairids=[]
        for i,j in np.unique(pp[ii],axis=0):
            mask=holdpairs&validpairs&(pp[:,0]==i)&(pp[:,1]==j)
            uv=[pairs['uv1'][mask],pairs['uv2'][mask]];world=[];good=[]
            for f,u in zip([i,j],uv):
                if f not in depth_cache:depth_cache[f]=np.load(DATA/'depth'/f'{f:06d}.npy')
                d=depth_cache[f];x=np.clip(u[:,0]*192/360,0,191).astype(int);y=np.clip(u[:,1]*256/480,0,255).astype(int);raw=d[y,x]
                z=original_depth(f,raw) if decoder is None else predict_depth(decoder,f,raw)
                ray=np.column_stack([(u[:,0]-KK[0,2])/KK[0,0],(u[:,1]-KK[1,2])/KK[1,1],np.ones(len(u))])
                world.append((ray*z[:,None])@RR[f].T+tt[f]);good.append((raw>3)&(raw<252)&np.isfinite(z)&(z>0))
            valid=good[0]&good[1];delta=np.linalg.norm(world[0][valid]-world[1][valid],axis=1)
            distances.extend(delta);pairids.extend([(i,j)]*len(delta))
        distances=np.asarray(distances);pairids=np.asarray(pairids)
        # Pair-cluster bootstrap respects correlations between points in a view pair.
        groups=np.unique(pairids,axis=0);group_values=[distances[np.all(pairids==g,axis=1)] for g in groups]
        rng=np.random.default_rng(42);boot=[]
        for _ in range(200):
            draw=rng.integers(0,len(groups),len(groups));boot.append(float(np.median(np.concatenate([group_values[k] for k in draw]))))
        fusion_errors[label]={'samples':len(distances),'median_cross_view_3d_disagreement_pose_units':float(np.median(distances)),'p90_disagreement_pose_units':float(np.percentile(distances,90)),'pair_cluster_bootstrap_95_interval':np.percentile(boot,[2.5,97.5]).tolist()}
    metrics={'depth_holdout':[baseline,global_result]+results,'chosen_family':results[chosen]['method'],'family_training_scores':training_scores,'training_samples_after_duplicate_exclusion':int(train.sum()),'holdout_samples':int(hold.sum()),'geometry_holdout':geometry,'split':'every fifth 40-frame anchor pair held out; same rounded image keypoints excluded from training; fixed regularization; model family selected on training only','caveat':'Correspondence-derived consistency proxies, not external ground-truth accuracy; same sequence used, not cross-scene generalization.'}
    metrics['unfiltered_correspondence_fusion_holdout']=fusion_errors
    # Stronger depth test: remove all observations touching 200-frame blocks 0/5/10.
    block=((s[:,0].astype(int)//200)%5==0)|((s[:,1].astype(int)//200)%5==0)
    block_model=fit_depth(s,bool(chosen),~block)
    metrics['temporal_block_depth_holdout']=evaluate_depth(s[block],block_model)
    metrics['temporal_block_depth_holdout']['note']='All depth anchors touching withheld 200-frame blocks removed. Intrinsics fixed from camera experiment; this is a depth calibration robustness test.'
    testmodel=attach_residual_spread(model,s[train])
    relerr=np.abs(predict_depth(testmodel,s[hold,0],s[hold,4])-s[hold,5])/s[hold,5]
    order=np.argsort(calibration_spread(testmodel,s[hold,0])/s[hold,5])
    metrics['uncertainty_risk_coverage']=[{'coverage':c,'mean_abs_relative_error':float(relerr[order[:max(1,int(c*len(order))) ]].mean())} for c in [.25,.5,.75,1.]]
    dump('evaluation.json',metrics)
    print('Calibration evaluation:',json.dumps(metrics,indent=2),flush=True)
    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    axes[0].scatter(s[hold,4],s[hold,5],s=3,alpha=.2,color='#526173');axes[0].set(xlabel='Exported depth value',ylabel='Triangulated depth (pose units)',title='A single global scale is insufficient')
    axes[1].plot(production['frames'],production['coeff'][:,0],label='slope');axes[1].plot(production['frames'],production['coeff'][:,1],label='offset');axes[1].legend();axes[1].set(xlabel='Frame',title='Recovered framewise calibration')
    vals=[baseline['median_abs_relative_error'],global_result['median_abs_relative_error'],results[chosen]['median_abs_relative_error']]
    axes[2].bar(['Original','Global fit','Framewise fit'],np.array(vals)*100,color=['#bc5544','#bda36c','#257a79']);axes[2].set(ylabel='Median relative error (%)',title='Held-out geometric depth consistency')
    fig.tight_layout();fig.savefig(OUT/'calibration_evaluation.png',dpi=170);plt.close(fig)
    return production,K,mode,s,metrics

def keyframes(poses):
    R=Rotation.from_quat(poses[:,3:]).as_matrix(); chosen=[0]
    for i in range(10,len(poses),5):
        j=chosen[-1]
        angle=Rotation.from_matrix(R[j].T@R[i]).magnitude()
        distance=np.linalg.norm(poses[i,:3]-poses[j,:3])
        if distance>=.07 or angle>=np.deg2rad(6) or i-j>=100:chosen.append(i)
    if chosen[-1]!=len(poses)-1:chosen.append(len(poses)-1)
    return np.array(chosen)

def load_frame(i,model):
    raw=np.load(DATA/'depth'/f'{i:06d}.npy')
    rgb=cv2.imread(str(DATA/'rgb'/f'{i:06d}.png'))[:,:,::-1]
    rgb=cv2.resize(rgb,(192,256),interpolation=cv2.INTER_AREA)
    depth=predict_depth(model,i,raw).astype(np.float32)
    valid=(raw>3)&(raw<252)&np.isfinite(depth)&(depth>.25)&(depth<8)
    # Avoid mixed pixels on depth discontinuities, where triangulation is unreliable.
    grad=np.hypot(cv2.Sobel(depth,cv2.CV_32F,1,0,ksize=3)/8,cv2.Sobel(depth,cv2.CV_32F,0,1,ksize=3)/8)
    valid &= grad<.08+.04*depth
    return raw,rgb,depth,valid

def backproject(depth,K):
    v,u=np.indices(depth.shape)
    return np.stack([(u-K[0,2])*depth/K[0,0],(v-K[1,2])*depth/K[1,1],depth],axis=-1)

def voxel_fuse(points,colors,frame_ids,weights,voxel=.035):
    """One vote per view per voxel: density cannot masquerade as confidence."""
    xyz=np.floor(points/voxel).astype(np.int32)
    viewkey=np.column_stack([xyz,frame_ids])
    _,index,inverse=np.unique(viewkey,axis=0,return_index=True,return_inverse=True)
    sums=np.bincount(inverse,weights=weights)
    p=np.column_stack([np.bincount(inverse,weights=weights*points[:,k])/np.maximum(sums,1e-9) for k in range(3)])
    c=np.column_stack([np.bincount(inverse,weights=weights*colors[:,k])/np.maximum(sums,1e-9) for k in range(3)])
    # Each frame has capped weight, regardless of pixels landing in the voxel.
    w=np.minimum(1.,sums/np.maximum(np.bincount(inverse),1))
    unique,groups=np.unique(xyz[index],axis=0,return_inverse=True)
    total=np.bincount(groups,weights=w);views=np.bincount(groups)
    fused=np.column_stack([np.bincount(groups,weights=w*p[:,k])/np.maximum(total,1e-9) for k in range(3)])
    color=np.column_stack([np.bincount(groups,weights=w*c[:,k])/np.maximum(total,1e-9) for k in range(3)])
    var=np.column_stack([np.bincount(groups,weights=w*p[:,k]**2)/np.maximum(total,1e-9)-fused[:,k]**2 for k in range(3)])
    sigma=np.sqrt(np.maximum(var,0).sum(axis=1))
    return fused,color,views,sigma

def save_cloud(name,points,colors):
    cloud=o3d.geometry.PointCloud();cloud.points=o3d.utility.Vector3dVector(points);cloud.colors=o3d.utility.Vector3dVector(np.clip(colors,0,1))
    if not o3d.io.write_point_cloud(str(OUT/name),cloud):raise IOError(name)
    return cloud

def mapping(model,Khalf,mode,poses):
    started=time.perf_counter();selected=keyframes(poses);R,t=cameras(poses,mode)
    K=Khalf.copy();K[:2,:]*=192/360
    frames={int(i):load_frame(int(i),model) for i in selected}
    allp=[];allc=[];allf=[];allw=[];ungatedp=[];ungatedc=[];records=[]
    rays=[]
    for count,i in enumerate(selected):
        i=int(i);raw,rgb,depth,valid=frames[i]
        local=backproject(depth,K);world=local@R[i].T+t[i]
        agree=np.zeros(depth.shape,dtype=np.int16);testable=np.zeros_like(agree);residual=np.zeros_like(depth)
        neighbours=selected[np.abs(np.arange(len(selected))-count)<=3]
        for j in neighbours:
            j=int(j)
            if j==i or np.linalg.norm(t[i]-t[j])<.025:continue
            _,_,other,othervalid=frames[j]
            cam=(world-t[j])@R[j]
            z=cam[:,:,2];u=np.rint(K[0,0]*cam[:,:,0]/np.maximum(z,1e-6)+K[0,2]).astype(int);v=np.rint(K[1,1]*cam[:,:,1]/np.maximum(z,1e-6)+K[1,2]).astype(int)
            inside=(z>.1)&(u>=0)&(u<192)&(v>=0)&(v<256)&valid
            uu=np.clip(u,0,191);vv=np.clip(v,0,255)
            target=other[vv,uu];delta=np.abs(z-target)
            # A point behind the measured surface is occluded, not a contradiction.
            visible=inside&othervalid[vv,uu]&(z<=target+.12+.06*target)
            ok=visible&(delta<.10+.05*target)
            agree+=ok;testable+=visible;residual+=np.where(ok,delta,0)
        v,u=np.indices(depth.shape);sample=((u%2)==0)&((v%2)==0)
        keep=valid&sample&(agree>=2)
        geometry_weight=np.exp(-calibration_spread(model,i)/(.12+.06*depth))
        confidence=np.clip(agree/np.maximum(testable,1),0,1)*np.exp(-residual/np.maximum(agree,1)/.15)*geometry_weight
        allp.append(world[keep]);allc.append(rgb[keep]/255.);allf.append(np.full(keep.sum(),i));allw.append(confidence[keep])
        un=valid&sample;ungatedp.append(world[un]);ungatedc.append(rgb[un]/255.)
        records.append({'frame':i,'valid_sampled':int(un.sum()),'cross_view_supported':int(keep.sum()),'supported_fraction':float(keep.sum()/max(un.sum(),1)),'calibration_residual_spread':float(calibration_spread(model,i))})
        # Horizontal rays near the camera plane, excluding floor and ceiling.
        raykeep=keep&(np.abs(world[:,:,1]-t[i,1])<.45)&((u%6)==0)&((v%6)==0)
        rays.append((i,world[raykeep][:,[0,2]],confidence[raykeep]))
        if count%20==0:print('Cross-view mapping',count,'/',len(selected),flush=True)
    points=np.vstack(allp);colors=np.vstack(allc);ids=np.concatenate(allf);weights=np.concatenate(allw)
    p,c,views,sigma=voxel_fuse(points,colors,ids,weights)
    supported=(views>=2)&(sigma<.06)
    save_cloud('calibrated_supported_cloud.ply',p[supported],c[supported])
    up=np.vstack(ungatedp);uc=np.vstack(ungatedc)
    save_cloud('calibrated_before_consistency.ply',up,uc)
    np.savez_compressed(OUT/'voxel_evidence.npz',points=p,colors=c,independent_views=views,spread=sigma,supported=supported)
    pd.DataFrame(records).to_csv(OUT/'frame_quality.csv',index=False)
    elapsed=time.perf_counter()-started
    summary={'keyframes':len(selected),'original_final_keyframes':419,'before_consistency_points':len(up),'cross_view_supported_points':len(points),'fused_voxels':len(p),'multi_view_supported_voxels':int(supported.sum()),'runtime_seconds':elapsed,'voxel_size_pose_units':.035,'minimum_independent_keyframe_views':2,'memory_strategy':'bounded keyframe depth cache and early pixel sampling; sparse calibration; NumPy aggregation; no dense 10M-point float64 concatenation'}
    dump('mapping_metrics.json',summary)
    print('Mapping:',json.dumps(summary),flush=True)
    return p[supported],c[supported],selected,K,R,t,frames,rays

def ray_map(points,selected,t,rays):
    xmin,zmin=np.min(np.vstack([points[:,[0,2]],t[selected][:,[0,2]]]),axis=0)-.3
    xmax,zmax=np.max(np.vstack([points[:,[0,2]],t[selected][:,[0,2]]]),axis=0)+.3
    res=.04;W=int(np.ceil((xmax-xmin)/res))+1;H=int(np.ceil((zmax-zmin)/res))+1
    hits=np.zeros((H,W),np.float32);free=np.zeros_like(hits)
    def grid(xz):return np.floor((xz-[xmin,zmin])/res).astype(int)
    for i,endpoints,weights in rays:
        start=grid(t[i,[0,2]]);end=grid(endpoints)
        hitview=np.zeros((H,W),np.uint8);freeview=np.zeros_like(hitview)
        for e,w in zip(end,weights):
            if w<.35:continue
            n=int(max(abs(e-start)))+1
            if n<2:continue
            line=np.rint(np.linspace(start,e,n)).astype(int)
            inside=(line[:,0]>=0)&(line[:,0]<W)&(line[:,1]>=0)&(line[:,1]<H)
            line=line[inside]
            if len(line)<2:continue
            freeview[line[:-2,1],line[:-2,0]]=1
            hitview[line[-1,1],line[-1,0]]=1
        hits+=hitview;free+=freeview
    logodds=np.clip(.85*hits-.35*free,-6,6)
    evidence=hits+free
    unknown=evidence<2
    state=np.full((H,W),-1,np.int8)
    state[(~unknown)&(logodds<-.8)]=0
    state[(~unknown)&(logodds>.8)&(hits>=2)]=1
    state[(~unknown)&(state==-1)]=2
    rgba=np.zeros((H,W,3),dtype=np.uint8);rgba[:]=[209,215,220]
    rgba[state==0]=[250,248,240];rgba[state==1]=[49,69,83];rgba[state==2]=[192,162,97]
    extent=[xmin,xmin+W*res,zmin,zmin+H*res]
    np.savez_compressed(OUT/'occupancy_evidence.npz',state=state,hit_views=hits,free_views=free,log_odds=logodds,extent=extent,resolution=res)
    fig,axes=plt.subplots(1,2,figsize=(12,5.2))
    axes[0].imshow(rgba,origin='lower',extent=extent);axes[0].plot(t[:,0],t[:,2],color='#cb5947',lw=1.3)
    axes[0].set(title='Measured ray evidence: free / occupied / unknown',xlabel='World X (pose units)',ylabel='World Z (pose units)')
    height=points[:,1];use=(height>np.median(t[:,1])-.8)&(height<np.median(t[:,1])+.5)
    axes[1].scatter(points[use,0],points[use,2],s=.6,c=height[use],cmap='viridis',alpha=.7);axes[1].plot(t[:,0],t[:,2],color='#cb5947',lw=1)
    axes[1].set(title='Supported surfaces; floor/ceiling excluded',xlabel='World X (pose units)',ylabel='World Z (pose units)')
    for ax in axes:ax.set_aspect('equal');ax.grid(alpha=.15)
    fig.tight_layout();fig.savefig(OUT/'evidence_map.png',dpi=180);plt.close(fig)
    dump('occupancy_metrics.json',{'unknown_cells':int(np.sum(state==-1)),'free_cells':int(np.sum(state==0)),'occupied_cells':int(np.sum(state==1)),'ambiguous_cells':int(np.sum(state==2)),'ray_frames':len(selected),'resolution_pose_units':res,'note':'Exploratory ray evidence with inferred calibration; not a validated navigation costmap. Unknown is preserved, no hole filling.'})
    return rgba,extent

def semantic_landmarks(model,K,R,t,selected,frames,map_image,extent):
    from ultralytics import YOLO
    from sklearn.cluster import DBSCAN
    cache=OUT/'detector_boxes.csv'
    if cache.exists():boxes=pd.read_csv(cache)
    else:
        detector=YOLO(str(ROOT/'yolo11n.pt')); rows=[]
        for n,i in enumerate(selected):
            result=detector.predict(str(DATA/'rgb'/f'{int(i):06d}.png'),conf=.35,verbose=False,device='cpu')[0]
            for box,score,label in zip(result.boxes.xyxy.cpu().numpy(),result.boxes.conf.cpu().numpy(),result.boxes.cls.cpu().numpy()):
                rows.append({'frame':int(i),'x0':float(box[0]),'y0':float(box[1]),'x1':float(box[2]),'y1':float(box[3]),'confidence':float(score),'class_name':detector.names[int(label)]})
            if n%20==0:print('Semantic keyframe',n,'/',len(selected),flush=True)
        boxes=pd.DataFrame(rows,columns=['frame','x0','y0','x1','y1','confidence','class_name']);boxes.to_csv(cache,index=False)
    # Open-vocabulary verification is cached separately, preserving detector output.
    verified=OUT/'vlm_verified_boxes.csv'
    if verified.exists():
        boxes=pd.read_csv(verified)
        boxes=boxes[boxes['vlm_status']=='accepted'].copy()
        boxes['detector_class_name']=boxes['class_name']
        boxes['class_name']=boxes['vlm_label']
    measurements=[]
    for _,b in boxes.iterrows():
        if b.class_name=='person':continue
        i=int(b.frame);_,_,d,valid=frames[i]
        x0,x1=int(b.x0*192/720),int(b.x1*192/720);y0,y1=int(b.y0*256/960),int(b.y1*256/960)
        # Median of inner bounding box avoids a single centre pixel and boundary bleed.
        dx=max(1,int((x1-x0)*.2));dy=max(1,int((y1-y0)*.2))
        x0=max(0,x0+dx);x1=min(192,x1-dx);y0=max(0,y0+dy);y1=min(256,y1-dy)
        if x1<=x0 or y1<=y0:continue
        region=d[y0:y1,x0:x1];mask=valid[y0:y1,x0:x1]
        vv,uu=np.where(mask)
        if len(uu)<20:continue
        vals=region[mask];med=float(np.median(vals));mad=float(1.4826*np.median(abs(vals-med)))
        good=np.abs(vals-med)<max(.10,2.5*mad);uu=uu[good]+x0;vv=vv[good]+y0;zz=vals[good]
        pts=np.column_stack([(uu-K[0,2])*zz/K[0,0],(vv-K[1,2])*zz/K[1,1],zz])@R[i].T+t[i]
        center=np.median(pts,axis=0)
        measurements.append({'frame':i,'class_name':b.class_name,'confidence':b.confidence,'world_x':center[0],'world_y':center[1],'world_z':center[2],'depth_mad':mad,'calibration_spread':float(calibration_spread(model,i)),'region_pixels':len(pts)})
    df=pd.DataFrame(measurements);df.to_csv(OUT/'semantic_observations.csv',index=False)
    landmarks=[]
    for label,g in df.groupby('class_name'):
        pos=g[['world_x','world_z']].to_numpy();clusters=DBSCAN(eps=.45,min_samples=3).fit_predict(pos)
        for cluster in sorted(set(clusters)-{-1}):
            gg=g.iloc[np.flatnonzero(clusters==cluster)].sort_values('confidence',ascending=False).drop_duplicates('frame')
            if len(gg)<3:continue
            # Count independent camera positions, not nearby timestamps.
            camera_pos=t[gg.frame.to_numpy(int)][:,[0,2]]
            unique_views=[]
            for k,camera in enumerate(camera_pos):
                if not unique_views or np.min(np.linalg.norm(camera_pos[unique_views]-camera,axis=1))>.10:unique_views.append(k)
            if len(unique_views)<3:continue
            gg=gg.iloc[unique_views];pos=gg[['world_x','world_z']].to_numpy()
            measurement_sigma=np.sqrt(gg.depth_mad.to_numpy()**2+gg.calibration_spread.to_numpy()**2+.04**2)
            w=gg.confidence.to_numpy()/np.maximum(.08,measurement_sigma)**2;w/=w.sum()
            center=(w[:,None]*pos).sum(axis=0);cov=(pos-center).T@(w[:,None]*(pos-center))
            # Spatial scatter, not a statistically calibrated confidence interval.
            cov+=np.eye(2)*float(np.sum(w*measurement_sigma**2))
            landmarks.append({'class_name':label,'cluster_id':int(cluster),'world_x':float(center[0]),'world_z':float(center[1]),'independent_viewpoints':len(gg),'mean_confidence':float(gg.confidence.mean()),'spread_radius':float(np.sqrt(np.linalg.eigvalsh(cov).max())),'cov_xx':float(cov[0,0]),'cov_xz':float(cov[0,1]),'cov_zz':float(cov[1,1]),'status':'position_supported' if len(gg)>=5 and np.sqrt(np.linalg.eigvalsh(cov).max())<.35 else 'tentative'})
    lm=pd.DataFrame(landmarks,columns=['class_name','cluster_id','world_x','world_z','independent_viewpoints','mean_confidence','spread_radius','cov_xx','cov_xz','cov_zz','status']);lm.to_csv(OUT/'semantic_landmarks.csv',index=False)
    fig,ax=plt.subplots(figsize=(8,7));ax.imshow(map_image,origin='lower',extent=extent);ax.plot(t[:,0],t[:,2],color='#cb5947',lw=1)
    cmap=plt.get_cmap('tab10')
    for k,b in lm.iterrows():
        cov=np.array([[b.cov_xx,b.cov_xz],[b.cov_xz,b.cov_zz]]);ev,evec=np.linalg.eigh(cov);angle=np.degrees(np.arctan2(evec[1,1],evec[0,1]));color=cmap(k%10)
        ax.add_patch(Ellipse((b.world_x,b.world_z),2*np.sqrt(ev[1]),2*np.sqrt(ev[0]),angle=angle,facecolor=color,edgecolor=color,alpha=.2))
        ax.scatter(b.world_x,b.world_z,c=[color],s=45)
        ax.annotate(f'{b.class_name}: {b.independent_viewpoints} views\n{b.status}',(b.world_x,b.world_z),xytext=(8,8),textcoords='offset points',fontsize=8,bbox={'facecolor':'white','alpha':.85,'edgecolor':'none'})
    ax.set(title='View-balanced semantic landmarks with measured spatial spread',xlabel='World X (pose units)',ylabel='World Z (pose units)');ax.set_aspect('equal');fig.tight_layout();fig.savefig(OUT/'semantic_evidence_map.png',dpi=180);plt.close(fig)
    dump('semantic_metrics.json',{'detections':len(boxes),'static_observations':len(df),'landmarks':len(lm),'supported_landmarks':int((lm.status=='position_supported').sum()),'semantics':'YOLO proposals plus CLIP vocabulary verification' if verified.exists() else 'YOLO closed-vocabulary detections; not VLMaps/open-vocabulary features','class_caveat':'Labels remain model hypotheses; position support is separate from semantic correctness. Frame count is not independent evidence.','uncertainty':'ellipse combines spatial scatter and local calibration residual spread; not a calibrated confidence probability'})

def comparison(points,colors,poses):
    old=o3d.io.read_point_cloud(str(ROOT/'outputs/final_vlmaps_outputs/qualitative_room_vlmaps_style.ply'));p=np.asarray(old.points);c=np.asarray(old.colors)
    fig,axes=plt.subplots(1,2,figsize=(13,6))
    for ax,pts,col,title in [(axes[0],p,c,'Original: pseudo-depth and incorrect axes'),(axes[1],points,colors,'Repaired: calibrated and cross-view supported')]:
        take=np.arange(len(pts))
        if len(take)>60000:take=RNG.choice(take,60000,replace=False)
        ax.scatter(pts[take,0],pts[take,2],c=col[take],s=.55,alpha=.65);ax.plot(poses[:,0],poses[:,2],color='#b73728',lw=1)
        ax.set(title=title,xlabel='World X (pose units)',ylabel='World Z (pose units)');ax.set_aspect('equal');ax.grid(alpha=.15)
    lo=np.minimum(p[:,[0,2]].min(axis=0),points[:,[0,2]].min(axis=0))-.2;hi=np.maximum(p[:,[0,2]].max(axis=0),points[:,[0,2]].max(axis=0))+.2
    for ax in axes:ax.set_xlim(lo[0],hi[0]);ax.set_ylim(lo[1],hi[1])
    fig.tight_layout();fig.savefig(OUT/'before_after_geometry.png',dpi=180);plt.close(fig)
    # A fixed, identical-height slice avoids presenting density alone as map quality.
    fig,axes=plt.subplots(1,2,figsize=(13,6))
    for ax,pts,col,title in [(axes[0],p,c,'Original height slice'),(axes[1],points,colors,'Repaired height slice')]:
        keep=abs(pts[:,1]-np.median(poses[:,1]))<.35
        ax.scatter(pts[keep,0],pts[keep,2],c=col[keep],s=1,alpha=.7);ax.plot(poses[:,0],poses[:,2],color='#b73728',lw=1);ax.set(title=title,xlabel='X (pose units)',ylabel='Z (pose units)',xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]));ax.set_aspect('equal');ax.grid(alpha=.15)
    fig.tight_layout();fig.savefig(OUT/'before_after_height_slice.png',dpi=180);plt.close(fig)

def render_replay(poses):
    """Explicitly a replay over the FINAL map, not an online-building claim."""
    import imageio_ffmpeg
    q=np.load(OUT/'occupancy_evidence.npz');state=q['state'];extent=q['extent']
    rgb=np.zeros((*state.shape,3),np.uint8);rgb[:]=[209,215,220];rgb[state==0]=[250,248,240];rgb[state==1]=[49,69,83];rgb[state==2]=[192,162,97]
    mapim=np.flipud(rgb).copy();H,W=mapim.shape[:2];scale=min(650/W,550/H)
    mapim=cv2.resize(mapim,(int(W*scale),int(H*scale)),interpolation=cv2.INTER_NEAREST)
    frames=pd.read_csv(OUT/'frame_quality.csv').frame.to_numpy(int)
    boxes=pd.read_csv(OUT/'vlm_verified_boxes.csv') if (OUT/'vlm_verified_boxes.csv').exists() else pd.DataFrame()
    writer=imageio_ffmpeg.write_frames(str(OUT/'repaired_room_replay.mp4'),(1280,720),fps=10,codec='libx264',output_params=['-crf','22']);writer.send(None)
    try:
        for i in frames:
            canvas=np.full((720,1280,3),248,np.uint8);im=cv2.imread(str(DATA/'rgb'/f'{i:06d}.png'))[:,:,::-1].copy()
            if not boxes.empty:
                for _,b in boxes[boxes.frame==i].iterrows():
                    accepted=b.vlm_status=='accepted';color=(43,139,118) if accepted else (192,130,40)
                    cv2.rectangle(im,(int(b.x0),int(b.y0)),(int(b.x1),int(b.y1)),color,2)
                    cv2.putText(im,b.vlm_label if accepted else 'unresolved',(int(b.x0),max(20,int(b.y0)-8)),cv2.FONT_HERSHEY_SIMPLEX,.7,color,2)
            canvas[95:655,35:455]=cv2.resize(im,(420,560),interpolation=cv2.INTER_AREA)
            x0,y0=580,100;h,w=mapim.shape[:2];canvas[y0:y0+h,x0:x0+w]=mapim
            xy=poses[:i+1][:,[0,2]].copy();xy[:,0]=(xy[:,0]-extent[0])/(extent[1]-extent[0])*w+x0;xy[:,1]=y0+h-(xy[:,1]-extent[2])/(extent[3]-extent[2])*h
            xy=np.rint(xy).astype(np.int32);cv2.polylines(canvas,[xy],False,(200,77,59),2);cv2.circle(canvas,tuple(xy[-1]),6,(200,77,59),-1)
            cv2.putText(canvas,f'RGB frame {i} | text-verified proposals',(35,45),cv2.FONT_HERSHEY_SIMPLEX,.7,(40,50,60),2)
            cv2.putText(canvas,'Final evidence map + trajectory replay',(580,45),cv2.FONT_HERSHEY_SIMPLEX,.7,(40,50,60),2)
            cv2.putText(canvas,'Gray: unknown  Ivory: free  Dark: occupied  Gold: ambiguous',(490,690),cv2.FONT_HERSHEY_SIMPLEX,.55,(40,50,60),1)
            writer.send(canvas)
            if i==frames[len(frames)//2]:cv2.imwrite(str(OUT/'replay_preview.png'),canvas[:,:,::-1])
    finally:writer.close()
    video=cv2.VideoCapture(str(OUT/'repaired_room_replay.mp4'));ok,_=video.read();video.release()
    if not ok:raise RuntimeError('Replay MP4 failed decode verification.')

def main():
    start=time.perf_counter();model,K,mode,s,metrics=calibrate();poses=np.loadtxt(DATA/'poses.txt')
    p,c,selected,Kdepth,R,t,frames,rays=mapping(model,K,mode,poses)
    im,extent=ray_map(p,selected,t,rays)
    semantic_landmarks(model,Kdepth,R,t,selected,frames,im,extent)
    comparison(p,c,poses)
    render_replay(poses)
    dump('run_manifest.json',{'dataset':str(DATA),'frames':len(poses),'seed':20261008,'selected_camera_mode':mode,'inferred_intrinsics_half_resolution':K.tolist(),'depth_source':'User reports iPhone LiDAR; exported 0..255 depth encoding is undocumented. Calibration recovered from RGB correspondences and provided poses.','units':'pose translation units; expected metres for unscaled ARKit export, not independently verified','total_runtime_seconds':time.perf_counter()-start,'originals_preserved':True,'novelty':'Research hypothesis only: provenance recovery coupled to independent-view semantic evidence; known ingredients are not claimed as novel.','versions':{'numpy':np.__version__,'opencv':cv2.__version__,'open3d':o3d.__version__}})
    print('DONE',flush=True)

if __name__=='__main__':main()
