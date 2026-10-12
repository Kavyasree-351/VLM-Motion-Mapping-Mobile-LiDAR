"""Reproducible camera/depth diagnostic; no ground truth accuracy claims."""
from pathlib import Path
import sys, json, time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.packages'))
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares

DATA = ROOT / 'dataset/content/vlmaps_dataset/my_custom_scene'
OUT = ROOT / 'research'
cv2.setNumThreads(4)

def cameras(poses, mode):
    R = Rotation.from_quat(poses[:, 3:]).as_matrix()
    t = poses[:, :3].copy()
    if mode.startswith('w2c'):
        R = R.transpose(0, 2, 1)
        t = -np.einsum('nij,nj->ni', R, t)
    if mode.endswith('opengl'):
        R = R @ np.diag([1., -1., -1.])
    return R, t

def intrinsic(focal=375.):
    return np.array([[focal, 0, 180.], [0, focal, 239.46], [0, 0, 1.]])

def projection(R, t, K):
    return K @ np.column_stack((R.T, -R.T @ t))

def skew(t):
    x,y,z=t
    return np.array([[0,-z,y],[z,0,-x],[-y,x,0]])

def epipolar(uv1, uv2, R1,t1,R2,t2,K):
    R=R2.T@R1; t=R2.T@(t1-t2)
    ik=np.linalg.inv(K); F=ik.T@skew(t)@R@ik
    a=np.column_stack([uv1,np.ones(len(uv1))]); b=np.column_stack([uv2,np.ones(len(uv2))])
    Fa=a@F.T; Fb=b@F
    return np.abs(np.sum(b*Fa,axis=1))/np.sqrt(Fa[:,0]**2+Fa[:,1]**2+Fb[:,0]**2+Fb[:,1]**2+1e-16)

def triangulate(uv1,uv2,R1,t1,R2,t2,K):
    h=cv2.triangulatePoints(projection(R1,t1,K),projection(R2,t2,K),uv1.T,uv2.T)
    xyz=(h[:3]/h[3:]).T
    a=(xyz-t1)@R1; b=(xyz-t2)@R2
    pa=a@K.T;pb=b@K.T
    err=(np.linalg.norm(pa[:,:2]/pa[:,2:]-uv1,axis=1)+np.linalg.norm(pb[:,:2]/pb[:,2:]-uv2,axis=1))/2
    rays1=xyz-t1;rays2=xyz-t2
    cos=np.sum(rays1*rays2,axis=1)/(np.linalg.norm(rays1,axis=1)*np.linalg.norm(rays2,axis=1)+1e-12)
    angle=np.degrees(np.arccos(np.clip(cos,-1,1)))
    return xyz,a[:,2],b[:,2],err,angle

def build_matches():
    cache=OUT/'feature_pairs.npz'
    if cache.exists():
        z=np.load(cache); return z['pairs'],z['uv1'],z['uv2']
    sift=cv2.SIFT_create(nfeatures=1600,contrastThreshold=.025)
    matcher=cv2.BFMatcher()
    features={}
    def feature(i):
        if i not in features:
            im=cv2.imread(str(DATA/'rgb'/f'{i:06d}.png'),0)
            im=cv2.resize(im,(360,480),interpolation=cv2.INTER_AREA)
            kp,d=sift.detectAndCompute(im,None)
            features[i]=(np.array([k.pt for k in kp]),d)
        return features[i]
    pairs=[];u1=[];u2=[]
    for i in range(0,2091-80,40):
        for gap in [20,60,100]:
            j=i+gap
            if j>=2091:continue
            p1,d1=feature(i);p2,d2=feature(j)
            if d1 is None or d2 is None:continue
            matches=matcher.knnMatch(d1,d2,k=2)
            good=[m for m,n in matches if m.distance<.72*n.distance]
            if len(good)<15:continue
            a=np.array([p1[m.queryIdx] for m in good]);b=np.array([p2[m.trainIdx] for m in good])
            _,mask=cv2.findFundamentalMat(a,b,cv2.USAC_MAGSAC,1.5,.999)
            if mask is None:continue
            keep=mask.ravel().astype(bool)
            if keep.sum()<12:continue
            pairs.extend([(i,j)]*int(keep.sum()));u1.extend(a[keep]);u2.extend(b[keep])
        if i%400==0:print('Matched frame',i,flush=True)
    arrays=(np.array(pairs),np.array(u1),np.array(u2))
    np.savez_compressed(cache,pairs=arrays[0],uv1=arrays[1],uv2=arrays[2])
    return arrays

def run():
    poses=np.loadtxt(DATA/'poses.txt');pairs,a,b=build_matches()
    modes=['c2w_opencv','c2w_opengl','w2c_opencv','w2c_opengl']
    rows=[]
    for mode in modes:
        R,t=cameras(poses,mode)
        for f in [250.,300.,350.,375.,400.,450.,500.]:
            errors=[];pos=[]
            for i,j in np.unique(pairs,axis=0):
                mask=(pairs[:,0]==i)&(pairs[:,1]==j)&((pairs[:,0]//40)%5!=0)
                if not mask.any():continue
                if np.linalg.norm(t[i]-t[j])<.025:continue
                aa,bb=a[mask],b[mask];K=intrinsic(f)
                errors.extend(epipolar(aa,bb,R[i],t[i],R[j],t[j],K))
                _,z1,z2,_,_=triangulate(aa,bb,R[i],t[i],R[j],t[j],K)
                pos.extend((z1>0)&(z2>0))
            rows.append(dict(mode=mode,focal_half_resolution=f,median_epipolar_px=float(np.median(errors)),positive_depth_fraction=float(np.mean(pos)),samples=len(errors),split='training anchors only'))
    rows.sort(key=lambda r:r['median_epipolar_px']+3*(1-r['positive_depth_fraction']))
    print('Camera candidates:',json.dumps(rows[:6],indent=2),flush=True)
    best=rows[0]; mode=best['mode'];R,t=cameras(poses,mode)
    # Refine focal on training frame pairs only; reserve every fifth anchor for evaluation.
    train=(pairs[:,0]//40)%5!=0
    def residual(x):
        K=intrinsic(x[0]);res=[]
        for i,j in np.unique(pairs[train],axis=0):
            mask=train&(pairs[:,0]==i)&(pairs[:,1]==j)
            if np.linalg.norm(t[i]-t[j])<.025:continue
            res.extend(epipolar(a[mask],b[mask],R[i],t[i],R[j],t[j],K))
        return np.array(res)
    fit=least_squares(residual,[best['focal_half_resolution']],bounds=([180.],[700.]),loss='soft_l1',f_scale=1.)
    K=intrinsic(fit.x[0]);samples=[];depth_cache={}
    for i,j in np.unique(pairs,axis=0):
        mask=(pairs[:,0]==i)&(pairs[:,1]==j)
        if np.linalg.norm(t[i]-t[j])<.035:continue
        aa,bb=a[mask],b[mask]
        xyz,z1,z2,err,angle=triangulate(aa,bb,R[i],t[i],R[j],t[j],K)
        good=(z1>.15)&(z2>.15)&(z1<12)&(z2<12)&(err<2)&(angle>1.2)
        for frame,uv,z in [(i,aa,z1),(j,bb,z2)]:
            if frame not in depth_cache:depth_cache[frame]=np.load(DATA/'depth'/f'{frame:06d}.npy')
            d=depth_cache[frame]
            u=np.clip(uv[:,0]*192/360,0,191).astype(int);v=np.clip(uv[:,1]*256/480,0,255).astype(int)
            raw=d[v,u]
            for k in np.flatnonzero(good&(raw>3)&(raw<252)):
                samples.append((frame,int(j if frame==i else i),float(uv[k,0]),float(uv[k,1]),float(raw[k]),float(z[k]),float(err[k]),float(angle[k]),*xyz[k]))
    samples=np.array(samples)
    np.savez_compressed(OUT/'triangulated_samples.npz',samples=samples,K=K,mode=mode)
    summary={'camera_candidates':rows,'selected_mode':mode,'refined_K_half_resolution':K.tolist(),'triangulated_samples':len(samples),'pose_units':'unknown; geometric depths share pose translation units','selection_protocol':'camera convention and focal selected using training anchors; every fifth 40-frame anchor reserved'}
    (OUT/'geometry_diagnostics.json').write_text(json.dumps(summary,indent=2))
    print('Camera selected',mode,'focal',fit.x[0],'triangulated samples',len(samples),flush=True)

if __name__=='__main__':run()
