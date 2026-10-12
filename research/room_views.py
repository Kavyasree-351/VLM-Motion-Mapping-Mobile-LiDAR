"""Measured room layout, mask-fused objects, textured mesh and interactive 3D views."""
from pathlib import Path
import sys,os,json,time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.packages'))
os.environ['HF_HUB_OFFLINE']='1'
os.environ['MPLCONFIGDIR']=str(ROOT/'.matplotlib')
import numpy as np,pandas as pd,cv2,torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull,cKDTree
from sklearn.cluster import DBSCAN
import open3d as o3d
import plotly.graph_objects as go
import room_upgrade as upgrade
import improve_mapping as mapping
from geometry_probe import DATA

OUT=upgrade.OUT
RNG=np.random.default_rng(260108)
STRUCTURAL={'door','window','picture','air vent','ceiling light'}
WALL_ATTACHED={'door','window','picture'}

def semantic_review(observations):
    cache=OUT/'detections_reviewed.csv'
    if cache.exists():return pd.read_csv(cache)
    from transformers import CLIPModel,CLIPProcessor
    from verify_semantics import features
    from PIL import Image
    torch.set_num_threads(4)
    clip=CLIPModel.from_pretrained(ROOT/'.models/clip',local_files_only=True).eval()
    processor=CLIPProcessor.from_pretrained(ROOT/'.models/clip',local_files_only=True)
    names=['sofa','armchair','beanbag','storage table','shelf','door','window','picture','air vent','ceiling light','person','background']
    prompts=['a black leather sofa','a single upholstered armchair','a bean bag chair','a wooden side table with shelves','a bookcase or shelving unit','a wooden door','a glass window with a frame','a framed picture on a wall','a ceiling air vent','a ceiling light fixture','a person','an empty wall or floor']
    with torch.inference_mode():
        t=features(clip.get_text_features(**processor(text=[f'a photo of {p}.' for p in prompts],return_tensors='pt',padding=True)));t=t/t.norm(dim=-1,keepdim=True)
        scores=[]
        for start in range(0,len(observations),16):
            crops=[]
            for _,b in observations.iloc[start:start+16].iterrows():
                im=Image.open(DATA/'rgb'/f'{int(b.frame):06d}.png').convert('RGB');crops.append(im.crop((b.x0,b.y0,b.x1,b.y1)))
            f=features(clip.get_image_features(**processor(images=crops,return_tensors='pt')));f=f/f.norm(dim=-1,keepdim=True)
            scores.append((f@t.T).cpu().numpy())
    scores=np.vstack(scores);best=np.argmax(scores,axis=1)
    reviewed=observations.copy();reviewed['clip_label']=[names[k] for k in best];reviewed['clip_cosine']=scores[np.arange(len(best)),best]
    chosen=[];status=[]
    aliases={'beanbag chair':'beanbag','chair':'armchair','table':'storage table'}
    for k,b in observations.iterrows():
        label=aliases.get(b.class_name,b.class_name);target=names.index(label) if label in names else -1
        winner=names[best[k]];gap=float(scores[k,best[k]]-scores[k,target]) if target>=0 else 1.
        # Furniture ambiguity is resolved only with a distinct alternate score.
        if winner in ['armchair','sofa'] and label in ['armchair','sofa'] and gap>.025:label=winner
        disputed=(winner=='background' and gap>.025) or scores[k,best[k]]<.21
        chosen.append(label);status.append('disputed' if disputed else 'reviewed_candidate')
    reviewed['semantic_label']=chosen;reviewed['semantic_status']=status
    reviewed.to_csv(cache,index=False);np.savez_compressed(OUT/'semantic_review_scores.npz',scores=scores,labels=np.array(names))
    return reviewed

def fit_wall_lines(points):
    """Measured 2D line segments, not a forced rectangle or invented closed wall."""
    work=points.copy();lines=[]
    if len(work)>12000:work=work[RNG.choice(len(work),12000,replace=False)]
    for _ in range(7):
        best=None;best_count=0
        for _ in range(200):
            pair=work[RNG.choice(len(work),2,replace=False)];direction=pair[1]-pair[0];length=np.linalg.norm(direction)
            if length<.6:continue
            direction/=length;normal=np.array([-direction[1],direction[0]])
            distances=abs((work-pair[0])@normal);mask=distances<.045
            if mask.sum()>best_count:best=mask;best_count=int(mask.sum())
        if best is None or best_count<80:break
        fitted=work[best];center=fitted.mean(axis=0);_,_,Vt=np.linalg.svd(fitted-center,full_matrices=False);d=Vt[0]
        projected=(fitted-center)@d;ends=np.percentile(projected,[1,99]);endpoints=center+ends[:,None]*d
        if np.linalg.norm(endpoints[1]-endpoints[0])>.65:
            lines.append({'endpoints':endpoints.tolist(),'support_points':best_count,'direction':d.tolist(),'rms':float(np.sqrt(np.mean(((fitted-center)@np.array([-d[1],d[0]]))**2)))})
        work=work[~best]
        if len(work)<100:break
    consolidated=[]
    for line in sorted(lines,key=lambda l:l['support_points'],reverse=True):
        a=np.array(line['endpoints']);da=a[1]-a[0];da/=np.linalg.norm(da);duplicate=False
        for existing in consolidated:
            b=np.array(existing['endpoints']);db=b[1]-b[0];db/=np.linalg.norm(db);normal=np.array([-db[1],db[0]])
            angle=np.arccos(np.clip(abs(da@db),0,1));separation=abs((a.mean(axis=0)-b.mean(axis=0))@normal)
            aa=a@db;bb=b@db;overlap=max(0,min(aa.max(),bb.max())-max(aa.min(),bb.min()))
            if angle<np.deg2rad(12) and separation<.20 and overlap>.35*min(np.ptp(aa),np.ptp(bb)):
                duplicate=True;break
        if not duplicate:consolidated.append(line)
    return consolidated

def room_frame(points,poses):
    band=abs(points[:,1]-np.median(poses[:,1]))<.28
    lines=fit_wall_lines(points[band][:,[0,2]])
    if len(lines)<2:raise RuntimeError('Insufficient wall segments to align a room layout.')
    angles=np.array([np.arctan2(l['direction'][1],l['direction'][0]) for l in lines]);weights=np.array([l['support_points'] for l in lines])
    angle=float(np.arctan2(np.sum(weights*np.sin(4*angles)),np.sum(weights*np.cos(4*angles)))/4)
    A=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    wallplan=points[band][:,[0,2]]@A
    bounds=np.percentile(wallplan,[.5,99.5],axis=0)
    low_points=points[points[:,1]<-.6]
    floor_cloud=o3d.geometry.PointCloud();floor_cloud.points=o3d.utility.Vector3dVector(low_points)
    plane,inliers=floor_cloud.segment_plane(distance_threshold=.025,ransac_n=3,num_iterations=800)
    if abs(plane[1])<.9:raise RuntimeError('Dominant low surface is not a gravity-aligned floor.')
    floor=float(np.median(low_points[inliers,1]));ceiling=float(np.percentile(points[:,1],98))
    transform={'rotation_radians':angle,'plan_rotation_matrix':A.tolist(),'floor_world_y':floor,'ceiling_world_y':ceiling,'height_pose_units':ceiling-floor,'plan_bounds':bounds.tolist(),'wall_segments_world':lines,'units':'ARKit pose-scale units, nominal metres; exporter metric scale not independently verified','room_outline':'Observed line segments; dashed convex envelope is inferred, not a verified closed navigation boundary.'}
    upgrade.dump('room_geometry.json',transform)
    return A,floor,ceiling,bounds,lines

def fuse_objects(observations,arrays,A,floor,cloud_points):
    observations=semantic_review(observations)
    keep=(observations.semantic_status!='disputed')&(~observations.semantic_label.isin(['person']))&(observations.confidence>=.22)
    tree=cKDTree(cloud_points);objects=[];allp=[];allc=[];alllabels=[]
    for label,g in observations[keep].groupby('semantic_label'):
        centers=g[['world_x','world_y','world_z']].to_numpy();eps=.6 if label in STRUCTURAL else .5
        groups=DBSCAN(eps=eps,min_samples=3).fit_predict(centers)
        for cluster in sorted(set(groups)-{-1}):
            subset=g.iloc[np.flatnonzero(groups==cluster)].sort_values('confidence',ascending=False).drop_duplicates('frame')
            if len(subset)<3:continue
            ids=subset.observation_id.to_numpy(int);where=np.isin(arrays['observation_ids'],ids)
            pts=arrays['points'][where];colors=arrays['colors'][where];obsids=arrays['observation_ids'][where]
            distance,_=tree.query(pts,k=1,workers=4)
            valid=(distance<.14)&np.isfinite(pts).all(axis=1)
            if label not in STRUCTURAL:valid &= pts[:,1]>floor+.065
            pts=pts[valid];colors=colors[valid];obsids=obsids[valid]
            if len(pts)<80:continue
            index=subset.set_index('observation_id');frames=index.loc[obsids,'frame'].to_numpy(int)
            weights=index.loc[obsids,'confidence'].to_numpy()*np.exp(-index.loc[obsids,'calibration_spread'].to_numpy()/.25)
            fused,col,views,sigma=mapping.voxel_fuse(pts,colors,frames,weights,voxel=.045)
            supported=(views>=2)&(sigma<.065)
            fused=fused[supported];col=col[supported]
            if len(fused)<40:continue
            plan=fused[:,[0,2]]@A
            # Remove sparse peripheral contamination without filling missing faces.
            lo,hi=np.percentile(plan,[2,98],axis=0);central=np.all((plan>=lo)&(plan<=hi),axis=1)
            fitted=plan[central];center=np.median(fitted,axis=0)
            if label in WALL_ATTACHED:
                _,_,Vt=np.linalg.svd(fitted-center,full_matrices=False);d=Vt[0];s=(fitted-center)@d;ends=np.percentile(s,[2,98]);n=np.array([-d[1],d[0]])
                polygon=np.array([center+ends[0]*d-.025*n,center+ends[1]*d-.025*n,center+ends[1]*d+.025*n,center+ends[0]*d+.025*n])
            else:
                hull=ConvexHull(fitted);polygon=fitted[hull.vertices]
            camera_positions=np.loadtxt(DATA/'poses.txt')[subset.frame.to_numpy(int),:3]
            independent=[]
            for k,p in enumerate(camera_positions):
                if not independent or np.linalg.norm(camera_positions[independent]-p,axis=1).min()>=.08:independent.append(k)
            record={'id':f'{label.replace(" ","_")}_{cluster}','class_name':label,'plan_center':center.tolist(),'observed_footprint':polygon.tolist(),'frame_observations':len(subset),'independent_camera_positions':len(independent),'supported_voxels':len(fused),'mean_detector_confidence':float(subset.confidence.mean()),'plan_spread':np.std(subset[['world_x','world_z']].to_numpy()@A,axis=0).tolist(),'height_range_above_floor':np.percentile(fused[:,1]-floor,[2,98]).tolist(),'color':upgrade.PALETTE.get(label,'#7b8790'),'status':'supported_surface' if len(independent)>=3 else 'limited_view_surface','shape_caveat':'Visible mask-fused surfaces, not a complete object shape or verified floor-contact footprint. Architectural glyph width is for display only.'}
            objects.append(record);allp.append(fused);allc.append(col);alllabels.append(np.full(len(fused),len(objects)-1))
    upgrade.dump('room_objects.json',objects)
    if not objects:raise RuntimeError('No multi-view-supported objects survived semantic and geometry review.')
    arrays_out={'points':np.vstack(allp),'colors':np.vstack(allc),'object_ids':np.concatenate(alllabels)}
    np.savez_compressed(OUT/'object_surfaces.npz',**arrays_out)
    cloud=o3d.geometry.PointCloud();cloud.points=o3d.utility.Vector3dVector(arrays_out['points']);cloud.colors=o3d.utility.Vector3dVector(arrays_out['colors']);o3d.io.write_point_cloud(str(OUT/'objects_rgb.ply'),cloud)
    print('Supported objects:',[(o['class_name'],o['supported_voxels']) for o in objects],flush=True)
    return objects,arrays_out

def draw_plan(points,colors,A,floor,poses,bounds,lines,objects,mask_description='YOLOE-26 masks'):
    plan=points[:,[0,2]]@A;lo,hi=bounds
    selected=points[(abs(points[:,1]-np.median(poses[:,1]))<.28)][:,[0,2]]@A
    hull=ConvexHull(selected);envelope=selected[hull.vertices]
    fig,ax=plt.subplots(figsize=(11,10));fig.patch.set_facecolor('#f6f4ee');ax.set_facecolor('#f6f4ee')
    ax.add_patch(Polygon(envelope,facecolor='#fffefa',edgecolor='#b6bab8',lw=1,linestyle=(0,(4,4)),zorder=0))
    # Show only observed wall samples and robust segments, not floor/ceiling density.
    ax.scatter(selected[:,0],selected[:,1],s=.35,c='#839195',alpha=.25,zorder=1)
    for line in lines:
        e=np.array(line['endpoints'])@A
        ax.plot(e[:,0],e[:,1],color='#3b4b52',lw=5,solid_capstyle='round',zorder=2)
    for k,obj in enumerate(objects):
        poly=np.array(obj['observed_footprint']);color=obj['color']
        overhead=obj['class_name'] in {'air vent','ceiling light'}
        ax.add_patch(Polygon(poly,facecolor='none' if overhead else color,edgecolor=color,lw=1.4,linestyle='--' if overhead else '-',alpha=.60 if obj['class_name'] not in STRUCTURAL else .85,zorder=4))
        center=np.array(obj['plan_center']);anchor=center.copy()
        # Label placement is geometric; labels get margins without altering shapes.
        if obj['class_name'] in STRUCTURAL:
            delta=center-(lo+hi)/2;delta/=max(np.linalg.norm(delta),1e-8);anchor=center+.32*delta
        else:anchor=center+np.array([.05,.18])
        text=obj['class_name'].title()
        if overhead:text+='\n(ceiling)'
        if obj['class_name'] not in STRUCTURAL:text+=f"\n{obj['independent_camera_positions']} positions"
        ax.annotate(text,xy=center,xytext=anchor,ha='center',va='center',fontsize=9,color='#24343a',bbox={'facecolor':'#fffefa','edgecolor':'none','alpha':.92,'pad':3},arrowprops={'arrowstyle':'-','color':color,'lw':.8},zorder=6)
    width,depth=hi-lo
    ydim=lo[1]-.50;xdim=hi[0]+.55
    ax.annotate('',(lo[0],ydim),(hi[0],ydim),arrowprops={'arrowstyle':'|-|','color':'#69777c','lw':1})
    ax.text((lo[0]+hi[0])/2,ydim-.08,f'{width:.2f} pose-scale units',ha='center',va='top',color='#69777c',fontsize=10)
    ax.annotate('',(xdim,lo[1]),(xdim,hi[1]),arrowprops={'arrowstyle':'|-|','color':'#69777c','lw':1})
    ax.text(xdim+.10,(lo[1]+hi[1])/2,f'{depth:.2f} units',rotation=90,ha='left',va='center',color='#69777c',fontsize=10)
    ax.set(xlim=(lo[0]-.7,hi[0]+.9),ylim=(lo[1]-.8,hi[1]+.65));ax.set_aspect('equal');ax.axis('off')
    fig.suptitle('Room layout from measured surfaces',x=.12,ha='left',fontsize=21,color='#20343c',fontweight='bold')
    fig.text(.12,.915,f'{mask_description} · geometric consensus · room-aligned top view',fontsize=11,color='#6c777a')
    fig.text(.12,.06,'Solid walls: observed segments    Dashed outline: inferred envelope\nFurniture: observed surface footprints; incomplete backs may be missing. Scale follows supplied ARKit poses.',fontsize=9,color='#6c777a')
    fig.savefig(OUT/'room_floorplan.png',dpi=210,bbox_inches='tight');fig.savefig(OUT/'room_floorplan.svg',bbox_inches='tight');plt.close(fig)
    # A separate evidence image prevents the schematic from replacing raw geometry.
    fig,ax=plt.subplots(figsize=(10,8));take=np.arange(len(plan));ax.scatter(plan[take,0],plan[take,1],c=colors[take],s=.4,alpha=.7)
    for obj in objects:
        p=np.array(obj['observed_footprint']);ax.plot(np.r_[p[:,0],p[0,0]],np.r_[p[:,1],p[0,1]],color=obj['color'],lw=2)
    ax.set(title='Room-aligned RGB surface evidence',xlabel='Aligned X (pose-scale units)',ylabel='Aligned Z (pose-scale units)');ax.set_aspect('equal');fig.tight_layout();fig.savefig(OUT/'room_topview_rgb.png',dpi=180);plt.close(fig)

def reconstruct_mesh(model,K,poses,R,t,selected,cloud_points):
    path=OUT/'room_textured_mesh.ply'
    if path.exists():
        cached=o3d.io.read_triangle_mesh(str(path))
        if len(cached.vertices)>0:return cached
    # Open3D 0.20 legacy ScalableTSDFVolume has a verified depth-scale activation
    # regression (#7592). Tensor VoxelBlockGrid avoids that path. Its optimized
    # CPU kernel requires matching depth/color input dtypes (uint16/uint8 here).
    volume=create_tensor_volume(.025)
    tree=cKDTree(cloud_points)
    intrinsic=o3d.core.Tensor(K,o3d.core.float64)
    skipped=[]
    for count,frame in enumerate(selected):
        raw,rgb,depth,valid=mapping.load_frame(int(frame),model)
        world=mapping.backproject(depth,K)@R[frame].T+t[frame]
        # Dense pixels must agree with the independently supported cloud.
        dist,_=tree.query(world.reshape(-1,3),workers=4);valid &= dist.reshape(depth.shape)<.11
        depth=np.rint(np.where(valid,depth,0)*1000).astype(np.uint16)
        if np.count_nonzero(depth[::4,::4])<4:
            skipped.append(int(frame));continue
        ext=np.eye(4);ext[:3,:3]=R[frame].T;ext[:3,3]=-R[frame].T@t[frame]
        integrate_tensor_frame(volume,depth,rgb,intrinsic,ext)
        if count%20==0:print('Textured TSDF frame',count,'/',len(selected),flush=True)
    mesh=volume.extract_triangle_mesh(weight_threshold=2.).to_legacy();mesh.remove_degenerate_triangles();mesh.remove_duplicated_vertices();mesh.remove_duplicated_triangles();mesh.compute_vertex_normals()
    if len(mesh.vertices)==0 or len(mesh.triangles)==0:raise RuntimeError('TSDF produced an empty mesh; do not publish a misleading 3D render.')
    if not o3d.io.write_triangle_mesh(str(path),mesh):raise IOError('Mesh export failed.')
    upgrade.dump('tsdf_backend.json',{'backend':'Open3D tensor VoxelBlockGrid','open3d_version':o3d.__version__,'input_depth':'uint16 with scale 1000, quantization at 0.001 pose unit','color':'uint8 RGB','voxel_size':.025,'trunc_voxel_multiplier':4.,'mesh_minimum_weight':2.,'frames_skipped_for_no_supported_depth':skipped,'legacy_regression_reference':'https://github.com/isl-org/Open3D/issues/7592','legacy_reproduction':'Valid 2-unit synthetic plane yields 49152 back-projected points and zero legacy TSDF vertices; tensor backend yielded 3763 vertices in the probe.'})
    print('Mesh vertices',len(mesh.vertices),'triangles',len(mesh.triangles),flush=True)
    return mesh

def create_tensor_volume(voxel_size=.025):
    return o3d.t.geometry.VoxelBlockGrid(attr_names=('tsdf','weight','color'),attr_dtypes=(o3d.core.float32,o3d.core.float32,o3d.core.float32),attr_channels=(1,1,3),voxel_size=voxel_size,block_resolution=16,block_count=2000,device=o3d.core.Device('CPU:0'))

def integrate_tensor_frame(volume,depth_u16,rgb_u8,intrinsic,extrinsic):
    if np.count_nonzero(depth_u16[::4,::4])<4:return False
    depth=o3d.t.geometry.Image(o3d.core.Tensor(np.ascontiguousarray(depth_u16)))
    color=o3d.t.geometry.Image(o3d.core.Tensor(np.ascontiguousarray(rgb_u8)))
    ext=o3d.core.Tensor(extrinsic,o3d.core.float64)
    blocks=volume.compute_unique_block_coordinates(depth,intrinsic,ext,1000.,8.,4.)
    volume.integrate(blocks,depth,color,intrinsic,ext,1000.,8.,4.)
    return True

def aligned_coordinates(points,A,floor):
    plan=points[:,[0,2]]@A
    return np.column_stack([plan,points[:,1]-floor])

def mesh_arrays(mesh,A,floor,bounds,ceiling):
    xyz=aligned_coordinates(np.asarray(mesh.vertices),A,floor);faces=np.asarray(mesh.triangles);colors=np.asarray(mesh.vertex_colors)
    lo,hi=bounds;centers=xyz[faces].mean(axis=1)
    nonceiling=centers[:,2]<ceiling-floor-.10
    vectors=xyz[faces[:,1]]-xyz[faces[:,0]];vectors2=xyz[faces[:,2]]-xyz[faces[:,0]]
    normals=np.cross(vectors,vectors2);normals/=np.maximum(np.linalg.norm(normals,axis=1)[:,None],1e-9)
    near_walls=np.zeros(len(faces),bool)
    geometry=json.loads((OUT/'room_geometry.json').read_text())
    room_center=(lo+hi)/2
    for line in geometry['wall_segments_world']:
        ends=np.array(line['endpoints'])@A;direction=ends[1]-ends[0];length=np.linalg.norm(direction);direction/=length
        # Remove the right and front observed wall planes, including the notch.
        right=abs(direction[1])>.8 and ends[:,0].mean()>room_center[0]
        front=abs(direction[0])>.8 and ends[:,1].mean()<room_center[1]
        if not (right or front):continue
        normal=np.array([-direction[1],direction[0]])
        distance=abs((centers[:,:2]-ends[0])@normal);along=(centers[:,:2]-ends[0])@direction
        vertical=abs(normals[:,2])<.45
        near_walls |= (distance<.22)&(along>-.22)&(along<length+.22)&vertical&(centers[:,2]>.12)
    cut=nonceiling&(~near_walls)
    return xyz,faces,colors,cut,nonceiling

def render_3d(mesh,points,colors,objects,object_arrays,A,floor,bounds,ceiling):
    xyz,faces,vertexcolors,cut,nonceiling=mesh_arrays(mesh,A,floor,bounds,ceiling)
    facecolors=vertexcolors[faces].mean(axis=1)
    # Export actual cutaway mesh, preserving original world-space full mesh too.
    cutmesh=o3d.geometry.TriangleMesh();cutmesh.vertices=o3d.utility.Vector3dVector(xyz);cutmesh.triangles=o3d.utility.Vector3iVector(faces[cut]);cutmesh.vertex_colors=o3d.utility.Vector3dVector(vertexcolors);cutmesh.remove_unreferenced_vertices();cutmesh.compute_vertex_normals();o3d.io.write_triangle_mesh(str(OUT/'room_cutaway_aligned.ply'),cutmesh)
    selected_faces=np.flatnonzero(cut)
    # Matplotlib draws all remaining triangles in a faithful CPU render.
    fig=plt.figure(figsize=(15,7),facecolor='#f6f4ee')
    for index,azimuth in enumerate([-58,135]):
        ax=fig.add_subplot(1,2,index+1,projection='3d');ax.set_facecolor('#f6f4ee')
        collection=Poly3DCollection(xyz[faces[selected_faces]],facecolors=facecolors[selected_faces],edgecolors='none',linewidths=0,rasterized=True)
        ax.add_collection3d(collection)
        op=aligned_coordinates(object_arrays['points'],A,floor)
        # The cutaway hides near architectural layers; furniture masks remain.
        furniture_ids=[k for k,o in enumerate(objects) if o['class_name'] not in STRUCTURAL]
        visible_objects=np.isin(object_arrays['object_ids'],furniture_ids)
        ax.scatter(op[visible_objects,0],op[visible_objects,1],op[visible_objects,2],c=object_arrays['colors'][visible_objects],s=.35,depthshade=False)
        for obj in objects:
            if obj['class_name'] in STRUCTURAL:continue
            center=obj['plan_center'];height=float(obj['height_range_above_floor'][1]);ax.text(center[0],center[1],height+.09,obj['class_name'].title(),fontsize=8,color='#253c43',zorder=1000,bbox={'facecolor':'white','alpha':.92,'edgecolor':'none','pad':1.5})
        lo,hi=bounds;ax.set(xlim=(lo[0]-.15,hi[0]+.15),ylim=(lo[1]-.15,hi[1]+.15),zlim=(0,ceiling-floor+.10))
        ax.set_box_aspect((hi[0]-lo[0],hi[1]-lo[1],ceiling-floor));ax.view_init(elev=29,azim=azimuth);ax.set_axis_off();ax.set_title('Textured cutaway' if index==0 else 'Opposite view',fontsize=14,color='#30464d',pad=2)
    fig.suptitle('Reconstructed room in 3D',fontsize=22,x=.08,ha='left',color='#233840',fontweight='bold')
    fig.text(.08,.025,'Measured RGB-D surfaces. Ceiling and two near walls are removed for visibility; no missing furniture surfaces are invented.',fontsize=10,color='#6d787c')
    fig.savefig(OUT/'room_3d_cutaway.png',dpi=180,bbox_inches='tight');plt.close(fig)
    # Interactive views retain full geometry and independent object colors.
    fig=go.Figure()
    rgba=['rgb(%d,%d,%d)'%tuple(np.clip(c*255,0,255).astype(int)) for c in vertexcolors]
    def meshtrace(label,mask,visible):
        tri=faces[mask]
        return go.Mesh3d(x=xyz[:,0],y=xyz[:,1],z=xyz[:,2],i=tri[:,0],j=tri[:,1],k=tri[:,2],vertexcolor=rgba,name=label,visible=visible,flatshading=False,lighting={'ambient':.8,'diffuse':.7,'specular':.1},hoverinfo='skip')
    fig.add_trace(meshtrace('Cutaway room',cut,True));fig.add_trace(meshtrace('Full textured room',np.ones(len(faces),bool),False))
    op=aligned_coordinates(object_arrays['points'],A,floor)
    default_visibility=[True,False]
    for k,obj in enumerate(objects):
        where=object_arrays['object_ids']==k;p=op[where]
        visible='legendonly' if obj['class_name'] in STRUCTURAL else True
        default_visibility.append(visible)
        fig.add_trace(go.Scatter3d(x=p[:,0],y=p[:,1],z=p[:,2],mode='markers',marker={'size':2,'color':obj['color']},name=obj['class_name'].title(),visible=visible,text=[f"{obj['class_name']} · {obj['independent_camera_positions']} camera positions"]*len(p),hovertemplate='%{text}<br>X=%{x:.2f}, Z=%{y:.2f}, height=%{z:.2f}<extra></extra>'))
    n=len(fig.data);objects_visible=[False,False]+[True]*(n-2)
    fig.update_layout(title={'text':'Room reconstruction · drag to rotate, scroll to zoom','x':.04},paper_bgcolor='#f6f4ee',margin={'l':0,'r':0,'t':85,'b':0},scene={'aspectmode':'data','xaxis':{'title':'Aligned X'},'yaxis':{'title':'Aligned Z'},'zaxis':{'title':'Height above inferred floor'},'camera':{'eye':{'x':1.5,'y':-1.8,'z':1.3}}},legend={'x':.02,'y':.98},updatemenus=[{'type':'buttons','direction':'right','x':.04,'y':1.06,'buttons':[{'label':'Cutaway + objects','method':'update','args':[{'visible':default_visibility}]},{'label':'Full room','method':'update','args':[{'visible':[False,True]+[False]*(n-2)}]},{'label':'Objects only','method':'update','args':[{'visible':objects_visible}]}]}])
    fig.write_html(str(OUT/'room_3d_interactive.html'),include_plotlyjs=True,full_html=True)
    upgrade.dump('mesh_metrics.json',{'vertices':len(xyz),'triangles':len(faces),'cutaway_triangles':int(cut.sum()),'tsdf_voxel_pose_units':.025,'tsdf_truncation_pose_units':.10,'dense_pixel_support_gate':.11,'object_surface_points':len(op),'render':'Actual fused vertex RGB; interactive full-room/cutaway/object layers. No generative replacement or synthetic geometry completion.'})

def main():
    start=time.perf_counter();observations,arrays,selected=upgrade.detect();model,K,poses,R,t=upgrade.load_geometry()
    cloud=o3d.io.read_point_cloud(str(mapping.OUT/'calibrated_supported_cloud.ply'));points=np.asarray(cloud.points);colors=np.asarray(cloud.colors)
    A,floor,ceiling,bounds,lines=room_frame(points,poses)
    objects,object_arrays=fuse_objects(observations,arrays,A,floor,points)
    draw_plan(points,colors,A,floor,poses,bounds,lines,objects)
    mesh=reconstruct_mesh(model,K,poses,R,t,selected,points)
    render_3d(mesh,points,colors,objects,object_arrays,A,floor,bounds,ceiling)
    upgrade.dump('room_view_manifest.json',{'objects':len(objects),'classes':[o['class_name'] for o in objects],'selected_frames':len(selected),'total_runtime_seconds':time.perf_counter()-start,'detector':'YOLOE-26l instance segmentation with disjoint visual reference frames','localization':'Depth-mask pixels, two-view voxel support, CLIP semantic review, registered-cloud support','limitations':'Offline reconstruction; inferred calibration/scale; partial object surfaces; no annotated detection/shape benchmark. Structural envelope is inferred.'})
    print('ROOM VIEWS COMPLETE',flush=True)

if __name__=='__main__':main()
