"""YOLO26 mask-based room mapping and measured 2D/3D visualization.

Original research outputs are retained; upgraded artifacts live in room_refined.
"""
from pathlib import Path
import sys,os,json,time,argparse
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.packages'))
os.environ['YOLO_CONFIG_DIR']=str(ROOT/'.yolo')
os.environ['MPLCONFIGDIR']=str(ROOT/'.matplotlib')
os.environ['HF_HUB_OFFLINE']='1'
import cv2,numpy as np,pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull,cKDTree
from scipy.spatial.transform import Rotation
from sklearn.cluster import DBSCAN
import open3d as o3d
from ultralytics import YOLO,YOLOE
from geometry_probe import DATA,cameras
import improve_mapping as mapping

OUT=ROOT/'outputs/room_refined';OUT.mkdir(parents=True,exist_ok=True)
MODELS=ROOT/'.models/yolo'
PALETTE={'sofa':'#357b83','armchair':'#965a7a','beanbag':'#d88142','table':'#b1945b','storage table':'#b1945b','shelf':'#827c56','door':'#976d4c','window':'#5b9fbc','picture':'#857daa','air vent':'#78858f','ceiling light':'#c3a247','person':'#b84949'}
SAMPLE_FRAMES=[0,300,450,600,750,900,1050,1200,1350,1500,1720,2000]

def dump(name,data):
    (OUT/name).write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8')

def load_geometry():
    pack=np.load(mapping.OUT/'depth_calibration.npz');model={k:pack[k] for k in ['frames','coeff','inverse','residual_spread','anchor_counts']}
    K=pack['K_half'].copy();K[:2,:]*=192/360
    poses=np.loadtxt(DATA/'poses.txt');R,t=cameras(poses,'c2w_opengl')
    return model,K,poses,R,t

def benchmark():
    torch.set_num_threads(4);rows=[]
    for label,file,kind in [('YOLO11n',ROOT/'yolo11n.pt',YOLO),('YOLO26m-seg',MODELS/'yolo26m-seg.pt',YOLO),('YOLOE-26l-seg',MODELS/'yoloe-26l-room-seg.pt',YOLOE),('YOLOE-26l-visual',MODELS/'yoloe-26l-room-visual-seg.pt',YOLOE)]:
        detector=kind(str(file));start=time.perf_counter()
        for frame in SAMPLE_FRAMES:
            result=detector.predict(str(DATA/'rgb'/f'{frame:06d}.png'),imgsz=960,conf=.25,device='cpu',verbose=False,retina_masks=True)[0]
            for box,conf,labelid in zip(result.boxes.xyxy.cpu().numpy(),result.boxes.conf.cpu().numpy(),result.boxes.cls.cpu().numpy()):
                rows.append({'model':label,'frame':frame,'class_name':detector.names[int(labelid)],'confidence':float(conf),'x0':float(box[0]),'y0':float(box[1]),'x1':float(box[2]),'y1':float(box[3]),'has_mask':result.masks is not None})
            result.save(str(OUT/f'compare_{label}_{frame:06d}.jpg'))
        print('Benchmark',label,time.perf_counter()-start,flush=True)
        del detector
    df=pd.DataFrame(rows);df.to_csv(OUT/'detector_comparison.csv',index=False)
    dump('detector_protocol.json',{'models':['YOLO11n','YOLO26m-seg','YOLOE-26l-seg','YOLOE-26l-visual'],'same_frames':SAMPLE_FRAMES,'image_size':960,'confidence_threshold':.25,'device':'cpu','comparison':'matched-frame qualitative class/mask audit; no labelled detection test set and no mAP claim','reference':'https://docs.ultralytics.com/models/yoloe'})
    # Pair the actual predictions; count is not used as an accuracy score.
    from PIL import Image,ImageDraw
    sheet=Image.new('RGB',(1440,400*6),'white');draw=ImageDraw.Draw(sheet)
    for row,frame in enumerate([0,450,750,1050,1500,2000]):
        for col,label in enumerate(['YOLO11n','YOLO26m-seg','YOLOE-26l-seg','YOLOE-26l-visual']):
            im=Image.open(OUT/f'compare_{label}_{frame:06d}.jpg');im.thumbnail((350,365));sheet.paste(im,(col*360,row*400+28));draw.text((col*360+5,row*400+5),f'{label} / frame {frame}',fill='black')
    sheet.save(OUT/'detector_comparison_contact.png')
    return df

def detect():
    torch.set_num_threads(4);cv2.setNumThreads(4)
    model,K,poses,R,t=load_geometry();selected=np.unique(np.r_[mapping.keyframes(poses),np.arange(0,len(poses),50),len(poses)-1]).astype(int)
    detector=YOLOE(str(MODELS/'yoloe-26l-room-visual-seg.pt'))
    rows=[];piles=[];colors=[];point_ids=[];times=[]
    cache=OUT/'mask_observations_visual.npz'
    if cache.exists() and (OUT/'detections_visual.csv').exists():
        print('Using cached mask observations',flush=True);return pd.read_csv(OUT/'detections_visual.csv'),dict(np.load(cache)),selected
    start=time.perf_counter()
    for count,frame in enumerate(selected):
        raw,rgb,depth,valid=mapping.load_frame(int(frame),model)
        xyz=mapping.backproject(depth,K)@R[frame].T+t[frame]
        result=detector.predict(str(DATA/'rgb'/f'{frame:06d}.png'),imgsz=960,conf=.18,device='cpu',verbose=False,retina_masks=True)[0]
        if result.masks is None:continue
        boxes=result.boxes.xyxy.cpu().numpy();scores=result.boxes.conf.cpu().numpy();labels=result.boxes.cls.cpu().numpy();masks=result.masks.data.cpu().numpy()
        for box,score,labelid,mask in zip(boxes,scores,labels,masks):
            label=detector.names[int(labelid)]
            small=cv2.resize(mask,(192,256),interpolation=cv2.INTER_NEAREST)>.5
            # Erosion removes mixed-depth boundary pixels instead of box background.
            small=cv2.erode(small.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
            keep=small&valid
            if keep.sum()<25:continue
            points=xyz[keep];col=rgb[keep]/255.
            med=np.median(points,axis=0);dist=np.linalg.norm(points-med,axis=1)
            inside=dist<np.percentile(dist,95)
            points=points[inside];col=col[inside]
            idx=len(rows)
            rows.append({'observation_id':idx,'frame':int(frame),'class_name':label,'confidence':float(score),'x0':float(box[0]),'y0':float(box[1]),'x1':float(box[2]),'y1':float(box[3]),'world_x':float(med[0]),'world_y':float(med[1]),'world_z':float(med[2]),'mask_depth_pixels':int(len(points)),'calibration_spread':float(mapping.calibration_spread(model,frame))})
            step=max(1,len(points)//2000);piles.append(points[::step]);colors.append(col[::step]);point_ids.append(np.full(len(points[::step]),idx,np.int32))
        if count%15==0:print('YOLOE mask keyframe',count,'/',len(selected),flush=True)
        if frame in SAMPLE_FRAMES:result.save(str(OUT/f'upgraded_mask_{frame:06d}.jpg'))
    df=pd.DataFrame(rows);df.to_csv(OUT/'detections_visual.csv',index=False)
    arrays={'points':np.vstack(piles),'colors':np.vstack(colors),'observation_ids':np.concatenate(point_ids)}
    np.savez_compressed(cache,**arrays)
    dump('detector_run_visual.json',{'model':'yoloe-26l-seg.pt','prompted_model':'yoloe-26l-room-visual-seg.pt','prompts':list(detector.names.values()),'frames':len(selected),'observations':len(df),'image_size':960,'confidence_threshold':.18,'mask_boundary_erosion_depth_pixels':1,'runtime_seconds':time.perf_counter()-start,'device':'cpu','label_status':'model hypotheses; scene-specific visual reference adaptation; require multi-view support and semantic review'})
    return df,arrays,selected

def configure_visual_prompts():
    """One-shot exemplars from inspected reference frames, disjoint from audit frames."""
    torch.set_num_threads(4)
    names=['sofa','armchair','beanbag','storage table','shelf','door','window','picture','air vent','ceiling light','person']
    detector=YOLOE(str(MODELS/'yoloe-26l-seg.pt'));detector.model.eval()
    embeddings=detector.get_text_pe(names).clone()
    references={'armchair':(1600,[260,480,580,950]),'beanbag':(400,[120,580,512,890]),'storage table':(1000,[95,590,400,940]),'door':(50,[320,140,670,950]),'window':(1600,[0,0,390,560]),'picture':(1000,[410,175,520,360])}
    for label,(frame,box) in references.items():
        rgb=cv2.imread(str(DATA/'rgb'/f'{frame:06d}.png'))[:,:,::-1].copy()
        image=cv2.resize(rgb,(480,640));tensor=torch.from_numpy(image.transpose(2,0,1)).float().unsqueeze(0)/255.
        visual=torch.zeros((1,1,640,480));x0,y0,x1,y1=(np.array(box)*2/3).astype(int)
        visual[0,0,y0:y1,x0:x1]=1
        visual=torch.nn.functional.interpolate(visual,size=(80,60),mode='nearest')
        with torch.inference_mode():vpe=detector.get_visual_pe(tensor,visual)
        embeddings[0,names.index(label)]=vpe[0,0]
        print('Visual prompt',label,'frame',frame,'embedding',vpe.shape,flush=True)
    detector.set_classes(names,embeddings);detector.save(str(MODELS/'yoloe-26l-room-visual-seg.pt'))
    detector.save_prompt_embeddings(MODELS/'room_visual_prompts.npz')
    dump('visual_prompt_provenance.json',{'base_model':'yoloe-26l-seg.pt','classes':names,'reference_boxes':{k:{'frame':f,'box':b} for k,(f,b) in references.items()},'audit_frames_disjoint':not bool(set(f for f,b in references.values())&set(SAMPLE_FRAMES)),'training':'no gradient training; visual embedding conditioning on inspected room reference boxes','claim':'scene-specific adaptation, not zero-shot or cross-scene benchmark'})

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--benchmark',action='store_true');parser.add_argument('--configure',action='store_true');args=parser.parse_args()
    if args.configure:configure_visual_prompts()
    elif args.benchmark:benchmark()
    else:detect()

if __name__=='__main__':main()
