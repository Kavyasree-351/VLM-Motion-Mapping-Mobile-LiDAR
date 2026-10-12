"""Official SAM image/video trials, isolated from the CPU mapping environment."""
from pathlib import Path
import argparse, json, time, sys, inspect
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.models/sam3-source/sam3-main'))
import numpy as np
import torch
from PIL import Image, ImageDraw
# Windows PyTorch wheels do not include FlashAttention. The official decoder
# forces that backend even when use_fa3=False; allow PyTorch's supported SDPA
# implementations instead, without changing attention weights or equations.
import sam3.model.decoder as sam_decoder
from torch.nn.attention import sdpa_kernel, SDPBackend
def portable_sdpa(_backend):
    return sdpa_kernel([SDPBackend.FLASH_ATTENTION,SDPBackend.EFFICIENT_ATTENTION,SDPBackend.CUDNN_ATTENTION,SDPBackend.MATH])
sam_decoder.sdpa_kernel=portable_sdpa
original_attention=torch.nn.functional.scaled_dot_product_attention
def portable_attention(q,k,v,*args,**kwargs):
    # Multiplex uses singleton-batch q/k with multi-object values. Implicit
    # broadcasting selects the quadratic-memory math kernel on Windows.
    # Materializing the broadcast enables the efficient attention kernel.
    batch=torch.broadcast_shapes(q.shape[:-2],k.shape[:-2],v.shape[:-2])
    if q.shape[:-2]!=k.shape[:-2] or q.shape[:-2]!=v.shape[:-2]:
        q=q.expand(*batch,*q.shape[-2:]).contiguous()
        k=k.expand(*batch,*k.shape[-2:]).contiguous()
        v=v.expand(*batch,*v.shape[-2:]).contiguous()
    return original_attention(q,k,v,*args,**kwargs)
torch.nn.functional.scaled_dot_product_attention=portable_attention
DATA = ROOT / 'dataset/content/vlmaps_dataset/my_custom_scene/rgb'
OUT = ROOT / 'outputs/sam_trial'
OUT.mkdir(parents=True, exist_ok=True)
FRAMES = [0,300,450,600,750,900,1050,1200,1350,1500,1720,2000]
PROMPTS = ['sofa','armchair','beanbag','storage table','shelf','door','window','picture','air vent','ceiling light','person']

def save_visual(image, masks, labels, path):
    a = np.asarray(image).copy()
    for i, mask in enumerate(masks):
        color = np.array([(71+i*53)%255,(149+i*79)%255,(227+i*31)%255])
        a[mask] = (.55*a[mask]+.45*color).astype(np.uint8)
    im = Image.fromarray(a); draw = ImageDraw.Draw(im)
    for mask,label in zip(masks,labels):
        yy,xx=np.where(mask)
        if len(xx): draw.text((int(xx.min()),int(yy.min())),label,fill='white',stroke_width=2,stroke_fill='black')
    im.save(path)

def image_trial():
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    model=build_sam3_image_model(checkpoint_path=str(ROOT/'.models/sam3/sam3.pt'),load_from_HF=False,compile=False)
    processor=Sam3Processor(model,confidence_threshold=.25)
    rows=[]; small={}; start=time.perf_counter()
    for frame in FRAMES:
        image=Image.open(DATA/f'{frame:06d}.png').convert('RGB'); masks=[]; labels=[]
        with torch.inference_mode(), torch.autocast('cuda',dtype=torch.bfloat16):
            state=processor.set_image(image)
            for prompt in PROMPTS:
                processor.reset_all_prompts(state)
                result=processor.set_text_prompt(prompt,state)
                for score,box,mask in zip(result['scores'],result['boxes'],result['masks']):
                    mask=mask.squeeze().cpu().numpy().astype(bool)
                    key=str(len(rows)); small[key]=np.asarray(Image.fromarray(mask).resize((192,256),Image.Resampling.NEAREST))
                    rows.append(dict(observation_id=int(key),frame=frame,vlm_label=prompt.replace(' ','_'),confidence=float(score),box=box.cpu().tolist(),mask_pixels=int(mask.sum())))
                    masks.append(mask); labels.append(f'{prompt} {float(score):.2f}')
        save_visual(image,masks,labels,OUT/f'sam3_{frame:06d}.jpg')
        (OUT/'sam3_detections.json').write_text(json.dumps(rows,indent=2))
        np.savez_compressed(OUT/'sam3_masks.npz',**small)
        print(f'image {frame}: {len(masks)} regions; total {time.perf_counter()-start:.1f}s',flush=True)
    (OUT/'image_runtime.json').write_text(json.dumps(dict(seconds=time.perf_counter()-start,frames=FRAMES,prompts=PROMPTS,threshold=.25,peak_vram_bytes=torch.cuda.max_memory_allocated()),indent=2))

def hybrid_trial():
    import pandas as pd
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    observations=pd.read_csv(ROOT/'outputs/room_refined/detections_visual.csv')
    model=build_sam3_image_model(checkpoint_path=str(ROOT/'.models/sam3/sam3.pt'),load_from_HF=False,compile=False)
    processor=Sam3Processor(model,confidence_threshold=.25)
    masks_saved={}; rows=[]; start=time.perf_counter()
    for frame,group in observations.groupby('frame'):
        image=Image.open(DATA/f'{int(frame):06d}.png').convert('RGB'); w,h=image.size
        masks=[]; labels=[]
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            state=processor.set_image(image)
            for row in group.itertuples():
                box=np.array([row.x0,row.y0,row.x1,row.y1])
                center=(box[:2]+box[2:])/2; extent=box[2:]-box[:2]
                processor.reset_all_prompts(state)
                result=processor.add_geometric_prompt([center[0]/w,center[1]/h,extent[0]/w,extent[1]/h],True,state)
                candidates=result['boxes'].cpu().numpy()
                if not len(candidates):continue
                intersection=np.maximum(0,np.minimum(box[2:],candidates[:,2:])-np.maximum(box[:2],candidates[:,:2])).prod(axis=1)
                union=extent.prod()+(candidates[:,2:]-candidates[:,:2]).prod(axis=1)-intersection
                iou=intersection/np.maximum(union,1); best=int(np.argmax(iou))
                if iou[best]<.5:continue
                mask=result['masks'][best].squeeze().cpu().numpy().astype(bool)
                masks_saved[str(int(row.observation_id))]=np.asarray(Image.fromarray(mask).resize((192,256),Image.Resampling.NEAREST))
                rows.append(dict(observation_id=int(row.observation_id),frame=int(frame),class_name=row.class_name,sam_score=float(result['scores'][best]),prompt_box_iou=float(iou[best]),mask_pixels=int(mask.sum())))
                masks.append(mask);labels.append(row.class_name)
        if int(frame) in FRAMES:save_visual(image,masks,labels,OUT/f'hybrid_{int(frame):06d}.jpg')
        if len(rows)%20<5:print(f'hybrid frame {int(frame)}: {len(rows)} accepted masks',flush=True)
    np.savez_compressed(OUT/'hybrid_masks.npz',**masks_saved)
    pd.DataFrame(rows).to_csv(OUT/'hybrid_mask_quality.csv',index=False)
    (OUT/'hybrid_runtime.json').write_text(json.dumps(dict(seconds=time.perf_counter()-start,frames=int(observations.frame.nunique()),candidate_observations=len(observations),accepted_masks=len(rows),minimum_prompt_box_iou=.5,peak_vram_bytes=torch.cuda.max_memory_allocated()),indent=2))

def video_trial(prompt='furniture'):
    from sam3.model_builder import build_sam3_predictor
    video=OUT/'video_frames'; video.mkdir(exist_ok=True)
    frames=list(range(400,490))
    for i,frame in enumerate(frames):
        Image.open(DATA/f'{frame:06d}.png').convert('RGB').save(video/f'{i:06d}.jpg',quality=95)
    (OUT/'video_source_frames.json').write_text(json.dumps(frames))
    predictor=build_sam3_predictor(version='sam3.1',checkpoint_path=str(ROOT/'.models/sam31/sam3.1_multiplex.pt'),compile=False,warm_up=False,use_fa3=False,async_loading_frames=False)
    # Official defaults batch 16 full-resolution frames for large GPUs.
    # Serial batches retain the checkpoint's 16-object multiplex capacity.
    predictor.model.batched_grounding_batch_size=1
    predictor.model.postprocess_batch_size=1
    # Official shared request dispatcher passes a state-offload argument that
    # the multiplex model does not support. Explicitly reject requested state
    # offload and omit the unused False value; video frames still stay on CPU.
    if 'offload_state_to_cpu' not in inspect.signature(predictor.model.init_state).parameters:
        original_init=predictor.model.init_state
        def compatible_init(**kwargs):
            if kwargs.pop('offload_state_to_cpu',False):
                raise ValueError('Multiplex state offload is unsupported')
            return original_init(**kwargs)
        predictor.model.init_state=compatible_init
    sid=predictor.handle_request(dict(type='start_session',resource_path=str(video),offload_video_to_cpu=True))['session_id']
    predictor.handle_request(dict(type='add_prompt',session_id=sid,frame_index=0,text=prompt,output_prob_thresh=.25))
    rows=[]; masks_saved={}; start=time.perf_counter()
    for response in predictor.handle_stream_request(dict(type='propagate_in_video',session_id=sid,propagation_direction='forward',start_frame_index=0,output_prob_thresh=.25)):
        i=int(response['frame_index']); out=response['outputs']
        masks=np.asarray(out['out_binary_masks']); ids=np.asarray(out['out_obj_ids'])
        for oid,mask in zip(ids,masks):
            rows.append(dict(frame=frames[i],clip_frame=i,object_id=int(oid),mask_pixels=int(mask.sum())))
            masks_saved[f'{i}_{int(oid)}']=np.asarray(Image.fromarray(mask).resize((192,256),Image.Resampling.NEAREST))
        if i%10==0:
            save_visual(Image.open(video/f'{i:06d}.jpg'),masks,[f'{prompt} ID {int(x)}' for x in ids],OUT/f'sam31_{prompt}_{frames[i]:06d}.jpg')
        (OUT/f'sam31_{prompt}_tracks.json').write_text(json.dumps(rows,indent=2))
        print(f'video {i}: ids={ids.tolist()}',flush=True)
    np.savez_compressed(OUT/f'sam31_{prompt}_masks.npz',**masks_saved)
    predictor.handle_request(dict(type='close_session',session_id=sid))
    (OUT/f'video_{prompt}_runtime.json').write_text(json.dumps(dict(seconds=time.perf_counter()-start,source_frames=frames,prompt=prompt,threshold=.25,peak_vram_bytes=torch.cuda.max_memory_allocated()),indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('mode',choices=['image','video','hybrid']); parser.add_argument('--prompt',default='furniture'); args=parser.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required for this isolated trial')
    print(torch.__version__,torch.cuda.get_device_name(),flush=True)
    if args.mode=='image':image_trial()
    elif args.mode=='hybrid':hybrid_trial()
    else:video_trial(args.prompt)
