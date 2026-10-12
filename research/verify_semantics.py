"""Verify object proposals against a text vocabulary; abstain on weak margins.

Cosine similarity is not an accuracy probability. Candidate vocabulary is an
indoor-object vocabulary, not an exhaustive open-world recognition benchmark.
"""
from pathlib import Path
import sys,os,json
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.packages'))
os.environ['HF_HOME']=str(ROOT/'.models/huggingface')
os.environ['HF_HUB_OFFLINE']='1'
import torch
import pandas as pd
import numpy as np
from PIL import Image
from transformers import CLIPModel,CLIPProcessor
from geometry_probe import DATA

OUT=ROOT/'outputs/research_improved'
VOCAB={
 'beanbag':['beanbag chair','bean bag seat'],
 'bed':['bed with mattress','bed'],
 'armchair':['armchair','single upholstered chair'],
 'sofa':['sofa','couch'],
 'office chair':['office chair','desk chair'],
 'table':['table','small side table'],
 'shelf':['shelving unit','bookcase'],
 'cabinet':['cabinet','cupboard'],
 'refrigerator':['refrigerator','fridge'],
 'door':['door','wooden door'],
 'window':['window','glass window'],
 'person':['person','human'],
 'lamp':['lamp','light fixture'],
 'plant':['plant','potted plant'],
 'picture':['framed picture','wall painting'],
 'bag':['bag','backpack'],
 'curtain':['curtain','window blind'],
 'television':['television','TV screen'],
}

def features(value):
    if torch.is_tensor(value):return value
    if hasattr(value,'pooler_output'):return value.pooler_output
    raise TypeError(type(value))

def main():
    torch.set_num_threads(4)
    model=CLIPModel.from_pretrained(ROOT/'.models/clip',local_files_only=True).eval()
    processor=CLIPProcessor.from_pretrained(ROOT/'.models/clip',local_files_only=True)
    labels=list(VOCAB);emb=[]
    with torch.inference_mode():
        for label in labels:
            prompts=[f'a photo of a {name}.' for name in VOCAB[label]]
            tokens=processor(text=prompts,return_tensors='pt',padding=True)
            f=features(model.get_text_features(**tokens));f=f/f.norm(dim=-1,keepdim=True);f=f.mean(dim=0);f=f/f.norm();emb.append(f)
        text=torch.stack(emb)
        boxes=pd.read_csv(OUT/'detector_boxes.csv');all_scores=[];image_embeddings=[]
        for start in range(0,len(boxes),12):
            crops=[]
            for _,b in boxes.iloc[start:start+12].iterrows():
                im=Image.open(DATA/'rgb'/f'{int(b.frame):06d}.png').convert('RGB')
                crops.append(im.crop((b.x0,b.y0,b.x1,b.y1)))
            inputs=processor(images=crops,return_tensors='pt')
            f=features(model.get_image_features(**inputs));f=f/f.norm(dim=-1,keepdim=True)
            image_embeddings.append(f.cpu().numpy())
            all_scores.append((f@text.T).cpu().numpy())
            print('CLIP crops',min(start+12,len(boxes)),'/',len(boxes),flush=True)
    scores=np.vstack(all_scores);order=np.argsort(scores,axis=1);top=order[:,-1];second=order[:,-2]
    boxes['vlm_label']=[labels[k] for k in top];boxes['vlm_cosine']=scores[np.arange(len(scores)),top];boxes['vlm_margin']=scores[np.arange(len(scores)),top]-scores[np.arange(len(scores)),second]
    # Fixed heuristic thresholds, no labelled test tuning.
    accepted=(boxes.vlm_cosine>=.23)&(boxes.vlm_margin>=.015)
    boxes['vlm_status']=np.where(accepted,'accepted','unresolved')
    boxes.to_csv(OUT/'vlm_verified_boxes.csv',index=False)
    np.savez_compressed(OUT/'vlm_region_embeddings.npz',image_embeddings=np.vstack(image_embeddings),text_embeddings=text.cpu().numpy(),similarities=scores,labels=np.array(labels),frames=boxes.frame.to_numpy())
    report={'model':'openai/clip-vit-base-patch32','proposals':len(boxes),'accepted':int(accepted.sum()),'unresolved':int((~accepted).sum()),'original_bed_proposals':int((boxes.class_name=='bed').sum()),'bed_proposals_reidentified_as_beanbag':int(((boxes.class_name=='bed')&(boxes.vlm_label=='beanbag')&accepted).sum()),'vocabulary':VOCAB,'acceptance_rule':'cosine >= 0.23 and top-two margin >= 0.015; heuristic, not probability calibration','limitations':'Proposal recall limited by YOLO; no dense language map; vocabulary-conditioned decisions; CLIP acceptance does not prove class correctness.'}
    (OUT/'vlm_verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
