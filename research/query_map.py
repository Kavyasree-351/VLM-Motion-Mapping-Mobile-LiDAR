"""Sparse language retrieval over mapped regions with an unresolved outcome."""
from pathlib import Path
import sys,os,json,argparse
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.packages'))
os.environ['HF_HUB_OFFLINE']='1'
import numpy as np
import pandas as pd
import torch
from transformers import CLIPModel,CLIPProcessor
from verify_semantics import features

OUT=ROOT/'outputs/research_improved'

def build_index():
    boxes=pd.read_csv(OUT/'vlm_verified_boxes.csv')
    pack=np.load(OUT/'vlm_region_embeddings.npz');emb=pack['image_embeddings']
    lm=pd.read_csv(OUT/'semantic_landmarks.csv');obs=pd.read_csv(OUT/'semantic_observations.csv')
    poses=np.loadtxt(ROOT/'dataset/content/vlmaps_dataset/my_custom_scene/poses.txt')
    pooled=[];records=[]
    for _,landmark in lm.iterrows():
        candidates=boxes[(boxes.vlm_status=='accepted')&(boxes.vlm_label==landmark.class_name)]
        votes=[];weights=[];centers=[]
        for index,b in candidates.sort_values('confidence',ascending=False).iterrows():
            available=obs[(obs.frame==b.frame)&(obs.class_name==landmark.class_name)]
            if available.empty:continue
            location=available[['world_x','world_z']].to_numpy()
            if np.linalg.norm(location-[landmark.world_x,landmark.world_z],axis=1).min()>landmark.spread_radius+.45:continue
            camera=poses[int(b.frame),[0,2]]
            if centers and np.linalg.norm(np.array(centers)-camera,axis=1).min()<.10:continue
            centers.append(camera);votes.append(emb[index]);weights.append(b.confidence)
        if not votes:continue
        vector=np.average(votes,axis=0,weights=weights);vector/=np.linalg.norm(vector)
        pooled.append(vector);record=landmark.to_dict();record['embedding_views']=len(votes);records.append(record)
    np.savez_compressed(OUT/'language_landmark_index.npz',embeddings=np.array(pooled))
    (OUT/'language_landmark_index.json').write_text(json.dumps(records,indent=2))
    return records,np.array(pooled)

def query(text,model,processor,records,embeddings):
    with torch.inference_mode():
        f=features(model.get_text_features(**processor(text=[f'a photo of {text}.'],return_tensors='pt',padding=True)))
        f=f/f.norm(dim=-1,keepdim=True)
    scores=embeddings@f[0].cpu().numpy();order=np.argsort(scores)[::-1]
    top=int(order[0]);margin=float(scores[top]-scores[order[1]]) if len(order)>1 else float(scores[top])
    accepted=float(scores[top])>=.23 and margin>=.015
    result={'query':text,'status':'candidate' if accepted else 'unresolved','cosine_similarity':float(scores[top]),'margin':margin,'probability':None,'note':'Similarity heuristic over sparse detected regions; candidate position can remain tentative.'}
    if accepted:result['landmark']=records[top]
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--query');args=parser.parse_args()
    torch.set_num_threads(4);records,emb=build_index()
    model=CLIPModel.from_pretrained(ROOT/'.models/clip',local_files_only=True).eval()
    processor=CLIPProcessor.from_pretrained(ROOT/'.models/clip',local_files_only=True)
    queries=[args.query] if args.query else ['an orange beanbag chair','a black leather sofa','a refrigerator']
    results=[query(text,model,processor,records,emb) for text in queries]
    (OUT/'language_queries.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(results,indent=2),flush=True)

if __name__=='__main__':main()
