"""Package measured trial results and an executable review notebook."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'.packages'))
import room_upgrade as upgrade
import numpy as np,pandas as pd
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nbformat
OUT=ROOT/'outputs/next_trials'

def main():
    diagnostics=json.loads((OUT/'diagnostics.json').read_text());depth=json.loads((OUT/'da3_trial.json').read_text());tracks=json.loads((OUT/'trial_summary.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(12,4.5));paired=diagnostics['paired_motion']
    axes[0].bar(['Eroded boxes','Instance masks'],[paired['box_median_pixels'],paired['mask_median_pixels']],color=['#7e9a9d','#367d86']);axes[0].set(title='Object motion: usable pixels',ylabel='Median valid pixels · 62 matched observations')
    data=depth['in_batch_consistency'];x=np.arange(2)
    axes[1].bar(x-.16,[100*d['median_reprojection_disagreement']['calibrated_export'] for d in data],.32,label='Calibrated export',color='#7e9a9d')
    axes[1].bar(x+.16,[100*d['median_reprojection_disagreement']['DA3_aligned'] for d in data],.32,label='DA3 small',color='#bc884b');axes[1].set(xticks=x,xticklabels=['450 → 470','1500 → 1520'],title='Depth: cross-view consistency',ylabel='Median relative disagreement (%)');axes[1].legend()
    fig.text(.05,.01,'Diagnostics, not ground-truth accuracy. DA3 sees each triplet jointly; it helps the difficult pair but worsens the stable pair.',fontsize=9)
    fig.tight_layout(rect=(0,.06,1,1));fig.savefig(OUT/'trial_comparison.png',dpi=170);plt.close(fig)
    model,*_=upgrade.load_geometry();saved=np.load(OUT/'da3_aligned_depth.npz');fig,axes=plt.subplots(2,3,figsize=(10,9))
    for row,frame in enumerate([450,1500]):
        _,rgb,z,valid=upgrade.mapping.load_frame(frame,model);pred=saved[f'depth_{frame}'];lo,hi=np.percentile(z[valid],[2,98])
        axes[row,0].imshow(z,cmap='viridis',vmin=lo,vmax=hi);axes[row,0].set_title(f'Frame {frame}: calibrated export')
        axes[row,1].imshow(pred,cmap='viridis',vmin=lo,vmax=hi);axes[row,1].set_title('DA3 aligned · same colour scale')
        error=np.where(valid,abs(pred-z)/np.maximum(z,1e-4),np.nan);im=axes[row,2].imshow(error,cmap='magma',vmin=0,vmax=.2);axes[row,2].set_title('Relative disagreement, not GT error');fig.colorbar(im,ax=axes[row,2],fraction=.04)
        for ax in axes[row]:ax.axis('off')
    fig.tight_layout();fig.savefig(OUT/'da3_depth_comparison.png',dpi=150);plt.close(fig)
    report='''# Dataset trials: results and collection needs

These experiments ran locally on the existing iPhone scan. Original nine-region room outputs remain unchanged. The trials use published ideas but are lightweight adaptations, not reproductions of OVI-MAP, SuperMap or FUS3DMaps.

## What actually happened

| Trial | Dataset result | Decision |
|---|---|---|
| Geometry-first persistent identity | 34 candidate tracks, 14 seen in at least three frames. The same sofa is fragmented into multiple spatial tracks; the door is not reactivated on revisit. Synthetic label changes do not alter IDs, but geometry-only matching is insufficient in this scan. | Keep experimental; add appearance/reprojection association and visibility-aware reactivation before replacing the nine-region map. |
| Actual YOLOE masks for motion | Re-inferred 78 masks on 34 frames and compared masks with boxes. On the same 62 eligible observations, median usable pixels rise from 1,414 to 2,423. Mean flags are 0.603% for boxes and 0.709% for masks. | More surface evidence, not demonstrated motion accuracy. Earlier aggregate means used unequal eligible sets and overstated the apparent flag reduction. |
| DA3 small depth | Six frames, two pose-conditioned triplets, CPU float32. Scale/offset fit on neighbouring views gives middle-view median disagreement with calibrated export of 1.68% at frame 450 and 7.45% at 1500. | Feasible candidate for difficult regions, not a universal depth replacement. |
| DA3 consistency | Matched source-pixel reprojection disagreement: pair 450→470 worsens from 0.391% to 0.838%; pair 1500→1520 improves from 15.86% to 0.775%. | The difficult pair benefits strongly, but these are in-batch consistency proxies, not held-out geometry accuracy. No predicted pixels were inserted into the final mesh. |
| Adaptive cloud-support radius | Six frame pairs, radius relaxed from 0.11 up to 0.18 pose units. Adds little on stable frames; at frame 2000, support rises from 10,946 to 15,913 but mismatch fraction rises from 63.8% to 74.5%. Frame 1500 also has large target-depth disagreement. | Reject blind radius relaxation. Source-cloud proximity alone cannot certify visibility or depth accuracy. |
| Architectural relations | Door, picture and two window regions receive nearest measured-wall relation candidates. | Useful metadata; these do not verify wall openings, door swing or closed room topology. |
| Coarse language patches | 72 overlapping patches from six frames; local CLIP embeddings occupy 147,456 bytes. Queries can rank wall/ceiling/furniture patches beyond the object inventory. | A feasible query layer. Ranking an absent refrigerator still produces patches, so it is not yet safe as an existence detector or a complete dense 3D map. |

The motion trial finds 10 tracks not observed moving and 24 with insufficient/ambiguous motion evidence. It does not invent movers in a largely static scan. New histories use experimental identity tracks, not an asserted one-to-one match to all nine final room objects.

## Hardware and deferred models

Hardware probe found an NVIDIA RTX 4080 Laptop GPU with 12,282 MiB VRAM. The current workspace PyTorch build is CPU-only. DA3 small ran on CPU; model compute timings are limited feasibility results and exclude download/setup. SAM 3.1 was not run: official checkpoint access is gated, and the user has submitted the access request and Meta review is pending. Check https://huggingface.co/facebook/sam3 while signed in; no token needs to be shared in chat. A separate GPU environment is preferable to replacing working mapping dependencies. Full MapAnything and Gaussian-SLAM benchmarks were not run in this trial; the small DA3 experiment tests the depth-prior direction.

Official DA3 source and checkpoint are retained under `.models`. The current source adds optional auxiliary ray heads absent from the small checkpoint; missing keys were restricted to those heads. Main depth/confidence weights were loaded, and ray-head pose inference was not used. This is a low-resolution direct-network experiment rather than the full official evaluation recipe.

## Do you need more data?

**For improving this room: the existing scan is sufficient for further engineering trials. For trustworthy research claims: yes.** More frames from the same route alone will not resolve the key missing evidence.

1. Recover or record original metric float32 LiDAR depth, per-pixel confidence, RGB and depth timestamps, per-frame camera intrinsics, display/orientation transform, camera poses and the exact export code/settings. Preserve depth values rather than normalizing each frame to 0–255. This is the highest-priority collection improvement.
2. As an initial research collection, aim for 5–10 varied rooms with at least two independent passes each. This is a practical starting recommendation, not a statistically sufficient sample-size claim. Include repeated same-class chairs, doors/windows, textureless walls, glass, clutter and occlusion. Reserve entire rooms for evaluation; adjacent frames are not independent test scenes.
3. Include labelled static sequences and controlled object movement, especially slow motion toward/away from the camera, lateral movement, occlusion and disappearance/reappearance. Record synchronized timestamps and event intervals. Differentiate sensor/camera changes from object changes.
4. Annotate instance masks and stable IDs on selected held-out frames, including missed objects and negative queries. Measure a few object positions, room dimensions and wall/opening locations independently. A reference mesh would enable geometry accuracy/completeness evaluation.
5. If raw metric depth exists, generate controlled normalized/quantized/clipped export variants. This isolates whether decoder-aware tracking/motion is a research contribution rather than a repair of this particular export.

## Reproduce

Use `C:/Users/yuvar/anaconda3/python.exe`. Run `research/try_improvements.py`, `research/try_da3.py`, and `research/trial_diagnostics.py`; then `research/package_trials.py`. Mask inference is cached. The exploratory notebook is `notebookgyu-cpu.next-trials.ipynb`; its executed counterpart includes review figures. Fourteen regression tests pass, including label-flip invariance and preserving two same-class synthetic instances. No real motion, identity or geometry ground-truth benchmark is claimed.

Sources and proposed next work are linked in [NEXT_IMPROVEMENTS.md](NEXT_IMPROVEMENTS.md). The strongest next task is appearance-assisted reactivation plus calibration-aware motion on stable identities, evaluated against raw-depth and known-motion data.
'''
    (ROOT/'research/DATASET_TRIAL_REPORT.md').write_text(report,encoding='utf-8')
    n=nbformat.v4.new_notebook();n.cells=[nbformat.v4.new_markdown_cell('# Research trials on the iPhone scan\nSeparate experimental outputs; original room reconstruction is preserved. Read `research/DATASET_TRIAL_REPORT.md` for measured results and additional data requirements.'),nbformat.v4.new_code_cell('''from pathlib import Path
import sys,json
ROOT=Path.cwd()
assert (ROOT/'research').exists(), 'Run from D:/VLM-maps'
sys.path.insert(0,str(ROOT/'.packages'))
sys.path.insert(0,str(ROOT/'research'))
from IPython.display import display,Image
OUT=ROOT/'outputs/next_trials'
print('Runtime:',sys.executable)'''),nbformat.v4.new_code_cell('''# Cached actual masks avoid re-running inference when present.
import try_improvements
try_improvements.main()'''),nbformat.v4.new_code_cell('''import try_da3
try_da3.main()'''),nbformat.v4.new_code_cell('''import trial_diagnostics
trial_diagnostics.main()'''),nbformat.v4.new_code_cell('''import package_trials
package_trials.main()
display(Image(filename=str(OUT/'trial_comparison.png')))
display(Image(filename=str(OUT/'da3_depth_comparison.png')))''')]
    n.metadata['kernelspec']={'display_name':'Python (local mapping)','language':'python','name':'python3'}
    nbformat.write(n,ROOT/'notebookgyu-cpu.next-trials.ipynb')
    print('Packaged trial report and notebook',flush=True)

if __name__=='__main__':main()
