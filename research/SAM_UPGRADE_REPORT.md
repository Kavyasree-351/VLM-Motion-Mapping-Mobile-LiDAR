# SAM 3 / SAM 3.1 on the iPhone room dataset

The working candidate combines **YOLOE-26 large visual proposals + geometry-gated SAM 3 masks** for static mapping. **SAM 3.1** is evaluated for temporal masks separately. Newer does not automatically mean more accurate: the official release reports mixed video PCS results. No new accuracy or novelty claim is established here.

## Completed experiments

- Official SAM 3 image checkpoint: 12 identical comparison frames, 11 text prompts, confidence 0.25; 19.74 seconds including image encoding and prompt inference, excluding model loading. Peak allocated memory 5.08 GiB. Scores are not calibrated across detectors; equal thresholds do not equal operating points.
- Text-only SAM detects the beanbag, sofa and doors, but duplicates vents and splits the table into shelf parts. Box-conditioned masks preserve YOLOE's scene-conditioned labels and give a coherent table region on inspected frames. See `outputs/sam_trial/sam_comparison.png` for successes and failures.
- Hybrid: 243 object observations across 103 frames with proposals (the earlier detector processed 107 total keyframes). All 243 yielded box-matched SAM masks; 186 passed the additional 3D checks, 57 retained their original YOLOE surfaces. Proposed box IoU >=0.5, supported-cloud fraction no more than 0.05 below baseline, median center shift <0.25 pose unit, valid area ratio between 0.4 and 2.5. These are engineering gates, not ground-truth quality scores.
- Failure preserved in the comparison: the geometric prompt selects a small ceiling strip instead of the door at frame 2000. Its valid-area ratio is 0.038 and center shift 0.785 pose unit, so the refinement is rejected. Text SAM segments that door more coherently. Choosing one model for every object would conceal this regression.
- New plan and interactive 3D: `outputs/sam_room`. Nine supported regions remain. Room alignment and the measured mesh are reused; the object surfaces are refined. Missing geometry, absolute-depth uncertainty and unseen object backs remain.
- SAM 3.1: two separate text-prompt sessions on **90 consecutive source frames 400–489**, with furniture and beanbag prompts. Each has one ID and a nonempty mask on all 90 frames. This short, mostly static clip does not test long-term re-identification, occlusion, cross-room generalization or simultaneous multi-class tracking. IDs are session-local and must not be merged by number. The broad furniture prompt missed the beanbag; the specific prompt recovered it.
- Propagation took 30.38 seconds (furniture) and 29.78 seconds (beanbag), excluding loading/prompt initialization. Peak allocated GPU memory for furniture was 7.16 GiB. These timings include output handling and saving; they are not end-to-end FPS benchmarks.
- Tracked masks were passed through the existing VLMM-inspired residual/covariance pipeline on pairs 400→420 and 450→470, for sofa and beanbag. Total usable pixels: boxes 7529, SAM image masks 8917, tracked masks 9322 (**23.8% above boxes**). This is usable correspondence coverage, not motion accuracy. All four tracked-mask scores are below the existing moving threshold; one sofa observation has a small nonzero flagged fraction. No actual moving object is established by this static scene.

## Windows/GPU bottlenecks fixed

The isolated `.venv-sam31` uses Python 3.12.13, PyTorch 2.10.0+cu128, torchvision 0.25.0+cu128 and Triton Windows 3.6.0.post26. The RTX 4080 Laptop GPU has 12 GB VRAM. The original `.packages` CPU mapping runtime is preserved.

The official shared session API passed an unsupported state-offload keyword to the multiplex model; the adapter removes only the unused False argument. Windows PyTorch lacks FlashAttention, so the trial enables supported SDPA backends. Explicit attention batch broadcasting matches the original calculation (small float32 numerical check max difference 3.58e-7) and allows efficient kernels. The largest memory problem was the official 16-frame grounding batch; reducing grounding/postprocessing batches to one avoids the stall at frame 16 while retaining the checkpoint's 16-object multiplex capacity. Model weights and inference resolution remain unchanged. This configuration is not the official H100 speed configuration.

The official builder prints missing/unexpected keys during intermediate tracker-only construction. Final merged load reports 64 missing real RoPE buffers, which are generated from model positions; learned weights must not be silently skipped. The trial source adapter does not modify checkpoint parameters.

## Reproduce

Run the GPU stages with `.venv-sam31/Scripts/python.exe research/try_sam.py image`, `hybrid`, `video`, and `video --prompt beanbag` in separate processes. Run the CPU stages with `C:/Users/yuvar/anaconda3/python.exe research/fuse_sam_room.py`, `research/sam_motion_check.py`, and `research/package_sam_results.py`. `research/try_sam.py` contains all compatibility adapters. Hub checkpoint revisions and SHA-256 values are in `outputs/sam_trial/model_provenance.json`; installed packages are recorded in `research/sam_runtime_requirements.txt`.

`notebookgyu.sam.ipynb` validates and reviews the completed cached experiments and includes explicit recomputation commands. Its executed counterpart contains the maps and comparisons. The SAM 3.1 masks currently inform this short motion trial; they are not yet full-room 3D identity associations.

Validation: all five review notebook code cells completed without errors; the three room-view regression tests pass. The review asserts temporal coverage, nonempty masks, fusion/fallback accounting and finite object coordinates. `research/check_sam_attention.py` checks the broadcast adapter against the original attention equation. The notebook requires permission for a local Jupyter kernel; a sandboxed launch stalled before creating that kernel, while the permitted local execution completed.

## Research next steps and data needs

The plausible contribution remains uncertainty-aware fusion of proposal masks, temporal identity and iPhone export-depth geometry. The models themselves are established components. Proving novelty needs a literature comparison; proving better results needs independent annotations and ablations. Annotate held-out instance masks/IDs, include negatives, occlusions and controlled moving objects, and collect independent rooms. Export raw floating-point depth, confidence, exact intrinsics/registration, ARKit conventions and timestamps. More RGB from the same uncalibrated export will not fix the principal scale and reference-data bottlenecks. Start with the previously proposed 5–10 diverse rooms and repeat scans; that is a pilot plan, not a sufficient statistical sample by itself.

Primary sources: [official SAM repository](https://github.com/facebookresearch/sam3), [SAM 3.1 release notes](https://github.com/facebookresearch/sam3/blob/main/RELEASE_SAM3p1.md), [SAM paper](https://arxiv.org/abs/2511.16719), [Windows Triton implementation](https://github.com/woct0rdho/triton-windows).
