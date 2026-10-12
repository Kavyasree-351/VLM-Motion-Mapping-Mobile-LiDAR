# VLM-maps research hand-off — 8 October 2026

## Start here

1. Open `notebookgyu.sam.executed.ipynb` for the latest completed SAM comparisons, tracking checks, motion diagnostics and room views. All five code cells completed without errors. `notebookgyu.sam.ipynb` is its editable source.
2. Open `outputs/sam_room/room_3d_interactive.html` in a browser. It is self-contained and supports cutaway, full-room and object-only views. `room_floorplan.png` / `.svg` contain the top view; `room_3d_cutaway.png` contains the static 3D view.
3. Read `research/SAM_UPGRADE_REPORT.md`, then the earlier room and dataset trial reports for research interpretation and limitations.

Extract the entire ZIP first, preserving directory structure. HTML can be viewed locally; notebook links resolve relative to the project root. Some notebook code still refers to the original Windows workspace `D:/VLM-maps`: update those paths if rerunning elsewhere. Stored notebook outputs can be reviewed without installing the models. The original notebooks and their historical errors are retained for provenance, rather than silently removed.

## Contents

- `outputs/`: the COMPLETE results tree, including all PLY files, meshes, clouds, SVG/PNG/JPEG figures, standalone interactive HTML, CSV/JSON metrics, calibration arrays, embeddings, masks, temporal tracks and comparison frames.
- Root notebooks: editable versions AND executed versions, with stored outputs preserved. Includes original baselines, calibrated research mapping, YOLOE room upgrade, further dataset trials and SAM upgrade.
- `research/`: implementation and validation scripts, all Markdown reports, run logs, diagnostics, experiment artifacts and pinned SAM package requirements. Python bytecode caches are excluded.
- Root README, Python runner/repair scripts, execution logs and diagnostics.
- `.models/yolo/`: scene-adapted visual/text YOLOE weights and visual prompt embeddings. Adaptation provenance is in `outputs/room_refined/visual_prompt_provenance.json`. These are same-room visual references; they are not trained or tested on independent rooms.
- `Ultralytics/`: project-side configuration files, where present.
- `HANDOFF_MANIFEST.json`: every packaged file's byte count and SHA-256, plus a notebook execution/output inventory. The manifest does not hash itself.

## Result folders and PLY files

- `sam_room`: latest gated SAM-refined object surfaces and room views. `objects_rgb.ply` contains refined object evidence. `room_cutaway_aligned.ply` is the aligned display cutaway. Full measured geometry is reused from `room_refined/room_textured_mesh.ply`.
- `sam_trial`: SAM 3 image comparisons, hybrid masks, two SAM 3.1 sessions, consecutive source-frame mapping, motion comparisons, checkpoint provenance and integrity validation.
- `room_refined`: earlier YOLOE room results, full textured mesh, RGB object PLY, cutaway PLY and detector comparisons.
- `research_improved`: calibrated geometry, supported clouds, calibration/semantic/motion evidence and prior research maps.
- `next_trials`: geometry-first identity association, DA3 depth comparisons, dense language patches and mask-motion experiments.
- `reconstruction_outputs` and `final_vlmaps_outputs`: original baseline artifacts, retained for comparison.

PLY coordinates follow the producing pipeline. Files with `aligned` in their name use the room display coordinates; other full-room PLYs preserve world coordinates. Scale follows supplied ARKit poses and inferred export-depth calibration, and has not been independently verified as metric. Keep `room_geometry.json` with the aligned views.

## Current findings

Latest static candidate: YOLOE-26 visual object proposals plus SAM 3 box-conditioned mask refinement, guarded by registered-cloud support, center shift and area checks. Of 243 candidate observations, 186 refinements passed and 57 retained their YOLOE surfaces. Nine supported object regions remain. These counts describe processing, not detection accuracy.

SAM 3.1 tracked sofa and beanbag separately through 90 consecutive frames, using a broad furniture prompt for the sofa and a specific beanbag prompt for the beanbag. IDs are session-local. This is a short static-scene check, not a full-room identity/occlusion benchmark. Tracked masks gave 23.8% more usable motion correspondences than boxes on four matched object observations; no moving object was established. The broad furniture prompt missed the beanbag. Text SAM produced duplicate vents and fragmented shelf/table masks. The hybrid failed on a late door view, and its geometry gate rejected that refinement.

Known reconstruction gaps, unverified depth export scale and missing reference annotations remain. The candidate research contribution is uncertainty-aware coupling of calibrated iPhone geometry, semantic masks and temporal/motion evidence. The underlying model components are established methods. Novelty and accuracy improvements still require literature comparison, independent rooms, annotations and ablations.

## Rerun environments

Original CPU mapping: `C:/Users/yuvar/anaconda3/python.exe`, with project dependencies in `.packages`. SAM GPU stages: `.venv-sam31/Scripts/python.exe` (Python 3.12.13, PyTorch 2.10.0+cu128, torchvision 0.25.0+cu128). Exact installed SAM versions: `research/sam_runtime_requirements.txt`. GPU used: RTX 4080 Laptop, 12 GB. The trial includes Windows attention and session compatibility adapters and reduces frame batches to fit this GPU.

GPU commands, from the extracted project root:

```powershell
& .venv-sam31/Scripts/python.exe research/try_sam.py image
& .venv-sam31/Scripts/python.exe research/try_sam.py hybrid
& .venv-sam31/Scripts/python.exe research/try_sam.py video
& .venv-sam31/Scripts/python.exe research/try_sam.py video --prompt beanbag
```

CPU stages use the original mapping environment: `research/fuse_sam_room.py`, `research/sam_motion_check.py`, then `research/package_sam_results.py`. The last script rebuilds and executes the review notebook. Earlier pipelines are documented in their reports. Preserve caches before deliberately recomputing them.

## External prerequisites not included

The raw dataset and its source `archive.zip`, downloaded base-model checkpoints, `.packages`, `.runtime`, `.venv-sam31`, general caches and credentials are excluded. They are not required for reviewing the delivered outputs. To rerun, restore the dataset under `dataset/content/vlmaps_dataset/my_custom_scene`, recreate the relevant environments and obtain the referenced models. SAM checkpoints require approved Hugging Face access and local authentication; credentials are not portable project material. SAM model revisions and checksums are in `outputs/sam_trial/model_provenance.json`. Refer to the official upstream model repositories for licenses and acquisition.

Additional data priorities: raw floating-point LiDAR depth and confidence, exact intrinsics/registration/pose conventions and timestamps; independent diverse rooms and repeat scans; held-out masks/instance IDs; controlled moving objects and occlusions; independent measurements for geometry evaluation. See `research/DATASET_TRIAL_REPORT.md` for the proposed pilot collection plan.

## Archive validation

The builder includes all files under `outputs/` and all root notebooks without altering their outputs, checks the ZIP CRCs and verifies the archived file inventory. `handoff/<archive-name>.validation.json` records counts, all packaged PLY paths, notebook status, compressed size and SHA-256. A `.sha256` sidecar supports transfer integrity checking. Research experiment checks remain in their original logs and reports; archive validation does not establish model accuracy.
