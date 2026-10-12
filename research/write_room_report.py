from pathlib import Path
text='''# Room detection and reconstruction upgrade

The chosen detector is **YOLOE-26 large instance segmentation**, conditioned on six inspected room exemplars. A newer closed-category YOLO26 medium segmentation model was also tested, but still confused furniture classes. The original YOLO11 nano remains a baseline; the new notebook uses YOLOE-26.

## Results and evidence

- Four detectors were run on the same 12 frames at image size 960, confidence 0.25 and CPU execution. See `outputs/room_refined/detector_comparison_contact.png` and `detector_comparison.csv`. The visual-prompt model visibly improves beanbag, storage table, window, door and armchair detection. This is a qualitative comparison; detection count is not accuracy and no mAP claim is made.
- Visual examples use frames 50, 400, 1000 and 1600, disjoint from the audit frames. There is no gradient training. These results demonstrate adaptation within this room, not generalization to unseen rooms. Boxes and provenance are recorded in `visual_prompt_provenance.json`.
- Segmentation masks replace box interiors for depth localization. Eroded boundaries, registered-cloud support, independent camera observations and two-view voxel support reduce background contamination and duplicate objects. CLIP provides an auxiliary review, not ground-truth verification.
- 107 inference frames yield nine supported regions: sofa, armchair, beanbag, storage table, door, picture, ceiling vent and two window regions. Model class labels remain hypotheses, including the storage-table label and the distinction between separate window instances.
- The top view aligns to measured wall directions, removes duplicate line segments, shows observed wall segments and an explicitly dashed inferred room envelope. Furniture footprints represent observed surfaces; wall features use thin display strips and the vent is marked as overhead.
- The textured reconstruction contains **128,254 vertices and 247,387 triangles**. The standalone HTML provides cutaway, complete room and object-only views. RGB colors come from measured frames; missing surfaces are not synthesized.

## Bottlenecks fixed

1. Closed-category indoor class mismatch: use open-vocabulary segmentation plus scene visual references rather than assuming a newer closed detector solves the vocabulary problem.
2. Box background leaking into 3D: use valid eroded mask depth with multi-view support and calibration-spread gating.
3. Duplicated wall lines and cluttered perspective: consolidate measured segments and create aligned, labelled plans plus a wall-aware cutaway.
4. Empty legacy Open3D TSDF mesh: reproduce the failure with a known metric plane and switch to the tensor voxel-block backend. A reported Open3D issue describes a depth-scale activation error. Empty frames are skipped and empty mesh exports fail explicitly.

## Validation and reproducibility

All **12 regression tests passed**, including a synthetic plane producing a colored mesh at the expected depth, safe empty-frame handling, and display-coordinate distance preservation. Prior camera, calibration and motion tests also pass.

Use the original Anaconda Python explicitly: `C:/Users/yuvar/anaconda3/python.exe`. The unrelated system Python on PATH does not have the same dependencies. Local packages and weights remain in `.packages` and `.models`.

Run `research/room_upgrade.py --configure` to recreate visual embeddings, `research/room_upgrade.py --benchmark` for the comparison, and `research/room_views.py` for the outputs. The completed notebook is `notebookgyu-cpu.room26.executed.ipynb`; its editable source is `notebookgyu-cpu.room26.ipynb`. Cached inference and mesh outputs make review fast. Preserve caches before deliberately recomputing them.

## Research interpretation

The candidate contribution is **export-decoder uncertainty-aware, motion-aware semantic mapping**: couple inferred iPhone depth calibration uncertainty with mask support and the existing visual-language motion evidence. YOLOE prompting and TSDF reconstruction are established components, not claimed inventions. The new object inventory is not yet wired into all earlier motion queries: those outputs retain the original three-object inventory. They must be rebuilt and evaluated before claiming a nine-object motion map.

A publishable improvement needs instance annotations on held-out frames and independent rooms; mask IoU and mAP; 3D localization/shape error against reference measurements; and ablations of calibration, visual prompting, mask fusion, semantic review and motion conditioning. Evaluate on reference-disjoint views and held-out rooms, with failure cases and CPU runtime. Current geometry consistency is a proxy, not ground truth. Exported depth normalization and absolute scale are still inferred, and reconstruction has real coverage holes.

## Primary sources

- [Ultralytics YOLO26 documentation](https://docs.ultralytics.com/models/yolo26/)
- [Ultralytics YOLOE documentation](https://docs.ultralytics.com/models/yoloe/)
- [Open3D legacy TSDF depth-scale issue #7592](https://github.com/isl-org/Open3D/issues/7592)
- Latest motion and mapping paper discussion is recorded in `research/RESEARCH_REVIEW.md`. The VLMM-inspired module is an approximation, not a reproduced VLMM model.
'''
Path('research/ROOM_UPGRADE_REPORT.md').write_text(text,encoding='utf-8')
Path('README.md').write_text('''# iPhone room mapping

Start with [the upgraded YOLOE-26 notebook](notebookgyu-cpu.room26.ipynb), or inspect its executed counterpart. Earlier notebooks are preserved baselines.

Open `outputs/room_refined/room_3d_interactive.html` for the standalone interactive room. The floor plan and textured cutaway are in the same folder. Read [the upgrade report](research/ROOM_UPGRADE_REPORT.md) for comparisons, validation and research limits.

Runtime: `C:/Users/yuvar/anaconda3/python.exe`; local dependencies are in `.packages`. Run `research/room_views.py` to regenerate views from cached inference and calibrated geometry. Run tests with `-m unittest discover -s research -p "test*.py"`.
''',encoding='utf-8')
