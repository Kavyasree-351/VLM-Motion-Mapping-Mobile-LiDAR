# Further research improvements

Search date: 8 October 2026. Primary paper pages and official repositories were checked. This is a targeted search, not an exhaustive literature review. These are proposed experiments; none has been measured on this room yet.

## Recommended order

| Priority | Change | Current bottleneck | Experiment and acceptance evidence |
|---|---|---|---|
| 1 | Class-agnostic persistent object association | `room_views.fuse_objects` groups by semantic label before spatial clustering. Label flips can split one physical object; nearby same-class objects can merge. | Associate by projected mask overlap, supported 3D overlap, appearance and visibility, with stable IDs and alternative labels. Annotate same-object pairs across disjoint views; report association precision/recall, ID switches, false merges and semantic accuracy. |
| 2 | Mask-based motion on the upgraded inventory | Existing motion output still uses three old object regions and eroded boxes. The new nine-region map does not yet have equivalent motion histories. | Connect track IDs and masks to background-compensated flow; keep semantic movability separate from observed motion. Measure static false flags and known-motion recall, especially slow axial motion. Frame indices alone cannot establish velocity. |
| 3 | Adaptive support and depth completion | Fixed distance gates reject uncertain points and leave holes; nearest-cloud proximity alone cannot distinguish foreground from an occluded surface. | Add depth reprojection, occlusion checks and residual uncertainty. Compare current strict fusion with adaptive fusion using held-out depth/render consistency and retained coverage together. Only then test learned depth priors in unsupported areas, visibly separated from measured geometry. |
| 4 | Wall/opening graph with uncertainty | The convex envelope can bridge real recesses or unobserved openings. Lines are not a complete architectural topology. | Fit gravity-aligned planes and intersect observed wall segments; assign doors/windows to compatible wall planes. Mark unsupported closure as unknown. Evaluate corner positions, wall/opening precision and topology against manual measurements. Do not force every room to be rectangular. |
| 5 | Dense-plus-instance language features | Sparse object crops cannot answer queries about surfaces or entities missed by the detector. | Add a bounded dense feature layer and fuse it with instance features. Evaluate absent-object abstention, object recall and surface queries; track memory/runtime separately. |
| 6 | Appearance-aware rendering refinement | TSDF RGB rendering has gaps and inconsistent appearance. More vertices alone do not establish better geometry. | Evaluate uncertainty-aware Gaussian rendering as a separate appearance layer, retaining the measured mesh. Hold out views and score render quality alongside depth accuracy and completeness. |

## Relevant papers and released models

**OVI-MAP (27 March 2026).** Separates class-agnostic instance reconstruction from semantic inference and selects a small set of informative views. This directly motivates priority 1 and better crop selection. Its reported benchmark results cannot be transferred to this room. [Paper](https://arxiv.org/abs/2603.26541).

**SuperMap (24 August 2026).** Uses 3D-aware association/reactivation and existence/label updates for changing scenes. It is a relevant baseline for persistent IDs and visibility-aware state updates. A hidden object should not be marked absent just because it was not detected. [Paper](https://arxiv.org/abs/2608.22896).

**SAM 3 / SAM 3.1.** SAM 3 combines text/exemplar segmentation with video tracking. The 27 March 2026 SAM 3.1 release adds shared-memory Object Multiplex tracking; its release notes show mixed video benchmark changes, so newer is not automatically more accurate. Test it as a mask/tracking alternative to YOLOE on identical annotated frames. The official setup lists a CUDA-compatible GPU and checkpoint-access approval; the current mapping runtime is CPU, so it is not a drop-in upgrade here. [Paper](https://arxiv.org/abs/2511.16719), [release notes](https://github.com/facebookresearch/sam3/blob/main/RELEASE_SAM3p1.md), [official setup](https://github.com/facebookresearch/sam3).

**MapAnything (v3, 23 January 2026; 3DV 2026).** Accepts images with optional intrinsics, poses and depth for reconstruction and completion. It is a strong candidate for conditioning on trusted ARKit geometry. Its official example includes a CPU device fallback, but this does not establish practical CPU speed. Quantized exported depth must be calibrated before supplying it as metric input. [Paper](https://arxiv.org/abs/2509.13414v3), [code](https://github.com/facebookresearch/map-anything).

**Depth Anything 3 (November 2025; ICLR 2026 paper).** Predicts consistent geometry from multiple views with or without known poses. Compare it with MapAnything as an independent geometric prior; alignment to the pose frame must use training anchors, and evaluation must use separate views. A model's metric prediction is not independent scale ground truth. [Paper](https://arxiv.org/abs/2511.10647), [code](https://github.com/ByteDance-Seed/Depth-Anything-3).

**FUS3DMaps (5 May 2026).** Combines dense voxel language features and instance features, including a bounded sliding-window variant. This addresses detector-limited query coverage and provides prior art against claiming dense/instance fusion as our invention. [Paper](https://arxiv.org/abs/2605.03669).

**SceneAligner (21 May 2026) and Z-FLoc (3 June 2026).** Use reconstructed bird's-eye evidence or geometric primitives for localization against a supplied floorplan. They motivate structural representations but do not solve generating a verified floorplan from this scan without a reference plan. [SceneAligner](https://arxiv.org/abs/2605.22581), [Z-FLoc](https://arxiv.org/abs/2606.04788).

**VarSplat (March 2026).** Learns per-splat appearance uncertainty for RGB-D Gaussian SLAM. It is a relevant appearance/reliability baseline, not evidence that a Gaussian renderer will recover missing measured surfaces. [Paper](https://arxiv.org/abs/2603.09673).

## Research contribution to test

The strongest candidate remains distinguishing **shared export-depth calibration drift from real object motion**, now coupled to class-agnostic object identity. A track should be allowed to retain uncertain labels and abstain about motion when decoder uncertainty explains the residual. This is a hypothesis, not an established unique contribution: tracking, uncertainty fusion and motion mapping already have extensive prior art.

Use controlled experiments that perturb retained metric depth into quantized, normalized and clipped exports, with known object motion and held-out rooms. Compare corrected conventional RGB-D fusion, decoder-unaware tracking, view-balanced tracking and decoder-aware tracking. Report association errors, motion precision/recall, static false flags, slow axial-motion recall, semantic query accuracy, accuracy/completeness tradeoffs and full runtime. A lower motion-flag count can hide real movers, so it cannot serve as the improvement metric alone.

The next implementation should start with priority 1 and then priority 2. They address concrete weaknesses in the current code and can be tested without replacing the successful room reconstruction. Heavy model experiments should follow an annotated baseline and hardware feasibility check.
