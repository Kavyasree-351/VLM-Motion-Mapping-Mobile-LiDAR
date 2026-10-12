# Research review and repaired mapping prototype

Review date: 8 October 2026. Status: DONE_WITH_CONCERNS.

The original output is a qualitative projection of inconsistently decoded depth, not a validated metric or visual-language reconstruction. The repaired result is a coherent room map with much better held-out geometric consistency. This is a research prototype, with a proposed contribution to test, not an established novel or state-of-the-art method.

## Evidence reviewed

All 80 original code cells and their saved textual outputs were inspected. The audit covers 28 generated artifacts, including every PLY's point count, finite coordinates and bounding box; all saved map images and metadata/CSV files; all 25 inline figures in a contact sheet; and the original 210-frame, 20-fps video, including a decoded midpoint. Six distributed RGB/depth pairs and detector crops were inspected visually. Audit evidence is in `original_artifact_audit.json`, `original_cell_outputs.txt`, `all_original_figures.png`, `input_contact_sheet.png`, and `detection_contact_sheet.png` alongside this report.

The dataset has 2,091 RGB/depth pairs, portrait RGB 720 x 960 and depth 192 x 256, and seven-column poses. The user identifies the source as iPhone LiDAR. The supplied depth arrays contain integer-like float32 values around 0–255; the export transform, original intrinsics, timestamps, and LiDAR confidence maps are absent. This does not resemble raw metric ARKit scene depth. Apple documents raw depth pixels in metres in [Explore ARKit 4](https://developer.apple.com/videos/play/wwdc2020/10611/?time=1272).

## Root causes and implemented fixes

| Issue | Evidence | Implemented fix |
|---|---|---|
| Camera axes mismatched | Original camera convention produces 20.81 px median epipolar error; ARKit-compatible axes and fitted focal give 0.298 px | Test four pose/axis hypotheses; use `R_ARKit @ diag(1,-1,-1)` for OpenCV rays |
| Guessed focal length | Original 750 at full RGB resolution is inconsistent with training correspondences | Fit scalar focal to robust epipolar residuals; recovered full-resolution focal ~687.57, fixed centre assumptions retained |
| Arbitrary depth conversion | Final notebook maps frame percentiles to 0.5–2.5, earlier cells map 255 to 5 metres | Recover per-frame affine depth using triangulated anchors and temporal regularization; compare inverse-depth alternative |
| Underconstrained fit can invert ordering | Initial unconstrained prototype produced negative slopes | Positive bounded decoder spans/endpoints; explicit synthetic regression test |
| Scale search rewards collapse | Cell 21's nearest-neighbour objective always favors its smallest tested scale | Validate against triangulation in pose translation units, plus correspondence-specific 3D agreement; no unanchored nearest-neighbour scale minimization |
| Duplicate, density-driven evidence | 419 frames in the final map; repeated pixels/nearby frames inflate counts | Select translation/rotation keyframes; one weighted contribution per view per voxel; require two cross-view agreements and two voxel views |
| Map labels confuse filled density with occupancy | Original projections mix floor/ceiling, followed by closing/opening/dilation in cell 64 | Ray-based free/occupied/unknown/ambiguous states; preserve unsupported space and display an explicit height slice |
| Repeated class error | Crop audit shows orange beanbag repeatedly labelled bed | CLIP region/text verification with cosine/margin rejection: 13 of 15 bed proposals accepted as beanbag; two unresolved |
| Object confidence ignores geometry | Original “reliable” label comes from observation count and mean detector confidence | Independent positions, inner-region depth statistics, local calibration residual weighting, and spatial-spread ellipses; position support separated from semantic truth |
| No actual language features | Original uses fixed YOLO labels and a “VLMaps-style” filename | Store normalized CLIP region embeddings and independent-view landmark embeddings; sparse text queries with unresolved outcome |
| Large redundant execution/memory | Multiple reconstruction cells concatenate 10M+ points; one original cloud has 5.44M vertices | Separate probe/calibration/mapping/semantics/query stages, reusable caches, sparse fit, early pixel sampling, vectorized voxel aggregation |

Original files and outputs are preserved. New outputs are isolated in `outputs/research_improved/`.

## Method and research hypothesis

**Candidate direction: calibration-conditioned, view-balanced language mapping for lossy mobile LiDAR exports.** The problem is an exported depth sequence with missing or damaged provenance, not a generic new SLAM system. Recover camera convention and depth decoding from RGB/pose agreement, then limit semantic evidence according to geometry quality and viewpoint diversity. Avoid issuing confident language-localized landmarks when the geometry or class evidence is weak.

For a pixel ray, world position is:

`p_world = R_ARKit · diag(1,-1,-1) · K^-1 · [u,v,1] · z_t + t`.

For normalized exported depth `r/255`, fit `z_t = a_t(r/255) + b_t` with positive bounded coefficients. Optimize Huber-weighted relative residuals against positive, sufficient-parallax triangulated anchors, plus a fixed temporal coefficient penalty. The alternative decoder represents inverse depth with positive near/far endpoints. Family choice uses training scores, not test scores.

The evidence budget combines cross-view agreement and local calibration residual spread, caps duplicate pixel evidence, and retains independent camera positions for semantic landmark pooling. The residual spread is an empirical quality signal, not a probability-calibrated uncertainty estimate. The current prototype uses local spread rather than a full posterior over calibration, poses, and semantics.

### What is already known

- Camera convention correction, triangulation, robust affine alignment, voxel fusion, and occupancy ray tracing are established techniques. These are engineering repairs.
- [Relative Pose Estimation through Affine Corrections of Monocular Depth Priors, CVPR 2025](https://arxiv.org/abs/2501.05446) already treats affine depth ambiguity geometrically.
- [MonoInstance, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_MonoInstance_Enhancing_Monocular_Priors_via_Multi-view_Instance_Alignment_for_Neural_CVPR_2025_paper.html) connects multi-view instance alignment and uncertain geometry.
- [On the Overconfidence Problem in Semantic 3D Mapping, ICRA 2024](https://motion.cs.illinois.edu/papers/ICRA2024_Marques_OverconfidenceSemanticMapping.pdf) identifies correlated views and fusion overconfidence. View balancing is not novel by itself.
- [Camera-Lidar Integration, 2020](https://arxiv.org/abs/2007.05490) already uses uncertainty and visibility for semantic sensor fusion.
- [SCOUT, 2026 workshop preprint](https://arxiv.org/abs/2606.06721) couples semantic uncertainty and scene coverage. Generic “uncertainty-aware scene graphs” would not be a defensible unique claim.
- [VLMaps](https://arxiv.org/abs/2210.05714) integrates spatial visual-language features. Our map contains sparse proposed-object CLIP features, not dense VLMaps or a complete navigation system.

The distinctive hypothesis to investigate is **explicitly coupling export-decoder uncertainty to the amount of independent language evidence that a map is allowed to accumulate**, including an abstention result. A literature search did not establish that this combination is unique. A defensible contribution needs controlled lossy-export experiments, a formal evidence rule and comparison with existing calibrated-fusion baselines. Do not claim that the large gains from correcting broken axes prove research novelty.

## Quantitative results

| Measurement | Original | Improved |
|---|---:|---:|
| Held-out median epipolar residual, 360 x 480 pixels | 20.807 px | 0.298 px |
| Held-out median absolute relative depth error | 31.53% | 3.76% |
| Held-out mean absolute relative depth error | 32.32% | 8.62% |
| Held-out depth samples within 10% of triangulated anchor | 21.0% | 80.8% |
| Median unfiltered correspondence-specific 3D disagreement | 0.403 | 0.079 |
| 90th-percentile unfiltered 3D disagreement | 1.048 | 0.960 |

Depth metrics use 2052 held-out samples, with 7852 training samples after rounded same-image keypoint duplicate exclusion. Every fifth 40-frame anchor pair is held out. Camera metrics use 2,190 matches; 3D disagreement uses the same 2,183 valid matches for all methods, before filtering. Camera convention/focal selection uses training anchors only. Pixel residuals are reported at half RGB resolution; multiply by two for full resolution.

Whole 200-frame temporal blocks are also excluded from depth fitting whenever either endpoint touches a withheld block: 2410 samples, median depth error 3.56% and mean 6.76%. Camera intrinsics are fixed from the separate camera experiment, so this is specifically a depth robustness check.

### Ablations and difficult cases

Global affine calibration alone gives 15.23% median depth error. The inverse-depth alternative gives 4.22%; affine depth was selected on training scores. Axis correction alone reduces camera residual to 1.037 px before focal fitting.

For unfiltered 3D agreement, camera repair without depth fitting gives median 0.121 and 90th percentile 0.435. The affine calibration improves the median but **does not beat that intermediate method in the tail**. Some poor correspondences, shallow parallax, depth boundaries, and weak anchors remain. The final map explicitly rejects unsupported points; it does not establish accuracy or completeness for rejected surfaces.

Pair-cluster bootstrap 95% intervals for median disagreement are [0.29298199518066564, 0.6793399581853629] for the original and [0.052487590577381, 0.13406112423469188] for the calibrated version. These quantify variability within this sequence, not cross-scene confidence.

The mapping stage uses 73 keyframes, accepts 758,875 cross-view-supported sampled points, and retains 91,884 voxels supported by multiple views. It runs in roughly 2.6 seconds on this machine excluding feature extraction, calibration, model loading, detections, text verification, plotting, and video export. It is not an end-to-end speedup benchmark. Fewer points are a storage/evidence decision, not an accuracy metric. Before/after figures use equal world limits and identical height slices.

## Semantic and query results

The three mapped label hypotheses are beanbag, armchair, and sofa. The beanbag position has six separated camera viewpoints. The armchair has three; the sofa remains spatially broad and tentative. CLIP similarities are not class probabilities; ellipses combine empirical spatial scatter and calibration spread, not calibrated confidence intervals. A surface-region centroid is not necessarily an object's true 3D centre.

Illustrative text queries find the orange beanbag and black leather sofa. A refrigerator query returns unresolved, rather than returning the nearest region unconditionally. These three examples are qualitative demonstrations, not a retrieval benchmark. Detection-proposal recall still limits which objects can be queried; doors/windows may be absent even though they are in the vocabulary.

## Remaining bottlenecks and publication requirements

1. Restore original metric float LiDAR, confidence maps, synchronized timestamps, orientation/display transform and per-frame intrinsics if available. Quantization/export normalization cannot be inverted exactly. Current distances share pose translation units, expected to be metres for an unscaled ARKit export, but independently unverified.
2. Collect multiple rooms and repeated trajectories. A single sequence cannot substantiate unique contribution, generalization, semantic accuracy or state-of-the-art performance.
3. Build controlled export variants from retained metric originals: 8-bit quantization, per-frame normalization, inverse-depth encoding, clipping/gamma, dropped intrinsics, pose noise, and missing depth confidence. This isolates recovery quality from engineering bugs.
4. Compare against a corrected conventional RGB-D/TSDF baseline, global scale, framewise affine calibration without evidence budgeting, view balancing without calibration weights, and calibrated semantic fusion such as GLFS. Reuse identical frames, proposals, grids, and compute budgets.
5. Evaluate geometry accuracy **and completeness** with reference meshes/depth, object localization and semantic labels with human annotations, and query precision/recall including absent and ambiguous objects. Fit true calibration probabilities before reporting ECE/Brier scores or coverage guarantees. Use scene-level splits and scene-level confidence intervals.
6. The wall/ceiling/floor alignment now looks coherent, but weak surfaces, transparent windows, shallow-parallax views, sofa centre drift, and semantic false negatives remain. The evidence map samples a band near camera height; it is not a full robot-footprint obstacle map or a validated navigation costmap.
7. Calibration is offline and uses future views. The replay overlays a final map on the trajectory; it does not claim causal online reconstruction. A future online variant needs delayed updates and a fair latency/accuracy comparison.

## Reproduce and inspect

Open `notebookgyu-cpu.research.ipynb` in the workspace root and run all cells. It uses the downloaded workspace dependencies/models, validates data, tests camera hypotheses, fits constrained depth, builds the map, verifies labels and queries, and runs regression tests. The executed copy saves all outputs.

Modules: `geometry_probe.py`, `improve_mapping.py`, `verify_semantics.py`, `query_map.py`, `test_mapping.py`. Numerical evidence: `outputs/research_improved/evaluation.json`; uncertainty/evidence arrays, calibration parameters, raw detections and verification results are also preserved. Figures: `before_after_height_slice.png`, `calibration_evaluation.png`, `evidence_map.png`, `semantic_evidence_map.png`. Final cloud: `calibrated_supported_cloud.ply`. Video: `repaired_room_replay.mp4`.

Nine regression tests pass: ARKit axis/world geometry, known affine calibration recovery, single-view evidence not masquerading as multiple views, duplicate-pixel invariance, and monotonic positive depth under contradictory anchors.

Durable learnings: quantized floating arrays are not metric depth; ARKit/OpenCV camera axes must be reconciled; unanchored nearest-neighbour scale selection can reward collapse; repeated semantic errors survive observation counting; weak framewise fits require monotonic bounds. These lessons are preserved here for future sessions.


## Latest research and VLMM extension (updated 8 October 2026)

| Work and current checked version | Relevance and implementation decision |
|---|---|
| [Vision-Language-Motion Maps v2, 11 August 2026](https://arxiv.org/abs/2607.16173v2), preprint | Object motion attributes with semantic priors and uncertainty. The revision corrects which endpoint covariance is rotated. Our CPU adaptation adds export-decoder residual spread to the motion noise model. |
| [Memory for Attention v2, 12 September 2026](https://arxiv.org/abs/2607.23797v2), preprint | Persistent history helps selective re-observation under some conditions. Its updated results caution against interpreting planner-cost gains as executed-navigation gains. We output re-observation hints, not a claimed optimal scheduler. |
| [SuperMap, 24 August 2026](https://arxiv.org/abs/2608.22896), RSS 2026 | Consistency-driven instance association, temporal existence/label updates and 4D scene memory. Our label/spatial association remains simpler and is explicitly flagged; full tracking is a necessary future baseline. |
| [ChronoGraph v2, 3 October 2026](https://arxiv.org/abs/2609.39665v2), preprint | Links actions and affordance parts to observed/anticipated state transitions. This room sequence supplies neither interactions nor action labels, so this is a comparison and future evaluation direction, not an implemented planning system. |

### What was added

`motion_map.py` samples pairs 20 frame indices apart. It computes DIS forward/backward optical flow, rejects poorly textured/invalid/mixed-depth regions, compensates the recorded camera motion, and removes a small residual rigid transform fitted to background correspondences. Large or weak background refinements cause abstention. Eroded bounding boxes are an approximation to instance masks; this is not a reproduction of the paper's RAFT/segmentation experiments.

World residuals are scored with propagated pixel and axial-depth covariance, a conservative empirical export-decoder residual term, and an assumed pose floor. The covariance rotation follows the corrected endpoint order. Range-noise coefficients, assumed pose floor, thresholds and property scores are heuristic; they are not an iPhone sensor calibration. Shared decoder error creates correlated pixels, so the implementation does not multiply pixel likelihoods into an object posterior. Motion history is aggregated robustly across eligible pairs, with coherence and supporting-pair checks.

Every mapped element stores an embedding, position/spread, measured episode-motion score, reliability score, semantic movability score, motion class, history support and last-seen frame. The semantic prior compares CLIP movable-furniture versus fixed-structure prompts, not measured physical motion. The query router handles a limited intent grammar, catches explicit negation before positive motion phrases, rejects unsupported conjunctions and refuses to answer present-motion queries from archived frames. It is not a general language parser.

In this sequence the armchair, beanbag and sofa are `movable_not_observed_moving`. The implementation does not invent an actual object movement to make the demonstration look dynamic. One isolated above-threshold pair can remain an unconfirmed anomaly rather than changing the persistent state. Re-observation hints rank spatially broad or unresolved evidence; no acquisition loop, volatility-rate estimate or scheduling-benefit claim is made.

The mean per-region fraction of flagged pixels is 16.63% with sensor-only assumed covariance and 4.29% when decoder uncertainty is included. These are diagnostic flag fractions, **not false-positive rates**, because motion ground truth is absent. Suppressing flags can also suppress real movers; the synthetic axial-noise/lateral-motion test separates those cases, but is not a real benchmark. The original geometric holdout gains remain separate from this motion experiment.

Outputs: `vlmm_elements.json`, `motion_observations.csv`, `vlmm_queries.json`, `vlmm_summary.json`, and `vlmm_motion_attributes.png`. The notebook now includes the extension and all nine regression tests.

### Revised novelty direction

Generic visual-language-motion mapping is already introduced by the checked papers; merely adding that name or schema is not unique. The narrower candidate is **export-calibration-aware motion grounding**: distinguish shared mobile-depth decoder drift from actual object displacement, expose unresolved cases to language queries, and spend re-observation effort on resolving that ambiguity. Our implementation is a first empirical prototype of that idea, not a verified new method.

A credible next experiment needs retained metric LiDAR and known object motion. Apply controlled quantization/normalization and pose perturbations, hold out scenes, compare sensor-only VLMM-style covariance with decoder-aware covariance and corrected conventional RGB-D baselines, and report motion AP/recall, static false flags, state-query precision/recall, calibration, and retained-geometry completeness. Include slow axial motion: it is the hardest case because a wrong depth scale can mimic or hide it. Use fresh timestamps before estimating velocity. Compare re-observation against uniform and relevance-weighted recency policies before claiming a memory advantage.
