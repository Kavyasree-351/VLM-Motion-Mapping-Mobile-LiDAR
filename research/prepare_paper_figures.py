import os
import shutil
from PIL import Image

def prepare_figures():
    output_fig_dir = "paper/figures"
    os.makedirs(output_fig_dir, exist_ok=True)
    
    figures_map = {
        "outputs/research_improved/before_after_geometry.png": "fig_geometry_calibration.png",
        "outputs/research_improved/before_after_height_slice.png": "fig_height_slice.png",
        "outputs/research_improved/calibration_evaluation.png": "fig_calibration_eval.png",
        "outputs/sam_trial/sam_comparison.png": "fig_sam_comparison.png",
        "outputs/room_refined/detector_comparison_contact.png": "fig_detector_comparison.png",
        "outputs/next_trials/da3_depth_comparison.png": "fig_da3_depth.png",
        "outputs/sam_room/room_3d_cutaway.png": "fig_room_cutaway_3d.png",
        "outputs/sam_room/room_floorplan.png": "fig_room_floorplan.png",
        "outputs/research_improved/semantic_evidence_map.png": "fig_semantic_map.png",
        "outputs/research_improved/evidence_map.png": "fig_evidence_map.png",
        "outputs/sam_trial/sam31_000450.jpg": "fig_sam31_track_sample.png",
        "outputs/final_vlmaps_outputs/qualitative_topdown_final.png": "fig_baseline_topdown.png"
    }
    
    processed = []
    for src, dst_name in figures_map.items():
        if not os.path.exists(src):
            print(f"Warning: {src} not found, skipping.")
            continue
        
        dst_path = os.path.join(output_fig_dir, dst_name)
        with Image.open(src) as im:
            # Convert to RGB if needed
            if im.mode not in ("RGB", "RGBA"):
                im = im.convert("RGB")
            # Save with 300 DPI metadata
            im.save(dst_path, dpi=(300, 300))
            w, h = im.size
            print(f"Exported {dst_name}: {w}x{h} px @ 300 DPI")
            processed.append((dst_name, w, h))
            
    print(f"\nSuccessfully prepared {len(processed)} publication figures in {output_fig_dir} at 300 DPI.")

if __name__ == "__main__":
    prepare_figures()
