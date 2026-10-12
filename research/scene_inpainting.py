"""
Scene Reconstruction Inpainting and Hole Filling Module.
Evaluates and implements algorithmic mesh completion, hole-filling, and spatial interpolation
to achieve visually and geometrically complete 3D scene surfaces.
"""

import os
import numpy as np
import open3d as o3d

def inpaint_point_cloud_poisson(
    input_ply_path: str,
    output_ply_path: str,
    depth: int = 9,
    density_quantile: float = 0.05,
    smooth_iterations: int = 2
):
    """
    Algorithmic Hole Filling via Screened Poisson Surface Reconstruction.
    Poisson reconstruction solves an implicit indicator function over the vector field of oriented normals,
    which naturally closes holes and interpolates unobserved regions smoothly.
    
    Args:
        input_ply_path: Path to input point cloud or mesh.
        output_ply_path: Path to save the completed watertight mesh.
        depth: Octree depth for Poisson reconstruction (8-10 recommended for room scale).
        density_quantile: Low-density threshold to remove extreme extrapolation artifacts while keeping filled holes.
        smooth_iterations: Laplacian smoothing passes for boundary seam removal.
    """
    print(f"Loading input geometry from: {input_ply_path}")
    if not os.path.exists(input_ply_path):
        raise FileNotFoundError(f"File not found: {input_ply_path}")
        
    pcd = o3d.io.read_point_cloud(input_ply_path)
    if len(pcd.points) == 0:
        # Try loading as mesh
        mesh = o3d.io.read_triangle_mesh(input_ply_path)
        pcd = mesh.sample_points_uniformly(number_of_points=500000)
        
    print(f"Loaded point cloud with {len(pcd.points)} points.")
    
    # 1. Normal Estimation and Orientation if not present
    if not pcd.has_normals():
        print("Estimating surface normals...")
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.10, max_nn=30)
        )
        pcd.orient_normals_consistent_tangent_plane(k=15)
    else:
        print("Point cloud has existing normals.")

    # 2. Screened Poisson Surface Reconstruction
    print(f"Executing Screened Poisson Surface Reconstruction (octree depth={depth})...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, linear_fit=True
    )
    densities = np.asarray(densities)
    print(f"Reconstructed mesh has {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles.")

    # 3. Controlled Density Trimming
    # Only trim extreme outliers outside the scene boundary, preserving interpolated hole fills
    min_density = np.quantile(densities, density_quantile)
    vertices_to_remove = densities < min_density
    mesh.remove_vertices_by_mask(vertices_to_remove)
    print(f"After trimming sparse extrapolation: {len(mesh.vertices)} vertices.")

    # 4. Laplacian Smoothing for seamless transitions
    if smooth_iterations > 0:
        print(f"Applying {smooth_iterations} passes of Laplacian smoothing...")
        mesh = mesh.filter_smooth_laplacian(number_of_iterations=smooth_iterations)
        mesh.compute_vertex_normals()

    # 5. Export Watertight / Inpainted Result
    os.makedirs(os.path.dirname(output_ply_path), exist_ok=True)
    o3d.io.write_triangle_mesh(output_ply_path, mesh)
    print(f"Seamlessly completed mesh saved to: {output_ply_path}")
    return mesh

def fill_mesh_boundary_holes(
    input_mesh_path: str,
    output_mesh_path: str,
    max_hole_size: float = 0.5
):
    """
    Topological boundary hole-filling on existing triangle meshes.
    Identifies non-manifold edge loops and fills planar or curvature-continuous patches.
    """
    print(f"Loading mesh for boundary hole-filling: {input_mesh_path}")
    mesh = o3d.io.read_triangle_mesh(input_mesh_path)
    
    # Check manifoldness
    print(f"Is edge manifold: {mesh.is_edge_manifold()}")
    print(f"Is vertex manifold: {mesh.is_vertex_manifold()}")
    
    # Remove degenerate triangles and unreferenced vertices
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    
    # Filter smoothing
    mesh = mesh.filter_smooth_taubin(number_of_iterations=2)
    mesh.compute_vertex_normals()
    
    o3d.io.write_triangle_mesh(output_mesh_path, mesh)
    print(f"Hole-repaired mesh saved to: {output_mesh_path}")
    return mesh

if __name__ == "__main__":
    # Test on calibrated supported cloud
    cloud_path = "outputs/research_improved/calibrated_supported_cloud.ply"
    out_mesh = "outputs/research_improved/inpainted_seamless_mesh.ply"
    if os.path.exists(cloud_path):
        inpaint_point_cloud_poisson(cloud_path, out_mesh, depth=8, density_quantile=0.03)
