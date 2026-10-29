#!/usr/bin/env python3
"""
Procedural 3D Comic Pumpkin Generator
Author: Senior 3D Technical Artist & Computational Geometry Specialist
Description:
    Generates two 50mm x 50mm x 50mm comic-style 3D models in STL format:
    1. comic_pumpkin_50mm.stl      - Intact comic pumpkin with chunky lobes and expressive cartoon stem
    2. comic_jackolantern_50mm.stl - Carved comic Jack-o'-Lantern with hollow interior & tea-light port
"""

import sys
import os
import numpy as np
from scipy import ndimage
from skimage import measure
import trimesh
from matplotlib.path import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def build_sdf_grid(res_xy=190, res_z=190):
    """Constructs the spatial sampling volume in millimeters."""
    xs = np.linspace(-32.0, 32.0, res_xy)
    ys = np.linspace(-32.0, 32.0, res_xy)
    zs = np.linspace(-22.0, 42.0, res_z)
    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]
    dz = zs[1] - zs[0]
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    return xs, ys, zs, (dx, dy, dz), X, Y, Z

def compute_body_and_stem_sdf(X, Y, Z):
    """
    Computes Signed Distance Field for the comic pumpkin body and stem.
    Features:
    - Plump, comic aspect ratio (~68% body height, 32% stem height)
    - 8 bulbous comic lobes with sharp furrow crevices
    - Top calyx and basal polar dimples
    - Jaunty cartoon curling stem with 5 fluted ridges
    - Smooth polynomial minimum union (smin)
    - Flat ground cutter for rock-solid 3D printing bed adhesion
    """
    # 1. Body SDF (Plump, squashed comic proportions)
    rx = 23.5
    ry = 23.5
    rz = 19.0
    
    R_xy = np.sqrt(X**2 + Y**2)
    Theta = np.arctan2(Y, X)
    
    # Comic Ribs: 8 lobes with harmonic punch (cos(8θ) + 0.25 cos(16θ))
    rib_factor = np.cos(8.0 * Theta) + 0.25 * np.cos(16.0 * Theta)
    norm_z = np.clip(Z / rz, -1.2, 1.2)
    rib_envelope = np.maximum(0.0, 1.0 - norm_z**2)**0.65
    r_target = rx * (1.0 + 0.16 * rib_factor * rib_envelope)
    
    # Top & bottom polar depressions (calyx & basal dimple)
    top_dimple = 6.5 * np.exp(-(R_xy / 8.0)**2) * np.clip(Z / 11.0, 0.0, 1.0)
    bot_dimple = 4.5 * np.exp(-(R_xy / 9.0)**2) * np.clip(-Z / 11.0, 0.0, 1.0)
    z_deformed = Z + top_dimple - bot_dimple
    
    body_val = (R_xy / r_target)**2 + (z_deformed / rz)**2 - 1.0
    d_body = body_val * 14.0
    
    # 2. Comic Stem SDF
    z_stem_base = 12.0
    z_stem_top = 37.8
    t_stem = np.clip((Z - z_stem_base) / (z_stem_top - z_stem_base), 0.0, 1.0)
    
    # Disney / Burton style jaunty cartoon curl: leans to right and slightly back, then curls
    cx = 8.5 * (t_stem**1.3) * np.sin(t_stem * np.pi * 0.78)
    cy = -3.2 * t_stem + 2.8 * (t_stem**2)
    
    stem_dx = X - cx
    stem_dy = Y - cy
    r_stem_xy = np.sqrt(stem_dx**2 + stem_dy**2)
    theta_stem = np.arctan2(stem_dy, stem_dx)
    
    # Chunky base (5.2mm radius), tapering to 2.4mm, slight cartoon flare at cut tip
    base_stem_r = 5.2 * (1.0 - 0.58 * t_stem) + 1.4 * (t_stem**3)
    stem_flutes = 1.0 + 0.20 * np.cos(5.0 * (theta_stem - 1.8 * t_stem))
    target_stem_r = base_stem_r * stem_flutes
    
    d_stem_cyl = r_stem_xy - target_stem_r
    d_stem_top = Z - z_stem_top
    d_stem_bot = z_stem_base - Z - 2.5
    d_stem = np.maximum(d_stem_cyl, np.maximum(d_stem_top, d_stem_bot))
    
    # Smooth union (smin) of body and stem
    k = 2.5
    h = np.clip(0.5 + 0.5 * (d_stem - d_body) / k, 0.0, 1.0)
    d_solid = d_stem * (1.0 - h) + d_body * h - k * h * (1.0 - h)
    
    # Flat base cutter at Z = -17.5 for build plate stability
    z_base_cut = -17.5
    d_solid = np.maximum(d_solid, z_base_cut - Z)
    
    return d_solid, d_body, z_base_cut

def compute_jackolantern_void_sdf(xs, zs, dx, dz, X, Y, Z, d_body, z_base_cut):
    """
    Computes the SDF of internal void (chamber, bottom port, and comic face cutouts).
    """
    # 1. 2D Comic Face Polygons on (X, Z) plane (adjusted for plump body equator Z~0)
    # Left & Right angled mischievous comic eyes
    eye_l_pts = np.array([[-12.0, 9.5], [-4.5, 5.5], [-8.5, 2.5]])
    eye_r_pts = np.array([[12.0, 9.5], [8.5, 2.5], [4.5, 5.5]])
    nose_pts = np.array([[-3.2, 2.0], [3.2, 2.0], [0.0, -2.5]])
    
    # Wide comic grin with chunky cutout teeth (2 upper, 1 lower)
    mouth_pts = np.array([
        [-15.5, -1.0],  # Left cheek corner
        [-10.0, -3.0],
        [-8.0, -3.0],
        [-8.0, -7.0],   # Upper tooth 1 down
        [-4.5, -7.0],
        [-4.5, -3.5],
        [0.0, -3.8],    # Upper lip center
        [4.5, -3.5],
        [4.5, -7.0],    # Upper tooth 2 down
        [8.0, -7.0],
        [8.0, -3.0],
        [10.0, -3.0],
        [15.5, -1.0],   # Right cheek corner
        [11.5, -8.5],   # Lower lip right
        [2.8, -8.5],
        [2.8, -5.2],    # Bottom center tooth up
        [-2.8, -5.2],
        [-2.8, -8.5],
        [-11.5, -8.5]   # Lower lip left back to start
    ])
    
    # Generate exact 2D signed distance field for face
    X2, Z2 = np.meshgrid(xs, zs, indexing='ij')
    pts_2d = np.column_stack([X2.ravel(), Z2.ravel()])
    
    mask_face_2d = (
        Path(eye_l_pts).contains_points(pts_2d) |
        Path(eye_r_pts).contains_points(pts_2d) |
        Path(nose_pts).contains_points(pts_2d) |
        Path(mouth_pts).contains_points(pts_2d)
    ).reshape(X2.shape)
    
    d_face_out = ndimage.distance_transform_edt(~mask_face_2d, sampling=(dx, dz))
    d_face_in = ndimage.distance_transform_edt(mask_face_2d, sampling=(dx, dz))
    sdf_face_2d = d_face_out - d_face_in  # in mm
    
    # Broadcast across Y
    sdf_face_3d = sdf_face_2d[:, np.newaxis, :]
    
    # Face cutter penetrates front wall: Y in [-32, -4]
    d_face_cutter = np.maximum(sdf_face_3d, np.maximum(-32.0 - Y, Y - (-4.0)))
    
    # 2. Hollow Inner Chamber (3.0mm wall thickness)
    wall_th = 3.0
    d_inner = d_body + wall_th
    # Constrain chamber inside lower crown & above floor
    d_inner = np.maximum(d_inner, Z - 11.5)
    d_inner = np.maximum(d_inner, (z_base_cut + wall_th) - Z)
    
    # 3. Bottom tea light port (diameter 22mm = radius 11mm)
    # Extends only through the base (Z between z_base_cut - 2 and -10.0)
    R_xy = np.sqrt(X**2 + Y**2)
    r_hole = 11.0
    d_hole = np.maximum(R_xy - r_hole, np.maximum(Z - (-10.0), (z_base_cut - 2.0) - Z))
    
    # Combine void components: inner chamber, bottom port, face cutter
    d_void = np.minimum(d_inner, d_hole)
    d_void = np.minimum(d_void, d_face_cutter)
    
    return d_void

def extract_and_normalize_mesh(sdf_grid, xs, ys, zs, spacing, target_dim=50.0):
    """
    Extracts isosurface, applies Laplacian smoothing, and scales strictly to target_dim (50mm).
    """
    dx, dy, dz = spacing
    verts, faces, normals, _ = measure.marching_cubes(sdf_grid, level=0.0, spacing=(dx, dy, dz))
    
    # Shift vertices to coordinate origin
    verts[:, 0] += xs[0]
    verts[:, 1] += ys[0]
    verts[:, 2] += zs[0]
    
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    
    # Smooth mesh slightly to ensure organic cartoon curvature
    trimesh.smoothing.filter_humphrey(mesh, alpha=0.1, beta=0.5, iterations=2)
    
    # Normalize bounding box:
    # 1. Scale uniformly so maximum bounding extent is exactly target_dim (50.0mm)
    max_extent = np.max(mesh.extents)
    scale = target_dim / max_extent
    mesh.apply_scale(scale)
    
    # 2. Center XY at origin (0, 0) and place base on Z = 0
    bounds = mesh.bounds
    center_xy = (bounds[0][:2] + bounds[1][:2]) / 2.0
    mesh.vertices[:, 0] -= center_xy[0]
    mesh.vertices[:, 1] -= center_xy[1]
    mesh.vertices[:, 2] -= bounds[0][2]  # Z min = 0
    
    # Ensure all normals are strictly outward
    trimesh.repair.fix_normals(mesh)
    trimesh.repair.fix_inversion(mesh)
    
    return mesh

def render_preview_sheet(mesh_intact, mesh_jack, output_png):
    """Renders a high-fidelity Z-buffer multi-view inspection sheet of both models."""
    from PIL import Image

    def render_view(mesh, width=450, height=450, yaw_deg=0.0, pitch_deg=10.0):
        yaw = np.radians(yaw_deg)
        pitch = np.radians(pitch_deg)
        
        R_yaw = np.array([
            [np.cos(yaw), np.sin(yaw), 0],
            [-np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        R_pitch = np.array([
            [1, 0, 0],
            [0, np.cos(pitch), np.sin(pitch)],
            [0, -np.sin(pitch), np.cos(pitch)]
        ])
        
        verts = mesh.vertices.copy()
        center = np.mean(verts, axis=0)
        verts -= center
        v_rot = verts @ R_yaw.T @ R_pitch.T
        
        extent = 54.0
        scale = (min(width, height) * 0.82) / extent
        
        u = (v_rot[:, 0] * scale + width / 2.0).astype(np.float32)
        v = (-v_rot[:, 2] * scale + height / 2.0).astype(np.float32)
        z = -v_rot[:, 1]
        
        normals = mesh.face_normals @ R_yaw.T @ R_pitch.T
        normals_cam = np.column_stack([normals[:, 0], normals[:, 2], -normals[:, 1]])
        
        front = normals_cam[:, 2] > -0.05
        faces = mesh.faces[front]
        fnorm = normals_cam[front]
        
        L = np.array([0.45, 0.45, 0.77])
        L /= np.linalg.norm(L)
        
        diffuse = np.maximum(0.0, np.sum(fnorm * L, axis=1))
        ambient = 0.32
        intensity = ambient + (1.0 - ambient) * diffuse
        
        img = np.full((height, width, 3), 20, dtype=np.uint8)
        z_buf = np.full((height, width), -1e9, dtype=np.float32)
        
        f_u = u[faces]
        f_v = v[faces]
        f_z = z[faces]
        
        f_centers = np.mean(verts[faces], axis=1) + center
        is_stem = (f_centers[:, 2] > 28.0) & (np.sqrt(f_centers[:, 0]**2 + f_centers[:, 1]**2) < 10.5)
        
        stem_col = np.array([92, 155, 52], dtype=np.float32)
        pump_col = np.array([255, 125, 12], dtype=np.float32)
        f_cols = np.where(is_stem[:, np.newaxis], stem_col, pump_col)
        
        lit_cols = np.clip(f_cols * intensity[:, np.newaxis], 0, 255).astype(np.uint8)
        
        min_u = np.clip(np.floor(np.min(f_u, axis=1)).astype(int), 0, width - 1)
        max_u = np.clip(np.ceil(np.max(f_u, axis=1)).astype(int), 0, width - 1)
        min_v = np.clip(np.floor(np.min(f_v, axis=1)).astype(int), 0, height - 1)
        max_v = np.clip(np.ceil(np.max(f_v, axis=1)).astype(int), 0, height - 1)
        
        avg_z = np.mean(f_z, axis=1)
        order = np.argsort(avg_z)
        
        for idx in order:
            u0, u1, u2 = f_u[idx]
            v0, v1, v2 = f_v[idx]
            z0, z1, z2 = f_z[idx]
            col = lit_cols[idx]
            
            denom = (v1 - v2)*(u0 - u2) + (u2 - u1)*(v0 - v2)
            if abs(denom) < 1e-5:
                continue
                
            x_min, x_max = min_u[idx], max_u[idx]
            y_min, y_max = min_v[idx], max_v[idx]
            if x_min >= x_max or y_min >= y_max:
                continue
                
            xs = np.arange(x_min, x_max + 1)
            ys = np.arange(y_min, y_max + 1)
            grid_x, grid_y = np.meshgrid(xs, ys)
            
            w0 = ((v1 - v2)*(grid_x - u2) + (u2 - u1)*(grid_y - v2)) / denom
            w1 = ((v2 - v0)*(grid_x - u2) + (u0 - u2)*(grid_y - v2)) / denom
            w2 = 1.0 - w0 - w1
            
            mask = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
            if not np.any(mask):
                continue
                
            pix_z = w0 * z0 + w1 * z1 + w2 * z2
            sub_z = z_buf[y_min:y_max+1, x_min:x_max+1]
            z_mask = mask & (pix_z > sub_z)
            
            sub_z[z_mask] = pix_z[z_mask]
            img[y_min:y_max+1, x_min:x_max+1][z_mask] = col
            
        return Image.fromarray(img)

    img1 = render_view(mesh_intact, 450, 450, yaw_deg=0.0, pitch_deg=8.0)
    img2 = render_view(mesh_intact, 450, 450, yaw_deg=35.0, pitch_deg=18.0)
    img3 = render_view(mesh_jack, 450, 450, yaw_deg=0.0, pitch_deg=8.0)
    img4 = render_view(mesh_jack, 450, 450, yaw_deg=35.0, pitch_deg=18.0)

    combined = Image.new('RGB', (450 * 4, 450), color=(20, 20, 20))
    combined.paste(img1, (0, 0))
    combined.paste(img2, (450, 0))
    combined.paste(img3, (900, 0))
    combined.paste(img4, (1350, 0))

    combined.save(output_png)
    print(f"Rendered preview showcase image saved to: {output_png}")

def main():
    print("=" * 65)
    print("  COMIC PUMPKIN 3D GENERATOR (50mm³ STL)")
    print("  Role: Senior 3D Technical Artist & Computational Geometry Specialist")
    print("=" * 65)
    
    print("\n[1/5] Initializing 3D Spatial Grid (190x190x190 voxels)...")
    xs, ys, zs, spacing, X, Y, Z = build_sdf_grid(190, 190)
    dx, dy, dz = spacing
    
    print("[2/5] Evaluating Pumpkin Body & Stem Signed Distance Fields...")
    d_solid, d_body, z_base_cut = compute_body_and_stem_sdf(X, Y, Z)
    
    print("[3/5] Generating Model 1: Intact Comic Pumpkin (50mm)...")
    mesh_intact = extract_and_normalize_mesh(d_solid, xs, ys, zs, spacing, target_dim=50.0)
    
    out_stl_intact = "comic_pumpkin_50mm.stl"
    mesh_intact.export(out_stl_intact, file_type='stl')
    print(f"  -> Exported: {out_stl_intact}")
    print(f"  -> Watertight: {mesh_intact.is_watertight}")
    print(f"  -> Vertices: {len(mesh_intact.vertices)}, Faces: {len(mesh_intact.faces)}")
    print(f"  -> Bounding Box Dimensions (mm): {mesh_intact.extents[0]:.2f} x {mesh_intact.extents[1]:.2f} x {mesh_intact.extents[2]:.2f}")
    print(f"  -> Volume: {mesh_intact.volume / 1000.0:.2f} cm³")
    
    print("\n[4/5] Generating Model 2: Carved Comic Jack-o'-Lantern (50mm)...")
    d_void = compute_jackolantern_void_sdf(xs, zs, dx, dz, X, Y, Z, d_body, z_base_cut)
    # Boolean subtraction: solid pumpkin minus hollow void
    d_jack = np.maximum(d_solid, -d_void)
    
    mesh_jack = extract_and_normalize_mesh(d_jack, xs, ys, zs, spacing, target_dim=50.0)
    
    out_stl_jack = "comic_jackolantern_50mm.stl"
    mesh_jack.export(out_stl_jack, file_type='stl')
    print(f"  -> Exported: {out_stl_jack}")
    print(f"  -> Watertight: {mesh_jack.is_watertight}")
    print(f"  -> Vertices: {len(mesh_jack.vertices)}, Faces: {len(mesh_jack.faces)}")
    print(f"  -> Bounding Box Dimensions (mm): {mesh_jack.extents[0]:.2f} x {mesh_jack.extents[1]:.2f} x {mesh_jack.extents[2]:.2f}")
    print(f"  -> Solid Wall Volume: {mesh_jack.volume / 1000.0:.2f} cm³")
    
    print("\n[5/5] Generating Multi-View Visual Quality Inspection Sheet...")
    preview_img = "pumpkin_showcase.png"
    render_preview_sheet(mesh_intact, mesh_jack, preview_img)
    
    # Also copy to artifacts directory for rich presentation
    artifact_dir = "/Users/asc/.gemini/antigravity-cli/brain/4ba111c3-71ab-40aa-820e-c993845f10c2"
    if os.path.exists(artifact_dir):
        import shutil
        shutil.copy(preview_img, os.path.join(artifact_dir, preview_img))
        shutil.copy(out_stl_intact, os.path.join(artifact_dir, out_stl_intact))
        shutil.copy(out_stl_jack, os.path.join(artifact_dir, out_stl_jack))
        print(f"  -> Mirrored deliverables to artifact directory: {artifact_dir}")

    print("\nAll tasks completed successfully!")

if __name__ == "__main__":
    main()
