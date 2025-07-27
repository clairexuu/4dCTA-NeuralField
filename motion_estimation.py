import os
import math
import time
import numpy as np
import pandas as pd
import nibabel as nib
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchdiffeq import odeint
from skimage import measure
import trimesh
import matplotlib.pyplot as plt
from scipy.spatial.distance import directed_hausdorff

from dataloader import CTSequenceDataset


# =========================================================
# 1. Mesh Utilities
# =========================================================
def extract_mesh_from_nifti(nifti_path, output_path, level=0.5):
    nii = nib.load(nifti_path)
    data = nii.get_fdata().astype(np.float32)
    spacing = nii.header.get_zooms()
    verts, faces, _, _ = measure.marching_cubes(data, level=level, spacing=spacing)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    mesh.export(output_path)


def extract_all_meshes(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for i in range(20):
        fname = f"{i*5}pct.nii.gz"
        input_path = os.path.join(input_dir, fname)
        output_path = os.path.join(output_dir, f"{i*5}pct.ply")
        extract_mesh_from_nifti(input_path, output_path)


def compute_global_bbox(frames, voxel_spacing, level=0.5):
    bbox_mins, bbox_maxs = [], []
    for frame in frames:
        verts, _, _, _ = measure.marching_cubes(frame, level=level, spacing=voxel_spacing)
        bbox_mins.append(verts.min(axis=0))
        bbox_maxs.append(verts.max(axis=0))
    bbox_min = np.min(np.stack(bbox_mins), axis=0)
    bbox_max = np.max(np.stack(bbox_maxs), axis=0)
    print(f"[DEBUG] Global BBox min={bbox_min}, max={bbox_max}")
    return bbox_min, bbox_max


# =========================================================
# 2. SIREN Velocity Field
# =========================================================
class SineActivation(nn.Module):
    def __init__(self, w0=30.0):
        super().__init__()
        self.w0 = w0
    def forward(self, x):
        return torch.sin(self.w0 * x)


def encode_time(t):
    theta = 2 * math.pi * t
    return torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)


class SIRENVelocityField(nn.Module):
    def __init__(self, hidden_dim=256, num_layers=3, w0=30.0):
        super().__init__()
        self.w0 = w0
        layers = []
        in_dim = 5  # (x, y, z, cos t, sin t)
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(SineActivation(w0=w0))
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(SineActivation(w0=w0))
        layers.append(nn.Linear(hidden_dim, 3))
        self.model = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self):
        with torch.no_grad():
            first_linear = None
            for layer in self.model:
                if isinstance(layer, nn.Linear):
                    first_linear = layer
                    break
            if first_linear is not None:
                bound = 1 / first_linear.in_features
                first_linear.weight.uniform_(-bound, bound)
            is_first = True
            for layer in self.model:
                if isinstance(layer, nn.Linear):
                    if is_first:
                        is_first = False
                        continue
                    bound = math.sqrt(6 / layer.in_features) / self.w0
                    layer.weight.uniform_(-bound, bound)

    def forward(self, coords, t):
        t_enc = encode_time(t)
        if t_enc.dim() == 1:
            t_enc = t_enc.unsqueeze(0)
        if coords.shape[0] != t_enc.shape[0]:
            t_enc = t_enc.expand(coords.shape[0], -1)
        inputs = torch.cat([coords, t_enc], dim=-1)
        return self.model(inputs)


# =========================================================
# 3. ODE Integration
# =========================================================
class VelocityFieldODE(nn.Module):
    def __init__(self, siren_model):
        super().__init__()
        self.siren = siren_model

    def forward(self, t, phi):
        t_full = torch.full((phi.shape[0],), t, device=phi.device)
        return self.siren(phi, t_full)


def integrate_velocity_to_deformation(siren_model, points, time_points, method='rk4'):
    ode_func = VelocityFieldODE(siren_model)
    trajectories = odeint(ode_func, points, time_points, method=method)
    trajectories = torch.clamp(trajectories, -1.0, 1.0)
    return trajectories


# =========================================================
# 4. Normalization Utilities
# =========================================================
def normalize_vertices_to_bbox(vertices, bbox_min, bbox_max):
    return 2 * (vertices - bbox_min) / (bbox_max - bbox_min) - 1


def denormalize_vertices_from_bbox(vertices, bbox_min, bbox_max, debug=False):
    denorm = (vertices + 1) / 2 * (bbox_max - bbox_min) + bbox_min
    if debug:
        print(f"[DEBUG] Denorm X[{denorm[:,0].min():.3f},{denorm[:,0].max():.3f}] "
              f"Y[{denorm[:,1].min():.3f},{denorm[:,1].max():.3f}] "
              f"Z[{denorm[:,2].min():.3f},{denorm[:,2].max():.3f}] mm")
    return denorm


def extract_and_normalize_initial_mesh(nifti_path, bbox_min, bbox_max, level=0.5):
    nii = nib.load(nifti_path)
    data = nii.get_fdata().astype(np.float32)
    voxel_spacing = nii.header.get_zooms()

    vertices_mm, faces, _, _ = measure.marching_cubes(data, level=level, spacing=voxel_spacing)
    print(f"[DEBUG] Raw initial mesh vertices (mm) range: X[{vertices_mm[:,0].min():.3f},{vertices_mm[:,0].max():.3f}]")
    normalized_vertices = normalize_vertices_to_bbox(vertices_mm, bbox_min, bbox_max)
    print(f"[DEBUG] Normalized initial mesh vertices range: X[{normalized_vertices[:,0].min():.3f},{normalized_vertices[:,0].max():.3f}]")
    return normalized_vertices, faces


# =========================================================
# 5. Volume and Trajectory Evaluation
# =========================================================
def compute_mesh_volume(vertices, faces):
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    return abs(mesh.volume)


def compute_frame_volume_gt(frame, voxel_spacing, level=0.5):
    try:
        verts, faces, _, _ = measure.marching_cubes(frame, level=level, spacing=voxel_spacing)
        return compute_mesh_volume(verts, faces)
    except:
        return 0.0


def compute_hausdorff_distance(pred_vertices, gt_vertices):
    """Compute symmetric Hausdorff distance (mm)."""
    forward = directed_hausdorff(pred_vertices, gt_vertices)[0]
    backward = directed_hausdorff(gt_vertices, pred_vertices)[0]
    return max(forward, backward)

def compute_and_plot_volumes(frames, mesh_trajectories, mesh_faces, bbox_min, bbox_max, voxel_spacing, save_path=None):
    """
    Compute GT vs predicted volumes and Hausdorff distances per frame,
    save results to CSV, and plot both metrics.
    """
    num_frames = len(frames)
    gt_volumes, pred_volumes, hausdorff_distances = [], [], []

    for t in range(num_frames):
        # Ground truth mesh (marching cubes)
        verts_gt, faces_gt, _, _ = measure.marching_cubes(frames[t], level=0.5, spacing=voxel_spacing)
        gt_volumes.append(abs(compute_mesh_volume(verts_gt, faces_gt)))

        # Predicted mesh (denormalized)
        verts_pred_norm = mesh_trajectories[t].detach().cpu().numpy()
        verts_pred_mm = denormalize_vertices_from_bbox(verts_pred_norm, bbox_min, bbox_max)

        pred_volumes.append(abs(compute_mesh_volume(verts_pred_mm, mesh_faces)))

        # Hausdorff distance (symmetric)
        hd = compute_hausdorff_distance(verts_pred_mm, verts_gt)
        hausdorff_distances.append(hd)
        print(f"[DEBUG] Frame {t}: Hausdorff distance = {hd:.3f} mm")

    # Create CSV
    frames_idx = np.arange(num_frames)
    if save_path:
        volume_data = pd.DataFrame({
            'frame': frames_idx,
            'time_phase': frames_idx / (num_frames - 1),
            'gt_volume_mm3': gt_volumes,
            'predicted_volume_mm3': pred_volumes,
            'volume_diff_mm3': np.array(pred_volumes) - np.array(gt_volumes),
            'volume_diff_percent': ((np.array(pred_volumes) - np.array(gt_volumes)) / np.array(gt_volumes)) * 100,
            'hausdorff_mm': hausdorff_distances
        })
        csv_path = save_path.replace('.png', '.csv')
        volume_data.to_csv(csv_path, index=False, float_format='%.3f')
        print(f"[DEBUG] Volume + Hausdorff CSV saved to {csv_path}")

        print(f"[DEBUG] Hausdorff stats: mean={np.mean(hausdorff_distances):.3f} mm, "
              f"max={np.max(hausdorff_distances):.3f} mm")

    # Plot: Volume + Hausdorff
    plt.figure(figsize=(7, 6))

    # Subplot 1: Volume comparison
    plt.subplot(2, 1, 1)
    plt.plot(frames_idx, gt_volumes, 'o-', color='orange', label="Reference Volume")
    plt.plot(frames_idx, pred_volumes, 'o-', color='blue', label="Predicted Volume")
    plt.ylabel("Volume (mm³)")
    plt.title("Volume Comparison Over Time")
    plt.grid(True)
    plt.legend()

    # Subplot 2: Hausdorff distance
    plt.subplot(2, 1, 2)
    plt.plot(frames_idx, hausdorff_distances, 'o-', color='red', label="Hausdorff Distance")
    plt.xlabel("Frame (time)")
    plt.ylabel("Hausdorff (mm)")
    plt.title("Hausdorff Distance Over Time")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[DEBUG] Volume + Hausdorff plot saved to {save_path}")
    else:
        plt.show()


# =========================================================
# 6. Intensity Sampling
# =========================================================
def sample_intensity(volume, coords):
    coords = coords.view(1, -1, 1, 1, 3)
    sampled = F.grid_sample(volume, coords, align_corners=True, mode='bilinear', padding_mode='border')
    return sampled.view(-1)


# =========================================================
# 7. Training Loop
# =========================================================
def train_inr_model(
    siren_model,
    frames,
    spatial_coords,
    temporal_coords,
    mesh_faces,
    bbox_min,
    bbox_max,
    voxel_spacing,
    num_epochs=1000,
    sample_points=10000,
    lambda_cycle=0.1,
    lambda_volume=0.01,
    device='cuda'
):
    print(f"[DEBUG] Training with {num_epochs} epochs, {sample_points} sample points/epoch")
    siren_model = siren_model.to(device)
    optimizer = optim.Adam(siren_model.parameters(), lr=1e-5)

    T = len(frames)
    spatial_coords_torch = torch.from_numpy(spatial_coords).float().to(device)
    frames_tensor = [torch.from_numpy(frame).float().to(device) for frame in frames]
    target_frame = frames_tensor[-1]

    def random_sample_indices(total_points):
        return torch.randint(0, spatial_coords_torch.shape[0], (total_points,), device=device)

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        idx = random_sample_indices(sample_points)
        sampled_coords = spatial_coords_torch[idx]

        if epoch % 500 == 0 or epoch == 0:
            print(f"[DEBUG] Epoch {epoch}: sampled coord range X[{sampled_coords[:,0].min():.3f},{sampled_coords[:,0].max():.3f}]")

        recon_losses = []
        for frame_idx in range(T - 1):
            t_start = temporal_coords[frame_idx]
            t_end = temporal_coords[-1]
            time_subset = torch.linspace(t_start, t_end, steps=T - frame_idx, device=device)

            trajectories = integrate_velocity_to_deformation(siren_model, sampled_coords, time_subset)
            warped_points = trajectories[-1]

            I_ti_original = sample_intensity(frames_tensor[frame_idx].unsqueeze(0).unsqueeze(0), sampled_coords)
            I_T_warped = sample_intensity(target_frame.unsqueeze(0).unsqueeze(0), warped_points)
            recon_losses.append(torch.sum((I_ti_original - I_T_warped) ** 2))

        recon_loss = torch.sum(torch.stack(recon_losses))
        full_trajectories = integrate_velocity_to_deformation(
            siren_model, sampled_coords, torch.from_numpy(temporal_coords).float().to(device)
        )
        cycle_loss = torch.mean((sampled_coords - full_trajectories[-1]) ** 2)

        total_loss = recon_loss + lambda_cycle * cycle_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(siren_model.parameters(), max_norm=1.0)
        optimizer.step()

        if epoch % 50 == 0:
            print(f"[DEBUG] Epoch {epoch}: Total={total_loss.item():.4f}, Recon={recon_loss.item():.4f}, Cycle={cycle_loss.item():.4f}")

    print("[DEBUG] Training complete.")
    return siren_model


# =========================================================
# 8. Main
# =========================================================
if __name__ == "__main__":
    data_path = "data/4007775_aneurysm/nnunet_outputs_pp"
    mesh_output_path = "data/4007775_aneurysm/meshes"
    visualization_path = "data/4007775_aneurysm/visualizations"
    os.makedirs(visualization_path, exist_ok=True)

    print("=== Step 1: Load 4D CTA data ===")
    dataset = CTSequenceDataset(data_path, num_frames=20)
    frames, temporal_coords = dataset.get_all_frames()
    spatial_coords = dataset.get_spatial_coords()  # Already [-1,1]
    voxel_spacing = dataset.get_voxel_spacing()

    print("=== Step 2: Extract meshes ===")
    extract_all_meshes(data_path, mesh_output_path)

    print("=== Step 3: Compute global bounding box ===")
    bbox_min, bbox_max = compute_global_bbox(frames, voxel_spacing)

    print("=== Step 4: Extract and normalize initial mesh ===")
    initial_nifti_path = os.path.join(data_path, "0pct.nii.gz")
    initial_vertices_normalized, mesh_faces = extract_and_normalize_initial_mesh(
        initial_nifti_path, bbox_min, bbox_max
    )

    print("=== Step 5: Train SIREN model ===")
    siren_model = SIRENVelocityField(hidden_dim=256)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trained_model = train_inr_model(
        siren_model,
        frames,
        spatial_coords,
        temporal_coords,
        mesh_faces,
        bbox_min,
        bbox_max,
        voxel_spacing,
        num_epochs=1000,
        sample_points=10000,
        lambda_cycle=0.1,
        lambda_volume=0.01,
        device=device
    )

    print("=== Step 6: Generate mesh trajectories ===")
    trained_model.eval()
    mesh_vertices_tensor = torch.from_numpy(initial_vertices_normalized).float().to(device)
    time_tensor = torch.from_numpy(temporal_coords).float().to(device)

    with torch.no_grad():
        mesh_trajectories = integrate_velocity_to_deformation(trained_model, mesh_vertices_tensor, time_tensor)
    print(f"[DEBUG] Mesh trajectories shape: {mesh_trajectories.shape}")

    print("=== Step 7: Compute and plot volume comparison ===")
    vol_plot_path = os.path.join(visualization_path, "volume_comparison.png")
    compute_and_plot_volumes(frames, mesh_trajectories, mesh_faces, bbox_min, bbox_max, voxel_spacing, save_path=vol_plot_path)