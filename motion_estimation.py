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
from tqdm import tqdm

from dataloader import CTSequenceDataset


# =========================================================
# 1. Utilities: Bounding Box Normalization
# =========================================================
def normalize_vertices_to_bbox(vertices, bbox_min, bbox_max):
    return 2 * (vertices - bbox_min) / (bbox_max - bbox_min) - 1

def denormalize_vertices_from_bbox(vertices, bbox_min, bbox_max):
    return (vertices + 1) / 2 * (bbox_max - bbox_min) + bbox_min


# =========================================================
# 2. Mesh + Volume Utilities
# =========================================================
def compute_mesh_volume(vertices, faces):
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    return abs(mesh.volume)

def compute_frame_volume_gt(frame, voxel_spacing, level=0.5):
    verts, faces, _, _ = measure.marching_cubes(frame, level=level, spacing=voxel_spacing)
    return abs(compute_mesh_volume(verts, faces))


# =========================================================
# 3. SIREN Velocity Model
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
        in_dim = 5  # x,y,z + cos t + sin t
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(SineActivation(w0=w0))
        for _ in range(num_layers-1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(SineActivation(w0=w0))
        layers.append(nn.Linear(hidden_dim, 3))
        self.model = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self):
        with torch.no_grad():
            first = True
            for layer in self.model:
                if isinstance(layer, nn.Linear):
                    if first:
                        bound = 1 / layer.in_features
                        layer.weight.uniform_(-bound, bound)
                        first = False
                    else:
                        bound = math.sqrt(6 / layer.in_features) / self.w0
                        layer.weight.uniform_(-bound, bound)

    def forward(self, coords, t):
        t_enc = encode_time(t)
        if t_enc.dim() == 1:
            t_enc = t_enc.unsqueeze(0)
        if coords.shape[0] != t_enc.shape[0]:
            t_enc = t_enc.expand(coords.shape[0], -1)
        return self.model(torch.cat([coords, t_enc], dim=-1))


# =========================================================
# 4. ODE Integration
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
    return torch.clamp(trajectories, -1.0, 1.0)  # keep in [-1,1]


# =========================================================
# 5. Intensity Sampling
# =========================================================
def sample_intensity(volume, coords):
    coords = coords.view(1, -1, 1, 1, 3)
    sampled = F.grid_sample(volume, coords, align_corners=True, mode='bilinear', padding_mode='border')
    return sampled.view(-1)


# =========================================================
# 6. Volume Plotting (CSV + Plot)
# =========================================================
def plot_and_save_volumes(gt_volumes, pred_volumes, save_path):
    frames_idx = np.arange(len(gt_volumes))
    volume_data = pd.DataFrame({
        'frame': frames_idx,
        'time_phase': frames_idx / (len(gt_volumes)-1),
        'gt_volume_mm3': gt_volumes,
        'predicted_volume_mm3': pred_volumes,
        'volume_diff_mm3': np.array(pred_volumes) - np.array(gt_volumes),
        'volume_diff_percent': ((np.array(pred_volumes) - np.array(gt_volumes)) / np.array(gt_volumes)) * 100
    })
    csv_path = save_path.replace('.png', '.csv')
    volume_data.to_csv(csv_path, index=False, float_format='%.3f')
    print(f"[INFO] Saved volume CSV to {csv_path}")

    plt.figure(figsize=(6, 4))
    plt.plot(frames_idx, gt_volumes, 'o-', label='Reference', color='orange')
    plt.plot(frames_idx, pred_volumes, 'o-', label='Predicted', color='blue')
    plt.xlabel("Frame")
    plt.ylabel("Volume (mm³)")
    plt.title("Volume Comparison")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[INFO] Saved volume plot to {save_path}")


# =========================================================
# 7. Training Loop
# =========================================================
def train_inr_model(
    siren_model, frames, spatial_coords, temporal_coords,
    mesh_faces, bbox_min, bbox_max, voxel_spacing,
    num_epochs=1000, sample_points=10000,
    lambda_cycle=0.1, lambda_volume=0.01,
    device='cuda'
):
    print(f"=== Training SIREN ===")
    print(f"Bounding box: min={bbox_min}, max={bbox_max}")
    print(f"Spatial coords (normalized) range: "
          f"X[{spatial_coords[:,0].min():.3f},{spatial_coords[:,0].max():.3f}] "
          f"Y[{spatial_coords[:,1].min():.3f},{spatial_coords[:,1].max():.3f}] "
          f"Z[{spatial_coords[:,2].min():.3f},{spatial_coords[:,2].max():.3f}]")

    siren_model = siren_model.to(device)
    optimizer = optim.Adam(siren_model.parameters(), lr=1e-5)

    T = len(frames)
    spatial_coords_torch = torch.from_numpy(spatial_coords).float().to(device)
    frames_tensor = [torch.from_numpy(frame).float().to(device) for frame in frames]
    target_frame = frames_tensor[-1]

    # Precompute GT volumes
    gt_volumes = [compute_frame_volume_gt(f, voxel_spacing) for f in frames]

    # Sampling function
    def random_sample_indices(n):
        return torch.randint(0, spatial_coords_torch.shape[0], (n,), device=device)

    start_time = time.time()

    # Initialize tqdm
    pbar = tqdm(range(num_epochs), desc="Training", ncols=100)

    for epoch in pbar:
        optimizer.zero_grad()

        # Sample points
        idx = random_sample_indices(sample_points)
        sampled_coords = spatial_coords_torch[idx]

        if epoch in [0, 500, 999]:
            print(f"[Epoch {epoch}] Sampled coords (first 3): {sampled_coords[:3].cpu().numpy()}")

        # Reconstruction loss
        recon_losses = []
        for frame_idx in range(T-1):
            t_start = temporal_coords[frame_idx]
            t_end = temporal_coords[-1]
            time_subset = torch.linspace(t_start, t_end, steps=T-frame_idx, device=device)

            trajectories = integrate_velocity_to_deformation(siren_model, sampled_coords, time_subset)
            warped_points = trajectories[-1]

            I_ti_original = sample_intensity(frames_tensor[frame_idx].unsqueeze(0).unsqueeze(0), sampled_coords)
            I_T_warped = sample_intensity(target_frame.unsqueeze(0).unsqueeze(0), warped_points)

            recon_losses.append(torch.sum((I_ti_original - I_T_warped) ** 2))

        recon_loss = torch.sum(torch.stack(recon_losses))

        # Cycle loss
        full_trajectories = integrate_velocity_to_deformation(
            siren_model, sampled_coords, torch.from_numpy(temporal_coords).float().to(device)
        )
        cycle_loss = torch.mean((sampled_coords - full_trajectories[-1]) ** 2)

        # Volume loss (every 50 epochs)
        volume_loss = 0.0
        if lambda_volume > 0 and epoch % 50 == 0:
            subset = sampled_coords[:500]
            t_target = temporal_coords[len(temporal_coords)//2]
            time_to_frame = torch.linspace(0, t_target, steps=10, device=device)
            frame_trajectories = integrate_velocity_to_deformation(siren_model, subset, time_to_frame)
            initial_spread = torch.std(subset, dim=0).mean()
            final_spread = torch.std(frame_trajectories[-1], dim=0).mean()
            volume_ratio = (final_spread / initial_spread) ** 3
            pred_volume = gt_volumes[len(gt_volumes)//2] * volume_ratio.item()
            volume_loss = (gt_volumes[len(gt_volumes)//2] - pred_volume) ** 2

        # Total loss
        total_loss = recon_loss + lambda_cycle * cycle_loss + lambda_volume * volume_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(siren_model.parameters(), max_norm=1.0)
        optimizer.step()

        # Update tqdm postfix
        if epoch % 10 == 0:
            pbar.set_postfix({
                "Total": f"{total_loss.item():.2f}",
                "Recon": f"{recon_loss.item():.2f}",
                "Cycle": f"{cycle_loss.item():.2f}",
                "Vol": f"{volume_loss:.2f}"
            })

    print(f"Training done in {(time.time()-start_time)/60:.1f} min")
    return siren_model, gt_volumes


# =========================================================
# 8. Main
# =========================================================
if __name__ == "__main__":
    data_path = "data/4007775_aneurysm/nnunet_outputs_pp"
    mesh_output_path = "data/4007775_aneurysm/meshes"
    visualization_path = "data/4007775_aneurysm/visualizations"
    os.makedirs(visualization_path, exist_ok=True)

    # Load data
    dataset = CTSequenceDataset(data_path, num_frames=20)
    frames, temporal_coords = dataset.get_all_frames()
    spatial_coords = dataset.get_spatial_coords()
    voxel_spacing = dataset.get_voxel_spacing()

    # Initial mesh (0pct)
    initial_nifti_path = os.path.join(data_path, "0pct.nii.gz")
    nii = nib.load(initial_nifti_path)
    data = nii.get_fdata().astype(np.float32)
    raw_vertices_mm, mesh_faces, _, _ = measure.marching_cubes(data, level=0.5, spacing=voxel_spacing)

    # Bounding box normalization
    bbox_min = raw_vertices_mm.min(axis=0)
    bbox_max = raw_vertices_mm.max(axis=0)
    initial_vertices_normalized = normalize_vertices_to_bbox(raw_vertices_mm, bbox_min, bbox_max)
    spatial_coords = normalize_vertices_to_bbox(spatial_coords, bbox_min, bbox_max)

    # Train
    siren_model = SIRENVelocityField(hidden_dim=256)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trained_model, gt_volumes = train_inr_model(
        siren_model, frames, spatial_coords, temporal_coords,
        mesh_faces, bbox_min, bbox_max, voxel_spacing,
        num_epochs=1000, sample_points=10000,
        lambda_cycle=0.1, lambda_volume=0.01,
        device=device
    )

    # Generate mesh trajectories
    mesh_vertices_tensor = torch.from_numpy(initial_vertices_normalized).float().to(device)
    time_tensor = torch.from_numpy(temporal_coords).float().to(device)
    with torch.no_grad():
        mesh_trajectories = integrate_velocity_to_deformation(trained_model, mesh_vertices_tensor, time_tensor)

    # Predicted volumes
    pred_volumes = []
    for t in range(len(frames)):
        verts_mm = denormalize_vertices_from_bbox(mesh_trajectories[t].cpu().numpy(), bbox_min, bbox_max)
        pred_volumes.append(compute_mesh_volume(verts_mm, mesh_faces))

    # Plot + save
    vol_plot_path = os.path.join(visualization_path, "volume_comparison.png")
    plot_and_save_volumes(gt_volumes, pred_volumes, vol_plot_path)