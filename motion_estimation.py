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
from scipy.spatial.distance import directed_hausdorff
import pyvista as pv
import matplotlib.pyplot as plt

from dataloader import CTSequenceDataset




# =========================================================
# 2. Mesh Extraction
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


# =========================================================
# 3. SIREN Model + Time Encoding
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
        in_dim = 5  # (x,y,z, cos t, sin t)
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(SineActivation(w0=w0))
        for _ in range(num_layers-1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(SineActivation(w0=w0))  # Use same w0 for all layers per paper
        layers.append(nn.Linear(hidden_dim, 3))
        self.model = nn.Sequential(*layers)
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights according to SIREN paper."""
        with torch.no_grad():
            # First layer: uniform distribution from [-1/in_dim, 1/in_dim]
            first_linear = None
            for layer in self.model:
                if isinstance(layer, nn.Linear):
                    first_linear = layer
                    break
            
            if first_linear is not None:
                bound = 1 / first_linear.in_features
                first_linear.weight.uniform_(-bound, bound)
            
            # Hidden layers: uniform distribution from [-sqrt(6/hidden_dim)/w0, sqrt(6/hidden_dim)/w0]
            is_first = True
            for layer in self.model:
                if isinstance(layer, nn.Linear):
                    if is_first:
                        is_first = False
                        continue  # Skip first layer, already initialized
                    bound = math.sqrt(6 / layer.in_features) / self.w0
                    layer.weight.uniform_(-bound, bound)
    def forward(self, coords, t):
        t_enc = encode_time(t)
        # Broadcast time encoding to match coords batch size
        if t_enc.dim() == 1:
            t_enc = t_enc.unsqueeze(0)  # (1, 2)
        if coords.shape[0] != t_enc.shape[0]:
            t_enc = t_enc.expand(coords.shape[0], -1)  # (N, 2)
        inputs = torch.cat([coords, t_enc], dim=-1)
        return self.model(inputs)


# =========================================================
# 4. ODE Integration (VVF → DVF)
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
    # Clamp coordinates to prevent boundary violations
    trajectories = torch.clamp(trajectories, -1.0, 1.0)
    return trajectories  # (T, N, 3)


# =========================================================
# 5. Point Sampling Functions
# =========================================================


# Function to sample intensities at arbitrary coordinates
def sample_intensity(volume, coords):
    """
    volume: (1,1,H,W,D)
    coords: (N,3) in [-1,1]^3
    Returns: (N,) interpolated intensities
    """
    # Reshape coords to (1,D,H,W,3) format expected by grid_sample
    coords = coords.view(1, -1, 1, 1, 3)  # fake dimensions for 5D
    sampled = F.grid_sample(volume, coords, align_corners=True, mode='bilinear', padding_mode='border')
    return sampled.view(-1)






# =========================================================
# 8. Visualization Utilities
# =========================================================

def plot_point_trajectories(trajectories, point_indices, save_path=None):
    traj_np = trajectories.detach().cpu().numpy()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for idx in point_indices:
        ax.plot(traj_np[:, idx, 0], traj_np[:, idx, 1], traj_np[:, idx, 2], marker='o')
    ax.set_title("Point trajectories across cycle")
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved point trajectory plot to {save_path}")
    plt.show()

def compute_mesh_volume(vertices, faces):
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    return abs(mesh.volume)

def compute_frame_volume_gt(frame, voxel_spacing, level=0.5):
    """Compute ground truth volume for a single frame using marching cubes."""
    try:
        verts_gt, faces_gt, _, _ = measure.marching_cubes(frame, level=level, spacing=voxel_spacing)
        return abs(compute_mesh_volume(verts_gt, faces_gt))
    except:
        return 0.0  # Return 0 if marching cubes fails


def normalize_vertices_to_grid(vertices, volume_shape, voxel_spacing):
    """
    Convert real-world mm coordinates from marching_cubes to normalized [-1,1] grid coordinates.
    Args:
        vertices: (N, 3) vertices in mm space from marching_cubes
        volume_shape: (H, W, D) shape of the volume
        voxel_spacing: (sx, sy, sz) voxel spacing in mm
    Returns:
        normalized_vertices: (N, 3) vertices in [-1,1]^3 space
    """
    H, W, D = volume_shape
    sx, sy, sz = voxel_spacing
    
    # Convert mm coordinates back to voxel indices
    verts = vertices.copy()
    verts[:, 0] /= sx  # x
    verts[:, 1] /= sy  # y 
    verts[:, 2] /= sz  # z
    
    # Convert voxel indices to [-1,1] normalized coordinates
    # Ensure coordinates are properly clipped to [-1,1] range
    verts[:, 0] = np.clip((verts[:, 0] / (D-1)) * 2 - 1, -1, 1)  # x: [0,D-1] -> [-1,1]
    verts[:, 1] = np.clip((verts[:, 1] / (W-1)) * 2 - 1, -1, 1)  # y: [0,W-1] -> [-1,1]
    verts[:, 2] = np.clip((verts[:, 2] / (H-1)) * 2 - 1, -1, 1)  # z: [0,H-1] -> [-1,1]
    
    return verts

def denormalize_vertices(vertices, volume_shape, voxel_spacing, debug=False):
    """
    Convert normalized [-1,1] coordinates to real-world mm space
    """
    H, W, D = volume_shape
    sx, sy, sz = voxel_spacing  # from nii.header.get_zooms()
    
    if debug:
        print(f"🔍 Denormalize Debug:")
        print(f"   Volume shape: {volume_shape} (H={H}, W={W}, D={D})")
        print(f"   Voxel spacing: {voxel_spacing} (sx={sx:.3f}, sy={sy:.3f}, sz={sz:.3f}) mm")
        print(f"   Input vertices shape: {vertices.shape}")
        print(f"   First 3 normalized vertices:")
        for i in range(min(3, vertices.shape[0])):
            print(f"      [{i}]: ({vertices[i,0]:.3f}, {vertices[i,1]:.3f}, {vertices[i,2]:.3f})")

    # Convert [-1,1] -> [0,H/W/D]
    verts = (vertices + 1) / 2.0
    verts[:, 0] *= D
    verts[:, 1] *= W
    verts[:, 2] *= H

    # Scale by voxel spacing (note: marching_cubes uses spacing=(sx,sy,sz))
    verts[:, 0] *= sx
    verts[:, 1] *= sy
    verts[:, 2] *= sz

    if debug:
        print(f"   First 3 denormalized vertices (mm):")
        for i in range(min(3, verts.shape[0])):
            print(f"      [{i}]: ({verts[i,0]:.3f}, {verts[i,1]:.3f}, {verts[i,2]:.3f})")
        print(f"   Coordinate ranges after denormalization:")
        print(f"      X: [{verts[:,0].min():.3f}, {verts[:,0].max():.3f}] mm")
        print(f"      Y: [{verts[:,1].min():.3f}, {verts[:,1].max():.3f}] mm") 
        print(f"      Z: [{verts[:,2].min():.3f}, {verts[:,2].max():.3f}] mm")

    return verts

def extract_and_normalize_initial_mesh(nifti_path, volume_shape, voxel_spacing, level=0.5):
    """
    Extract mesh from initial frame and normalize vertices to [-1,1]^3.
    Args:
        nifti_path: path to initial frame NIfTI file
        volume_shape: (H, W, D) shape of the volume
        voxel_spacing: (sx, sy, sz) voxel spacing
        level: isosurface level for marching cubes
    Returns:
        normalized_vertices: (N, 3) vertices in [-1,1]^3
        faces: (F, 3) face connectivity
    """
    nii = nib.load(nifti_path)
    data = nii.get_fdata().astype(np.float32)
    
    # Extract mesh using marching cubes
    vertices_mm, faces, _, _ = measure.marching_cubes(data, level=level, spacing=voxel_spacing)
    
    # Normalize vertices to [-1,1]^3 coordinate system
    normalized_vertices = normalize_vertices_to_grid(vertices_mm, volume_shape, voxel_spacing)
    
    return normalized_vertices, faces


def compute_and_plot_volumes(frames, mesh_trajectories, mesh_faces, voxel_spacing, spatial_shape, save_path=None):
    """
    Compute GT and predicted volumes (mm³) over time and plot comparison.
    Args:
        frames: list of 3D numpy arrays (H,W,D)
        mesh_trajectories: torch.Tensor (T, N_vertices, 3) normalized predicted mesh vertices
        mesh_faces: numpy array (F,3) mesh connectivity from initial frame
        voxel_spacing: tuple (sx, sy, sz)
        spatial_shape: (H,W,D) of frames
    """
    num_frames = len(frames)
    gt_volumes, pred_volumes = [], []

    # GT: marching cubes with voxel spacing
    for t in range(num_frames):
        verts_gt, faces_gt, _, _ = measure.marching_cubes(frames[t], level=0.5, spacing=voxel_spacing)
        gt_volumes.append(abs(compute_mesh_volume(verts_gt, faces_gt)))

    # Predicted: mesh trajectories → mm
    for t in range(num_frames):
        verts_pred_norm = mesh_trajectories[t].detach().cpu().numpy()
        # Enable debug for first frame only to avoid spam
        debug_denorm = (t == 0)
        
        # Debug: Print original vertices before any processing
        if debug_denorm:
            print(f"📋 Original vertices (normalized [-1,1]³) at frame {t}:")
            print(f"   Shape: {verts_pred_norm.shape}")
            print(f"   Range: X=[{verts_pred_norm[:,0].min():.3f}, {verts_pred_norm[:,0].max():.3f}]")
            print(f"          Y=[{verts_pred_norm[:,1].min():.3f}, {verts_pred_norm[:,1].max():.3f}]")
            print(f"          Z=[{verts_pred_norm[:,2].min():.3f}, {verts_pred_norm[:,2].max():.3f}]")
            print(f"   First 3 original vertices:")
            for i in range(min(3, verts_pred_norm.shape[0])):
                print(f"      [{i}]: ({verts_pred_norm[i,0]:.3f}, {verts_pred_norm[i,1]:.3f}, {verts_pred_norm[i,2]:.3f})")
        
        verts_pred_mm = denormalize_vertices(verts_pred_norm, spatial_shape, voxel_spacing, debug=debug_denorm)
        pred_volumes.append(abs(compute_mesh_volume(verts_pred_mm, mesh_faces)))

    # Create frame indices array
    frames_idx = np.arange(num_frames)
    
    # Save volume data to CSV
    if save_path:
        # Create DataFrame with volume data
        volume_data = pd.DataFrame({
            'frame': frames_idx,
            'time_phase': frames_idx / (num_frames - 1),  # Normalized time [0,1]
            'gt_volume_mm3': gt_volumes,
            'predicted_volume_mm3': pred_volumes,
            'volume_diff_mm3': np.array(pred_volumes) - np.array(gt_volumes),
            'volume_diff_percent': ((np.array(pred_volumes) - np.array(gt_volumes)) / np.array(gt_volumes)) * 100
        })
        
        # Save CSV
        csv_path = save_path.replace('.png', '.csv')
        volume_data.to_csv(csv_path, index=False, float_format='%.3f')
        print(f"Saved volume data to {csv_path}")
        
        # Print summary statistics
        print(f"📊 Volume Statistics:")
        print(f"   GT Volume Range: {min(gt_volumes):.1f} - {max(gt_volumes):.1f} mm³")
        print(f"   Predicted Volume Range: {min(pred_volumes):.1f} - {max(pred_volumes):.1f} mm³")
        print(f"   Mean Absolute Error: {np.mean(np.abs(volume_data['volume_diff_mm3'])):.3f} mm³")
        print(f"   Mean Percent Error: {np.mean(np.abs(volume_data['volume_diff_percent'])):.2f}%")

    # Plot
    plt.figure(figsize=(6, 4))
    plt.plot(frames_idx, gt_volumes, 'o-', color='orange', label="Reference")
    plt.plot(frames_idx, pred_volumes, 'o-', color='blue', label="Predicted")
    plt.xlabel("Frame (time)")
    plt.ylabel("Volume (mm³)")
    plt.title("Volume Comparison")
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved volume comparison plot to {save_path}")
    else:
        plt.show()

# =========================================================
# 9. Training Loop
# =========================================================
def train_inr_model(
    siren_model,
    frames,
    spatial_coords,
    temporal_coords,
    mesh_faces,
    voxel_spacing,
    num_epochs=1000,
    sample_points=10000,
    lambda_cycle=0.1,
    lambda_volume=0.01,
    device='cuda'
):
    """
    Training loop implementing paper's exact loss formulation using point sampling.
    """

    print(f"Training with point sampling reconstruction for {num_epochs} epochs...")
    print(f"Using device: {device}")
    print(f"Sample points per epoch: {sample_points}")
    print(f"Lambda cycle: {lambda_cycle}")
    print(f"Lambda volume: {lambda_volume}")
    
    siren_model = siren_model.to(device)
    optimizer = optim.Adam(siren_model.parameters(), lr=1e-5)  # Reduced from 3e-5 for stability
    print("✓ Model and optimizer initialized")

    T = len(frames)
    spatial_coords_torch = torch.from_numpy(spatial_coords).float().to(device)
    print(f"✓ Loaded {T} frames, spatial coords: {spatial_coords_torch.shape}")
    
    # Convert frames to tensors
    frames_tensor = [torch.from_numpy(frame).float().to(device) for frame in frames]
    target_frame = frames_tensor[-1]  # I_T (final frame)
    print(f"✓ Converted frames to tensors, target frame shape: {target_frame.shape}")

    # Paper: "10000 (real dataset) randomly sampled points in the spatial domain"
    # Using pure random sampling as specified in the paper
    def random_sample_indices(total_points):
        return torch.randint(0, spatial_coords_torch.shape[0], (total_points,), device=device)
    
    # # Alternative: Balanced foreground/background sampling (commented out)
    # final_mask = (frames[-1].flatten() > 0)
    # final_mask_tensor = torch.from_numpy(final_mask).to(device)
    # fg_indices = torch.where(final_mask_tensor)[0]
    # bg_indices = torch.where(~final_mask_tensor)[0]
    # print(f"✓ Foreground points: {len(fg_indices)}, Background points: {len(bg_indices)}")
    # def balanced_sample_indices(total_points):
    #     if len(fg_indices) == 0:
    #         return torch.randint(0, spatial_coords_torch.shape[0], (total_points,), device=device)
    #     half = total_points // 2
    #     fg_sample = fg_indices[torch.randint(0, len(fg_indices), (half,))]
    #     bg_sample = bg_indices[torch.randint(0, len(bg_indices), (total_points - half,))]
    #     return torch.cat([fg_sample, bg_sample])

    print("Starting training loop...")
    print("=" * 60)
    
    training_start_time = time.time()
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # Progress indicators
        if epoch == 0:
            print("🚀 Starting epoch 0...")

        # Point sampling approach implementing paper's loss: ||I_ti ∘ φ_ti→T - I_T||²        
        idx = random_sample_indices(sample_points)
        sampled_coords = spatial_coords_torch[idx]

        # Debug: Print first 5 sampled coordinates at specific epochs
        if epoch in [0, 500, 999]:
            print(f"   📍 Epoch {epoch} - First 5 sampled coords:")
            for i in range(5):
                coord = sampled_coords[i].cpu().numpy()
                print(f"      [{i}]: ({coord[0]:.3f}, {coord[1]:.3f}, {coord[2]:.3f})")
            
        recon_losses = []
        for frame_idx in range(T-1):
            if epoch == 0 and frame_idx < 3:
                print(f"      Frame {frame_idx} → target (t={temporal_coords[frame_idx]:.3f} → {temporal_coords[-1]:.3f})")
                    
            t_start = temporal_coords[frame_idx]
            t_end = temporal_coords[-1]
            time_subset = torch.linspace(t_start, t_end, steps=T-frame_idx, device=device)

            # Compute forward deformation φ_ti→T(P)
            trajectories = integrate_velocity_to_deformation(siren_model, sampled_coords, time_subset)
            warped_points = trajectories[-1]

            # Paper's loss: ||I_ti ∘ φ_ti→T - I_T||² = ||I_ti(P) - I_T(φ_ti→T(P))||²
            # Sample I_ti at original locations P
            I_ti_original = sample_intensity(
                frames_tensor[frame_idx].unsqueeze(0).unsqueeze(0), sampled_coords
            )
            # Sample I_T at deformed locations φ_ti→T(P)
            I_T_warped = sample_intensity(
                target_frame.unsqueeze(0).unsqueeze(0), warped_points
            )

            recon_losses.append(torch.sum((I_ti_original - I_T_warped) ** 2))

        recon_loss = torch.sum(torch.stack(recon_losses))
            
        # Use same points for cycle loss to ensure consistent optimization
        full_trajectories = integrate_velocity_to_deformation(
            siren_model, sampled_coords, torch.from_numpy(temporal_coords).float().to(device)
        )
        cycle_loss = torch.mean((sampled_coords - full_trajectories[-1]) ** 2)

        # Volume regularization: Compare predicted vs ground truth volumes
        volume_loss = 0.0
        if lambda_volume > 0 and epoch % 10 == 0:  # Compute every 10 epochs to save time
            volume_losses = []
            for frame_idx in range(min(5, T)):  # Sample first 5 frames for efficiency
                # Get GT volume for this frame
                gt_volume = compute_frame_volume_gt(frames[frame_idx], voxel_spacing)
                
                # Get predicted volume by deforming initial mesh to this frame
                t_target = temporal_coords[frame_idx]
                time_to_frame = torch.linspace(0, t_target, steps=frame_idx+1, device=device)
                if len(time_to_frame) > 1:
                    frame_trajectories = integrate_velocity_to_deformation(
                        siren_model, sampled_coords[:500], time_to_frame  # Use subset for speed
                    )
                    # Approximate volume change based on coordinate displacement
                    initial_spread = torch.std(sampled_coords[:500], dim=0).mean()
                    final_spread = torch.std(frame_trajectories[-1], dim=0).mean()
                    volume_ratio = (final_spread / initial_spread) ** 3
                    pred_volume = gt_volume * volume_ratio.item()  # Scale initial volume
                    
                    volume_losses.append((gt_volume - pred_volume) ** 2)
            
            if volume_losses:
                volume_loss = torch.mean(torch.stack([torch.tensor(v, device=device) for v in volume_losses]))

        # Total loss: reconstruction + λ_cycle * cycle + λ_volume * volume
        total_loss = recon_loss + lambda_cycle * cycle_loss + lambda_volume * volume_loss
        total_loss.backward()
        
        # Add gradient clipping for training stability
        torch.nn.utils.clip_grad_norm_(siren_model.parameters(), max_norm=1.0)
        
        optimizer.step()

        if lambda_volume > 0 and epoch % 10 == 0:
            print(f"[Epoch {epoch}/{num_epochs}] Total={total_loss.item():.6f}, Recon={recon_loss.item():.6f}, Cycle={cycle_loss.item():.6f}, Volume={volume_loss.item():.6f}")
        else:
            print(f"[Epoch {epoch}/{num_epochs}] Total={total_loss.item():.6f}, Recon={recon_loss.item():.6f}, Cycle={cycle_loss.item():.6f}")

    training_end_time = time.time()
    total_training_time = training_end_time - training_start_time
    print(f"Training complete! Total time: {total_training_time/60:.1f} minutes ({total_training_time:.1f} seconds)")
    return siren_model

# =========================================================
# 10. Main Pipeline
# =========================================================
if __name__ == "__main__":
    data_path = "data/4007775_aneurysm/nnunet_outputs_pp"
    mesh_output_path = "data/4007775_aneurysm/meshes"
    visualization_path = "data/4007775_aneurysm/visualizations"
    os.makedirs(visualization_path, exist_ok=True)

    print("=== Step 1: Loading and normalizing 4D CTA data ===")
    dataset = CTSequenceDataset(data_path, num_frames=20)
    frames, temporal_coords = dataset.get_all_frames()
    spatial_coords = dataset.get_spatial_coords()
    voxel_spacing = dataset.get_voxel_spacing()
    print(f"Spatial coords: {spatial_coords.shape}, Temporal coords: {temporal_coords.shape}")

    print("=== Step 2: Extracting meshes from segmentations ===")
    extract_all_meshes(data_path, mesh_output_path)
    print(f"Meshes saved to: {mesh_output_path}")

    print("=== Step 2.5: Extracting and normalizing initial mesh vertices ===")
    # Extract mesh vertices from initial frame (0pct) and normalize to [-1,1]^3
    initial_nifti_path = os.path.join(data_path, "0pct.nii.gz")
    
    # Debug: Show raw vertices from marching cubes before any transformation
    nii = nib.load(initial_nifti_path)
    data = nii.get_fdata().astype(np.float32)
    raw_vertices_mm, faces_raw, _, _ = measure.marching_cubes(data, level=0.5, spacing=voxel_spacing)
    print(f"🔬 Raw vertices from marching cubes (mm coordinates):")
    print(f"   Shape: {raw_vertices_mm.shape}")
    print(f"   Range: X=[{raw_vertices_mm[:,0].min():.3f}, {raw_vertices_mm[:,0].max():.3f}] mm")
    print(f"          Y=[{raw_vertices_mm[:,1].min():.3f}, {raw_vertices_mm[:,1].max():.3f}] mm")
    print(f"          Z=[{raw_vertices_mm[:,2].min():.3f}, {raw_vertices_mm[:,2].max():.3f}] mm")
    print(f"   First 3 raw vertices:")
    for i in range(min(3, raw_vertices_mm.shape[0])):
        print(f"      [{i}]: ({raw_vertices_mm[i,0]:.3f}, {raw_vertices_mm[i,1]:.3f}, {raw_vertices_mm[i,2]:.3f}) mm")
    
    initial_vertices_normalized, mesh_faces = extract_and_normalize_initial_mesh(
        initial_nifti_path, frames[0].shape, voxel_spacing
    )
    print(f"Initial mesh: {initial_vertices_normalized.shape[0]} vertices, {mesh_faces.shape[0]} faces")
    print(f"Vertex coordinate range: [{initial_vertices_normalized.min():.3f}, {initial_vertices_normalized.max():.3f}]")

    print("=== Step 3: Initializing SIREN model ===")
    siren_model = SIRENVelocityField(hidden_dim=256)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    print("=== Step 4: Training SIREN model ===")
    # Start with point sampling due to memory constraints
    trained_model = train_inr_model(
        siren_model,
        frames,
        spatial_coords,
        temporal_coords,
        mesh_faces,
        voxel_spacing,
        num_epochs=1000,
        sample_points=10000,
        lambda_cycle=0.1,
        lambda_volume=0.01,
        device=device
    )
    print("Training complete.")

    print("=== Step 5: Generating mesh vertex trajectories ===")
    # Set model to evaluation mode for inference
    trained_model.eval()
    
    trajectory_start_time = time.time()
    # Use initial mesh vertices (not all spatial coordinates) for trajectory computation
    mesh_vertices_tensor = torch.from_numpy(initial_vertices_normalized).float().to(device)
    time_tensor = torch.from_numpy(temporal_coords).float().to(device)
    
    print(f"Computing trajectories for {mesh_vertices_tensor.shape[0]} mesh vertices...")
    with torch.no_grad():  # Disable gradients for inference
        mesh_trajectories = integrate_velocity_to_deformation(trained_model, mesh_vertices_tensor, time_tensor)
    trajectory_end_time = time.time()
    trajectory_time = trajectory_end_time - trajectory_start_time
    print(f"Mesh trajectories computed: {mesh_trajectories.shape} (T, N_vertices, 3) in {trajectory_time:.1f} seconds")

    print("Plotting sample mesh vertex trajectories...")
    traj_plot_path = os.path.join(visualization_path, "mesh_vertex_trajectories.png")
    # Use vertex indices instead of arbitrary spatial indices
    num_vertices = mesh_vertices_tensor.shape[0]
    sample_vertex_indices = [0, min(num_vertices//4, num_vertices-1), min(num_vertices//2, num_vertices-1)]
    plot_point_trajectories(mesh_trajectories, sample_vertex_indices, save_path=traj_plot_path)

    print("=== Step 6: Computing and plotting volume comparison ===")
    plotting_start_time = time.time()
    vol_plot_path = os.path.join(visualization_path, "volume_comparison.png")
    compute_and_plot_volumes(frames, mesh_trajectories, mesh_faces, voxel_spacing, frames[0].shape, save_path=vol_plot_path)
    plotting_end_time = time.time()
    plotting_time = plotting_end_time - plotting_start_time
    print(f"Volume comparison plotting completed in {plotting_time:.1f} seconds")