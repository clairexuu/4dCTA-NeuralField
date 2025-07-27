import os
import math
import numpy as np
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
# 1. Data Loading & Normalization
# =========================================================
def load_4d_cta(dataset_path):
    frames = []
    for i in range(20):
        filename = f"{i*5}pct.nii.gz"
        file_path = os.path.join(dataset_path, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Missing file: {file_path}")
        img = nib.load(file_path)
        data = img.get_fdata().astype(np.float32)
        frames.append(data)

    # Stack frames into 4D array: (T, H, W, D)
    volume_4d = np.stack(frames, axis=0)  # (20,H,W,D)

    # Normalize intensity to [0,1]
    min_val = np.min(volume_4d)
    max_val = np.max(volume_4d)
    volume_4d = (volume_4d - min_val) / (max_val - min_val + 1e-8)

    return volume_4d


def normalize_spatial_coordinates(shape):
    H, W, D = shape
    z = np.linspace(-1, 1, H)
    y = np.linspace(-1, 1, W)
    x = np.linspace(-1, 1, D)
    zz, yy, xx = np.meshgrid(z, y, x, indexing='ij')
    coords = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
    return coords


def normalize_temporal_coordinates(num_frames=20):
    return np.linspace(0, 1, num_frames, dtype=np.float32)


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
        layers = []
        in_dim = 5  # (x,y,z, cos t, sin t)
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(SineActivation(w0=w0))
        for _ in range(num_layers-1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(SineActivation(w0=1.0))
        layers.append(nn.Linear(hidden_dim, 3))
        self.model = nn.Sequential(*layers)
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

def integrate_velocity_to_deformation(siren_model, points, time_points, method='euler'):
    ode_func = VelocityFieldODE(siren_model)
    trajectories = odeint(ode_func, points, time_points, method=method)
    return trajectories  # (T, N, 3)


# =========================================================
# 5. Warping Functions
# =========================================================
def compute_deformation_field(siren_model, spatial_coords, t_start, t_end, device):
    """
    Compute deformation field φ_t_start→t_end for all spatial coordinates.
    Args:
        siren_model: trained SIREN velocity field
        spatial_coords: (N, 3) normalized spatial coordinates
        t_start, t_end: scalar time values
        device: torch device
    Returns:
        deformation_field: (N, 3) deformed coordinates
    """
    time_points = torch.linspace(t_start, t_end, steps=10, device=device)  # More steps for accuracy
    coords_tensor = torch.from_numpy(spatial_coords).float().to(device)
    trajectories = integrate_velocity_to_deformation(siren_model, coords_tensor, time_points)
    return trajectories[-1]  # Final deformed positions

def warp_volume_full(volume, siren_model, t_start, t_end, device):
    """
    Warp entire volume using deformation field φ_t_start→t_end.
    Implements paper's ||I_ti ∘ φ_ti→T - I_T||² formulation.
    Args:
        volume: (H, W, D) input volume
        siren_model: trained SIREN velocity field
        t_start, t_end: scalar time values
        device: torch device
    Returns:
        warped_volume: (H, W, D) warped volume
    """
    H, W, D = volume.shape
    
    # Create normalized spatial grid [-1,1]³
    z = torch.linspace(-1, 1, H, device=device)
    y = torch.linspace(-1, 1, W, device=device) 
    x = torch.linspace(-1, 1, D, device=device)
    zz, yy, xx = torch.meshgrid(z, y, x, indexing='ij')
    spatial_coords = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)
    
    # Compute deformation field
    deformed_coords = compute_deformation_field(siren_model, spatial_coords, t_start, t_end, device)
    
    # Reshape to grid format and apply warping
    deformed_grid = deformed_coords.reshape(H, W, D, 3)
    
    # Convert volume to tensor format for grid_sample
    volume_tensor = torch.from_numpy(volume).float().to(device).unsqueeze(0).unsqueeze(0)
    
    # Apply grid sampling (note: grid_sample expects (x,y,z) order)
    sampling_grid = deformed_grid.unsqueeze(0)  # (1, H, W, D, 3)
    warped = F.grid_sample(
        volume_tensor, sampling_grid, 
        align_corners=True, mode='bilinear', padding_mode='border'
    )
    
    return warped.squeeze(0).squeeze(0)  # Remove batch and channel dims

def warp_volume(volume, deformation, volume_shape):
    """Legacy function - kept for backward compatibility"""
    H, W, D = volume_shape
    grid_z, grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1,1,H), torch.linspace(-1,1,W), torch.linspace(-1,1,D), indexing='ij'
    )
    base_grid = torch.stack((grid_x, grid_y, grid_z), dim=-1).to(volume.device)
    displacement = deformation.reshape(H,W,D,3) - base_grid
    sampling_grid = base_grid + displacement
    sampling_grid = sampling_grid.unsqueeze(0)  # (1,H,W,D,3)
    warped = F.grid_sample(
        volume, sampling_grid.permute(0,4,1,2,3), align_corners=True,
        mode='bilinear', padding_mode='border'
    )
    return warped

def warp_mesh(vertices, trajectories, time_idx):
    return trajectories[time_idx]


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
# 6. Loss Functions
# =========================================================
def reconstruction_loss(warped_volume, target_volume):
    return torch.mean((warped_volume - target_volume) ** 2)

def cycle_consistency_loss(initial_points, trajectories):
    final_positions = trajectories[-1]
    return torch.mean((initial_points - final_positions) ** 2)


# =========================================================
# 7. Evaluation Metrics
# =========================================================
def compute_psnr(pred, target, max_val=1.0):
    mse = torch.mean((pred - target) ** 2).item()
    if mse == 0:
        return float('inf')
    return 20 * math.log10(max_val / math.sqrt(mse))

def compute_hausdorff_distance(points_A, points_B):
    d_AB = directed_hausdorff(points_A, points_B)[0]
    d_BA = directed_hausdorff(points_B, points_A)[0]
    return max(d_AB, d_BA)


# =========================================================
# 8. Visualization Utilities
# =========================================================
def visualize_mesh_overlay(volume, mesh_vertices, mesh_faces, save_path=None):
    faces_flat = np.hstack([np.full((mesh_faces.shape[0],1),3), mesh_faces]).flatten()
    pv_mesh = pv.PolyData(mesh_vertices, faces_flat)
    plotter = pv.Plotter(off_screen=True)  # offscreen rendering
    plotter.add_volume(volume, cmap="gray", opacity="linear")
    plotter.add_mesh(pv_mesh, color="red", opacity=0.5)
    if save_path:
        plotter.screenshot(save_path)
        print(f"Saved mesh overlay to {save_path}")
    plotter.show()

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

def plot_volume_comparison(volumes_gt, volumes_pred, save_path=None, title="Volume Comparison"):
    frames = np.arange(len(volumes_gt))
    plt.figure(figsize=(6,4))
    plt.plot(frames, volumes_gt, 'o-', color='orange', label="Reference")
    plt.plot(frames, volumes_pred, 'o-', color='blue', label="Predicted")
    plt.xlabel("Frame (time)")
    plt.ylabel("Volume (mm³)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved volume comparison plot to {save_path}")
    plt.show()

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

def denormalize_vertices(vertices, volume_shape, voxel_spacing):
    """
    Convert normalized [-1,1] coordinates to real-world mm space
    """
    H, W, D = volume_shape
    sx, sy, sz = voxel_spacing  # from nii.header.get_zooms()

    # Convert [-1,1] -> [0,H/W/D]
    verts = (vertices + 1) / 2.0
    verts[:, 0] *= D
    verts[:, 1] *= W
    verts[:, 2] *= H

    # Scale by voxel spacing (note: marching_cubes uses spacing=(sx,sy,sz))
    verts[:, 0] *= sx
    verts[:, 1] *= sy
    verts[:, 2] *= sz

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
        verts_pred_mm = denormalize_vertices(verts_pred_norm, spatial_shape, voxel_spacing)
        pred_volumes.append(abs(compute_mesh_volume(verts_pred_mm, mesh_faces)))

    # Plot
    frames_idx = np.arange(num_frames)
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
    num_epochs=1000,
    sample_points=10000,
    lambda_cycle=0.1,
    device='cuda',
    use_full_volume_loss=False  # Disable by default due to memory constraints
):
    """
    Training loop implementing paper's exact loss formulation.
    Uses point sampling by default for memory efficiency, with option for full volume warping.
    """

    print(f"Training with {'full volume' if use_full_volume_loss else 'point sampling'} reconstruction for {num_epochs} epochs...")
    print(f"Using device: {device}")
    print(f"Sample points per epoch: {sample_points}")
    print(f"Lambda cycle: {lambda_cycle}")
    
    siren_model = siren_model.to(device)
    optimizer = optim.Adam(siren_model.parameters(), lr=3e-5)
    print("✓ Model and optimizer initialized")

    T = len(frames)
    spatial_coords_torch = torch.from_numpy(spatial_coords).float().to(device)
    print(f"✓ Loaded {T} frames, spatial coords: {spatial_coords_torch.shape}")
    
    # Convert frames to tensors
    frames_tensor = [torch.from_numpy(frame).float().to(device) for frame in frames]
    target_frame = frames_tensor[-1]  # I_T (final frame)
    print(f"✓ Converted frames to tensors, target frame shape: {target_frame.shape}")

    # For point sampling fallback - balanced sampling indices
    final_mask = (frames[-1].flatten() > 0)
    final_mask_tensor = torch.from_numpy(final_mask).to(device)
    fg_indices = torch.where(final_mask_tensor)[0]
    bg_indices = torch.where(~final_mask_tensor)[0]
    print(f"✓ Foreground points: {len(fg_indices)}, Background points: {len(bg_indices)}")

    def balanced_sample_indices(total_points):
        if len(fg_indices) == 0:
            return torch.randint(0, spatial_coords_torch.shape[0], (total_points,), device=device)
        half = total_points // 2
        fg_sample = fg_indices[torch.randint(0, len(fg_indices), (half,))]
        bg_sample = bg_indices[torch.randint(0, len(bg_indices), (total_points - half,))]
        return torch.cat([fg_sample, bg_sample])

    print("Starting training loop...")
    print("=" * 60)
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # Progress indicators
        if epoch == 0:
            print("🚀 Starting epoch 0...")

        if use_full_volume_loss:
            # Paper's exact formulation: ||I_ti ∘ φ_ti→T - I_T||² (memory intensive)
            recon_losses = []
            # Process fewer frames to save memory
            frame_step = max(1, (T-1) // 5)  # Process every 5th frame
            for frame_idx in range(0, T-1, frame_step):
                t_start = temporal_coords[frame_idx]
                t_end = temporal_coords[-1]
                
                try:
                    # Use reduced resolution for memory efficiency
                    H, W, D = frames_tensor[frame_idx].shape
                    if H * W * D > 50000:  # Downsample if too large
                        downsample_factor = 2
                        frame_ds = F.avg_pool3d(
                            frames_tensor[frame_idx].unsqueeze(0).unsqueeze(0),
                            kernel_size=downsample_factor
                        ).squeeze(0).squeeze(0)
                        target_ds = F.avg_pool3d(
                            target_frame.unsqueeze(0).unsqueeze(0),
                            kernel_size=downsample_factor
                        ).squeeze(0).squeeze(0)
                        warped_frame = warp_volume_full(frame_ds, siren_model, t_start, t_end, device)
                        frame_loss = torch.mean((warped_frame - target_ds) ** 2)
                    else:
                        warped_frame = warp_volume_full(
                            frames_tensor[frame_idx], siren_model, t_start, t_end, device
                        )
                        frame_loss = torch.mean((warped_frame - target_frame) ** 2)
                        
                    recon_losses.append(frame_loss)
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"Warning: OOM at frame {frame_idx}, switching to point sampling")
                        torch.cuda.empty_cache()
                        use_full_volume_loss = False
                        break
                    else:
                        raise e
                        
            if recon_losses:
                recon_loss = torch.mean(torch.stack(recon_losses))
            else:
                use_full_volume_loss = False
            
        else:
            # Point sampling approach implementing paper's loss: ||I_ti ∘ φ_ti→T - I_T||²
            if epoch == 0:
                print("   🎯 Using point sampling approach")
                print(f"   📊 Processing {T-1} frame pairs")
                
            idx = balanced_sample_indices(sample_points)
            sampled_coords = spatial_coords_torch[idx]
            
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

                # Paper's loss: ||I_ti(φ_ti→T(P)) - I_T(P)||²
                # Sample I_ti at deformed locations φ_ti→T(P)
                I_ti_warped = sample_intensity(
                    frames_tensor[frame_idx].unsqueeze(0).unsqueeze(0), warped_points
                )
                # Sample I_T at original locations P
                I_T_original = sample_intensity(
                    target_frame.unsqueeze(0).unsqueeze(0), sampled_coords
                )

                recon_losses.append(torch.mean((I_ti_warped - I_T_original) ** 2))

            recon_loss = torch.mean(torch.stack(recon_losses))

        # Cycle consistency loss R_cycle
        if epoch == 0:
            print("   🔄 Computing cycle consistency loss...")
            
        # Sample points for cycle loss computation
        cycle_idx = balanced_sample_indices(min(sample_points, 5000))  # Reduce for memory
        cycle_coords = spatial_coords_torch[cycle_idx]
        
        full_trajectories = integrate_velocity_to_deformation(
            siren_model, cycle_coords, torch.from_numpy(temporal_coords).float().to(device)
        )
        cycle_loss = torch.mean((cycle_coords - full_trajectories[-1]) ** 2)

        # Total loss: reconstruction + λ * cycle
        total_loss = recon_loss + lambda_cycle * cycle_loss
        total_loss.backward()
        optimizer.step()

        print(f"[Epoch {epoch}/{num_epochs}] Total={total_loss.item():.6f}, "f"Recon={recon_loss.item():.6f}, Cycle={cycle_loss.item():.6f}")

    print("Training complete.")
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
        num_epochs=1000,
        sample_points=10000,  # Reduced for memory efficiency
        lambda_cycle=0.1,
        device=device,
        use_full_volume_loss=False  # Use point sampling for stability
    )
    print("Training complete.")

    print("=== Step 5: Generating mesh vertex trajectories ===")
    # Use initial mesh vertices (not all spatial coordinates) for trajectory computation
    mesh_vertices_tensor = torch.from_numpy(initial_vertices_normalized).float().to(device)
    time_tensor = torch.from_numpy(temporal_coords).float().to(device)
    
    print(f"Computing trajectories for {mesh_vertices_tensor.shape[0]} mesh vertices...")
    mesh_trajectories = integrate_velocity_to_deformation(trained_model, mesh_vertices_tensor, time_tensor)
    print(f"Mesh trajectories computed: {mesh_trajectories.shape} (T, N_vertices, 3)")

    print("Plotting sample mesh vertex trajectories...")
    traj_plot_path = os.path.join(visualization_path, "mesh_vertex_trajectories.png")
    # Use vertex indices instead of arbitrary spatial indices
    num_vertices = mesh_vertices_tensor.shape[0]
    sample_vertex_indices = [0, min(num_vertices//4, num_vertices-1), min(num_vertices//2, num_vertices-1)]
    plot_point_trajectories(mesh_trajectories, sample_vertex_indices, save_path=traj_plot_path)

    print("=== Step 6: Computing and plotting volume comparison ===")
    vol_plot_path = os.path.join(visualization_path, "volume_comparison.png")
    compute_and_plot_volumes(frames, mesh_trajectories, mesh_faces, voxel_spacing, frames[0].shape, save_path=vol_plot_path)