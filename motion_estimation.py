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
    verts, faces, normals, _ = measure.marching_cubes(data, level=level, spacing=spacing)
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
def warp_volume(volume, deformation, volume_shape):
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
def visualize_mesh_overlay(volume, mesh_vertices, mesh_faces):
    faces_flat = np.hstack([np.full((mesh_faces.shape[0],1),3), mesh_faces]).flatten()
    pv_mesh = pv.PolyData(mesh_vertices, faces_flat)
    plotter = pv.Plotter()
    plotter.add_volume(volume, cmap="gray", opacity="linear")
    plotter.add_mesh(pv_mesh, color="red", opacity=0.5)
    plotter.show()

def plot_point_trajectories(trajectories, point_indices):
    traj_np = trajectories.detach().cpu().numpy()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for idx in point_indices:
        ax.plot(traj_np[:, idx, 0], traj_np[:, idx, 1], traj_np[:, idx, 2], marker='o')
    ax.set_title("Point trajectories across cycle")
    plt.show()


def compute_mesh_volume(vertices, faces):
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    return mesh.volume

def plot_volume_comparison(volumes_gt, volumes_pred, title="Volume Comparison"):
    frames = np.arange(len(volumes_gt))
    plt.figure(figsize=(6,4))
    plt.plot(frames, volumes_gt, 'o-', color='orange', label="Reference")
    plt.plot(frames, volumes_pred, 'o-', color='blue', label="Predicted")
    plt.xlabel("Frame (time)")
    plt.ylabel("Volume (mm³)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def compute_and_plot_volumes(volumes_4d, trajectories, mesh_faces):
    """
    Compute GT and predicted volumes over 20 frames and plot comparison.
    Args:
        volumes_4d: numpy array (T,H,W,D) of binary segmentations
        trajectories: torch.Tensor (T,N,3) predicted vertex positions
        mesh_faces: numpy array (F,3)
    """
    num_frames = volumes_4d.shape[0]
    gt_volumes = []
    pred_volumes = []

    # 1. GT volume: marching cubes per frame
    for t in range(num_frames):
        verts_gt, faces_gt, normals_gt, _ = measure.marching_cubes(volumes_4d[t], level=0.5)
        gt_volumes.append(compute_mesh_volume(verts_gt, faces_gt))

    # 2. Predicted volume: use warped vertices from trajectories
    for t in range(num_frames):
        verts_pred = trajectories[t].detach().cpu().numpy()
        pred_volumes.append(compute_mesh_volume(verts_pred, mesh_faces))

    # 3. Plot
    plot_volume_comparison(gt_volumes, pred_volumes)

# =========================================================
# 9. Training Loop
# =========================================================
def train_inr_model(
    siren_model,
    volumes_4d,
    spatial_coords,
    temporal_coords,
    num_epochs=1000,
    sample_points=10000,
    lambda_cycle=0.1,
    device='cuda'
):
    print(f"Starting training for {num_epochs} epochs with {sample_points} sample points per epoch...")
    siren_model = siren_model.to(device)
    siren_model.train()
    optimizer = optim.Adam(siren_model.parameters(), lr=3e-5)

    T,H,W,D = volumes_4d.shape
    volumes_torch = torch.tensor(volumes_4d, dtype=torch.float32, device=device).unsqueeze(1)
    spatial_coords = torch.tensor(spatial_coords, dtype=torch.float32, device=device)

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # Random sample points
        idx = torch.randint(0, spatial_coords.shape[0], (sample_points,), device=device)
        sampled_coords = spatial_coords[idx]

        # Integrate velocity → deformation
        time_points = torch.tensor(temporal_coords, dtype=torch.float32, device=device)
        trajectories = integrate_velocity_to_deformation(siren_model, sampled_coords, time_points)

        # Warp volume I0 to IT
        warped_points = trajectories[-1]
        warped_volume = warp_volume(volumes_torch[0].unsqueeze(0), warped_points, (H,W,D))
        target_volume = volumes_torch[-1].unsqueeze(0)

        # Losses
        recon_loss = reconstruction_loss(warped_volume, target_volume)
        cycle_loss = cycle_consistency_loss(sampled_coords, trajectories)
        total_loss = recon_loss + lambda_cycle * cycle_loss

        total_loss.backward()
        optimizer.step()

        if epoch % 50 == 0 or epoch == num_epochs - 1:
            psnr_val = compute_psnr(warped_volume, target_volume)
            print(f"[Epoch {epoch}/{num_epochs}] Loss={total_loss.item():.6f}, PSNR={psnr_val:.2f} dB")

    print("Finished training.")
    return siren_model


# =========================================================
# 10. Main Pipeline
# =========================================================
if __name__ == "__main__":
    data_path = "data/4007775_aneurysm/nnunet_outputs_pp"
    mesh_output_path = "data/4007775_aneurysm/meshes"

    print("=== Step 1: Loading and normalizing 4D CTA data ===")
    volumes_4d = load_4d_cta(data_path)
    print(f"Loaded volumes with shape: {volumes_4d.shape} (T,H,W,D)")

    print("Normalizing spatial and temporal coordinates...")
    spatial_coords = normalize_spatial_coordinates(volumes_4d.shape[1:])
    temporal_coords = normalize_temporal_coordinates(volumes_4d.shape[0])
    print(f"Spatial coords: {spatial_coords.shape}, Temporal coords: {temporal_coords.shape}")

    print("=== Step 2: Extracting meshes from segmentations ===")
    extract_all_meshes(data_path, mesh_output_path)
    print(f"Meshes saved to: {mesh_output_path}")

    print("Loading initial mesh faces (0pct) for predicted volume computation...")
    first_mesh_path = os.path.join(mesh_output_path, "0pct.ply")
    initial_mesh = trimesh.load(first_mesh_path, process=False)
    mesh_faces = np.array(initial_mesh.faces)
    print(f"Mesh faces shape: {mesh_faces.shape}")

    print("=== Step 3: Initializing SIREN model ===")
    siren_model = SIRENVelocityField(hidden_dim=256)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    print("=== Step 4: Training SIREN model ===")
    trained_model = train_inr_model(
        siren_model,
        volumes_4d,
        spatial_coords,
        temporal_coords,
        num_epochs=1000,
        sample_points=10000,
        lambda_cycle=0.1,
        device=device
    )
    print("Training complete.")

    print("=== Step 5: Generating deformation trajectories ===")
    points_tensor = torch.tensor(spatial_coords, dtype=torch.float32, device=device)
    time_tensor = torch.tensor(temporal_coords, dtype=torch.float32, device=device)
    trajectories = integrate_velocity_to_deformation(trained_model, points_tensor, time_tensor)
    print(f"Trajectories computed: {trajectories.shape} (T, N, 3)")

    print("Plotting sample point trajectories (sanity check)...")
    plot_point_trajectories(trajectories, [0, 500, 1000])

    print("Computing and plotting volume comparison (Pred vs GT)...")
    compute_and_plot_volumes(volumes_4d, trajectories, mesh_faces)
    print("Pipeline complete.")