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
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from trimesh.proximity import ProximityQuery

from dataloader import CTSequenceDataset

# =========================================================
# 1. Mesh + Volume Utilities
# =========================================================
def compute_mesh_volume(vertices, faces):
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    return abs(mesh.volume)

def compute_frame_volume_gt(frame, voxel_spacing, level=0.5):
    verts, faces, _, _ = measure.marching_cubes(frame, level=level, spacing=voxel_spacing)
    return abs(compute_mesh_volume(verts, faces))

# =========================================================
# 2. SIREN Velocity Model
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
# 3. ODE Integration
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
    return torch.clamp(trajectories, -1.0, 1.0)

# =========================================================
# 4. Intensity Sampling
# =========================================================
def sample_intensity(volume, coords):
    """
    Sample intensity values from a 3D volume using grid coordinates.
    
    Args:
        volume: 3D tensor of shape (C, H, W, D) or (H, W, D)
        coords: 2D tensor of shape (N, 3) with normalized coordinates in [-1, 1]
    
    Returns:
        1D tensor of sampled intensities
    """
    # Ensure volume has batch and channel dimensions
    if volume.dim() == 3:
        volume = volume.unsqueeze(0)  # Add batch dimension
    if volume.dim() == 4:
        volume = volume.unsqueeze(0)  # Add channel dimension if needed
    
    # Ensure coords are in the correct format for grid_sample
    # grid_sample expects coordinates in (x, y, z) format
    if coords.shape[-1] != 3:
        raise ValueError(f"Expected 3D coordinates, got shape {coords.shape}")
    
    # Reshape coords for grid_sample: (N, 3) -> (1, N, 1, 1, 3)
    coords = coords.view(1, -1, 1, 1, 3)
    
    # grid_sample expects coordinates in [-1, 1] range
    sampled = F.grid_sample(volume, coords, align_corners=True, mode='bilinear', padding_mode='border')
    return sampled.view(-1)

# =========================================================
# 5. Volume Plotting (CSV + Plot) with sklearn metrics
# =========================================================
def plot_and_save_volumes(gt_volumes, pred_volumes, save_path):
    frames_idx = np.arange(len(gt_volumes))
    
    # Calculate sklearn-based metrics
    mse = mean_squared_error(gt_volumes, pred_volumes)
    mae = mean_absolute_error(gt_volumes, pred_volumes)
    r2 = r2_score(gt_volumes, pred_volumes)
    
    # Calculate relative errors
    volume_diff_mm3 = np.array(pred_volumes) - np.array(gt_volumes)
    volume_diff_percent = (volume_diff_mm3 / np.array(gt_volumes)) * 100
    
    volume_data = pd.DataFrame({
        'frame': frames_idx,
        'time_phase': frames_idx / (len(gt_volumes)-1),
        'gt_volume_mm3': gt_volumes,
        'predicted_volume_mm3': pred_volumes,
        'volume_diff_mm3': volume_diff_mm3,
        'volume_diff_percent': volume_diff_percent
    })
    
    # Add summary statistics
    stats_summary = pd.DataFrame({
        'metric': ['MSE', 'MAE', 'R²', 'Mean_GT_Volume', 'Mean_Pred_Volume', 
                   'Std_GT_Volume', 'Std_Pred_Volume', 'Mean_Abs_Error_%', 'Max_Abs_Error_%'],
        'value': [mse, mae, r2, np.mean(gt_volumes), np.mean(pred_volumes),
                 np.std(gt_volumes), np.std(pred_volumes), 
                 np.mean(np.abs(volume_diff_percent)), np.max(np.abs(volume_diff_percent))]
    })
    
    csv_path = save_path.replace('.png', '.csv')
    stats_csv_path = save_path.replace('.png', '_stats.csv')
    
    volume_data.to_csv(csv_path, index=False, float_format='%.3f')
    stats_summary.to_csv(stats_csv_path, index=False, float_format='%.6f')
    
    print(f"[INFO] Saved volume CSV to {csv_path}")
    print(f"[INFO] Saved statistics CSV to {stats_csv_path}")
    print(f"[INFO] Volume prediction metrics: MSE={mse:.3f}, MAE={mae:.3f}, R²={r2:.6f}")

    plt.figure(figsize=(12, 8))
    
    # Main plot
    plt.subplot(2, 2, 1)
    plt.plot(frames_idx, gt_volumes, 'o-', label='Reference', color='orange', linewidth=2)
    plt.plot(frames_idx, pred_volumes, 'o-', label='Predicted', color='blue', linewidth=2)
    plt.xlabel("Frame")
    plt.ylabel("Volume (mm³)")
    plt.title(f"Volume Comparison (R²={r2:.3f})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Error plot
    plt.subplot(2, 2, 2)
    plt.plot(frames_idx, volume_diff_percent, 'ro-', linewidth=2)
    plt.xlabel("Frame")
    plt.ylabel("Error (%)")
    plt.title(f"Relative Error (MAE={mae:.3f}mm³)")
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # Scatter plot
    plt.subplot(2, 2, 3)
    plt.scatter(gt_volumes, pred_volumes, alpha=0.7, s=50)
    min_vol, max_vol = min(min(gt_volumes), min(pred_volumes)), max(max(gt_volumes), max(pred_volumes))
    plt.plot([min_vol, max_vol], [min_vol, max_vol], 'r--', label='Perfect prediction')
    plt.xlabel("Ground Truth Volume (mm³)")
    plt.ylabel("Predicted Volume (mm³)")
    plt.title("Prediction vs Ground Truth")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Histogram of errors
    plt.subplot(2, 2, 4)
    plt.hist(volume_diff_percent, bins=10, alpha=0.7, edgecolor='black')
    plt.xlabel("Error (%)")
    plt.ylabel("Frequency")
    plt.title("Error Distribution")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[INFO] Saved enhanced volume plot to {save_path}")



def compute_hausdorff_distance(vertices_pred, faces_pred, vertices_gt, faces_gt):
    """
    Compute symmetric Hausdorff Distance (HSD) between two meshes.

    Args:
        vertices_pred: Nx3 numpy array of predicted vertices
        faces_pred: Mx3 numpy array of predicted faces
        vertices_gt: Nx3 numpy array of ground-truth vertices
        faces_gt: Mx3 numpy array of ground-truth faces

    Returns:
        float: Symmetric Hausdorff Distance (in same units as input, e.g., mm)
    """
    mesh_pred = trimesh.Trimesh(vertices=vertices_pred, faces=faces_pred, process=False)
    mesh_gt = trimesh.Trimesh(vertices=vertices_gt, faces=faces_gt, process=False)

    # Closest point queries
    pq_pred = ProximityQuery(mesh_gt)
    dist_pred_to_gt = pq_pred.vertex(mesh_pred.vertices)[0]  # distances

    pq_gt = ProximityQuery(mesh_pred)
    dist_gt_to_pred = pq_gt.vertex(mesh_gt.vertices)[0]

    # Symmetric HSD
    return max(dist_pred_to_gt.max(), dist_gt_to_pred.max())

def save_hsd_metrics(hsd_values, save_path):
    hsd_data = pd.DataFrame({
        'frame': np.arange(len(hsd_values)),
        'hsd_mm': hsd_values
    })
    hsd_csv_path = save_path.replace('.png', '_hsd.csv')
    hsd_data.to_csv(hsd_csv_path, index=False, float_format='%.3f')
    print(f"[INFO] Saved HSD CSV to {hsd_csv_path}")
    print(f"[INFO] HSD mean: {np.mean(hsd_values):.3f} mm | max: {np.max(hsd_values):.3f} mm")

# =========================================================
# 6. Training Loop (with debug prints) - PAPER IMPLEMENTATION
# =========================================================

def sweep_lambda_cycle(
    lambda_values,
    frames,
    spatial_coords,
    temporal_coords,
    mesh_vertices,
    mesh_faces,
    voxel_spacing,
    mesh_scaler,
    num_epochs=500,
    sample_points=5000,
    device='cuda',
    save_dir='sweep_results'
):
    """
    Perform a hyperparameter sweep over lambda_cycle values and save comparison plots + metrics.
    """
    os.makedirs(save_dir, exist_ok=True)
    results = []

    for lam in lambda_values:
        print(f"\n[SWEEP] Testing λ_cycle = {lam}")
        
        # Initialize new SIREN model each sweep
        siren_model = SIRENVelocityField(hidden_dim=256)
        
        # Train model
        trained_model, gt_volumes = train_inr_model(
            siren_model, frames, spatial_coords, temporal_coords,
            mesh_vertices, mesh_faces, voxel_spacing, mesh_scaler,
            num_epochs=num_epochs, sample_points=sample_points,
            lambda_cycle=lam, lambda_volume=0.0,
            device=device
        )

        # Predict volumes
        mesh_vertices_tensor = torch.from_numpy(mesh_vertices).float().to(device)
        time_tensor = torch.from_numpy(temporal_coords).float().to(device)
        with torch.no_grad():
            mesh_trajectories = integrate_velocity_to_deformation(trained_model, mesh_vertices_tensor, time_tensor)

        pred_volumes = []
        for t in range(len(frames)):
            verts_norm = mesh_trajectories[t].cpu().numpy()
            verts_original = mesh_scaler.inverse_transform(verts_norm)
            pred_volumes.append(compute_mesh_volume(verts_original, mesh_faces))

        # Save plot + CSV
        plot_path = os.path.join(save_dir, f"volume_comparison_lambda{lam}.png")
        plot_and_save_volumes(gt_volumes, pred_volumes, plot_path)

        # Compute HSD
        hsd_values = []
        for t in range(len(frames)):
            verts_gt, faces_gt, _, _ = measure.marching_cubes(frames[t], level=0.5, spacing=voxel_spacing)
            verts_pred = mesh_scaler.inverse_transform(mesh_trajectories[t].cpu().numpy())
            hsd = compute_hausdorff_distance(verts_pred, mesh_faces, verts_gt, faces_gt)
            hsd_values.append(hsd)

        mean_hsd = float(np.mean(hsd_values))
        print(f"[RESULT] λ_cycle={lam} → Mean HSD={mean_hsd:.3f} mm")

        # Save summary
        results.append({
            'lambda_cycle': lam,
            'mean_hsd_mm': mean_hsd,
            'volume_r2': r2_score(gt_volumes, pred_volumes),
            'mae_mm3': mean_absolute_error(gt_volumes, pred_volumes)
        })

    # Save sweep results to CSV
    df_results = pd.DataFrame(results)
    sweep_csv = os.path.join(save_dir, "sweep_summary.csv")
    df_results.to_csv(sweep_csv, index=False, float_format='%.4f')
    print(f"[SWEEP] Summary saved to {sweep_csv}")

    return df_results


def train_inr_model(
    siren_model, frames, spatial_coords, temporal_coords,
    mesh_vertices, mesh_faces, voxel_spacing, mesh_scaler,
    num_epochs=1000, sample_points=10000,
    lambda_cycle=0.1, lambda_volume=0.0,
    device='cuda'
):
    print("=== 论文实现：图像重建损失 + 周期一致性 ===")
    siren_model = siren_model.to(device)
    optimizer = optim.Adam(siren_model.parameters(), lr=3e-5, weight_decay=1e-6)

    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.8, patience=30
    )

    # Convert data
    spatial_coords_torch = torch.from_numpy(spatial_coords).float().to(device)
    frames_tensor = [torch.from_numpy(frame).float().to(device) for frame in frames]
    time_tensor = torch.from_numpy(temporal_coords).float().to(device)

    # Precompute GT volumes for final evaluation only
    gt_volumes = [compute_frame_volume_gt(f, voxel_spacing) for f in frames]
    print(f"[DEBUG] GT volume range: {min(gt_volumes):.1f}-{max(gt_volumes):.1f} mm³")

    # 稳定的采样策略：固定采样模式
    def sample_coordinates_stable(epoch, sample_points):
        """稳定的采样策略：每10个epoch才轻微调整采样."""
        total_points = spatial_coords_torch.shape[0]
        if sample_points >= total_points:
            return torch.arange(total_points, device=device)
        
        # 每10个epoch才变化一次采样，提高训练稳定性
        stable_seed = 42 + (epoch // 10)
        np.random.seed(stable_seed)
        indices = np.random.choice(total_points, sample_points, replace=False)
        return torch.from_numpy(indices).to(device)

    start_time = time.time()

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # Sample points using sklearn
        idx = sample_coordinates_stable(epoch, sample_points)
        sampled_coords = spatial_coords_torch[idx]

        if epoch in [0, 500, 999]:
            print(f"[DEBUG] Epoch {epoch} sampled coords (first 3): {sampled_coords[:3].cpu().numpy()}")

        # ===== 论文的核心重建损失实现 =====
        # 论文公式: min Σ ||I_ti ∘ φ_ti→T - I_T||² + λ R_cycle
        
        # 获取参考帧（最后一帧）
        reference_frame = frames_tensor[-1]  # I_T
        
        # 计算所有其他帧到参考帧的变形
        reconstruction_losses = []
        for t in range(len(frames) - 1):  # 除了最后一帧
            current_frame = frames_tensor[t]  # I_ti
            
            # 计算当前时刻到参考时刻的变形
            # φ_ti→T = φ_T ∘ φ_ti^(-1)
            # 我们需要计算从当前时刻到参考时刻的变形
            current_time = time_tensor[t]
            reference_time = time_tensor[-1]
            
            # 计算变形轨迹：从当前时刻到参考时刻
            time_points = torch.linspace(current_time, reference_time, 10, device=device)
            deformed_coords = integrate_velocity_to_deformation(siren_model, sampled_coords, time_points)
            
            # 获取变形后的坐标（最后一帧的坐标）
            final_deformed_coords = deformed_coords[-1]
            
            # 采样变形后的图像强度
            if epoch == 0 and t == 0:
                print(f"[DEBUG] Volume shape: {current_frame.shape}")
                print(f"[DEBUG] Coords shape: {final_deformed_coords.shape}")
                print(f"[DEBUG] Coords range: [{final_deformed_coords.min().item():.3f}, {final_deformed_coords.max().item():.3f}]")
            
            deformed_intensities = sample_intensity(current_frame, final_deformed_coords)
            reference_intensities = sample_intensity(reference_frame, final_deformed_coords)
            
            # 重建损失：||I_ti ∘ φ_ti→T - I_T||²
            frame_recon_loss = torch.mean((deformed_intensities - reference_intensities) ** 2)
            reconstruction_losses.append(frame_recon_loss)
        
        # 总重建损失
        recon_loss = torch.mean(torch.stack(reconstruction_losses))

        # ===== 周期一致性损失 =====
        # 论文公式: R_cycle = (1/n) Σ ||P_0,i - φ_T(P_0,i)||²
        
        # 计算从起始时刻到结束时刻的完整轨迹
        full_trajectories = integrate_velocity_to_deformation(siren_model, sampled_coords, time_tensor)
        
        # 起始位置和结束位置
        initial_positions = full_trajectories[0]
        final_positions = full_trajectories[-1]
        
        # 周期一致性损失
        cycle_loss = torch.mean((initial_positions - final_positions) ** 2)

        # ===== 组合损失 =====
        # 论文公式: min [Σ ||I_ti ∘ φ_ti→T - I_T||²] + λ R_cycle
        total_loss = recon_loss + lambda_cycle * cycle_loss
        total_loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(siren_model.parameters(), max_norm=1.0)
        optimizer.step()

        # 学习率调度器步进
        scheduler.step(total_loss)

        # 详细的损失监控
        if epoch % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"[Epoch {epoch:3d}] 论文实现 - Total={total_loss.item():.6f} "
                  f"Recon={recon_loss.item():.6f} Cycle={cycle_loss.item():.6f} "
                  f"LR={current_lr:.2e}")

        # 早停机制（可选）
        if epoch == 0:
            best_loss = total_loss.item()
            best_epoch = 0
        elif total_loss.item() < best_loss:
            best_loss = total_loss.item()
            best_epoch = epoch

    print(f"论文实现训练完成，用时 {(time.time()-start_time)/60:.1f} 分钟")
    print(f"最佳损失: {best_loss:.4f} (epoch {best_epoch})")
    return siren_model, gt_volumes

# =========================================================
# 7. Main
# =========================================================
if __name__ == "__main__":
    data_path = "data/4007775_aneurysm/nnunet_outputs_pp"
    visualization_path = "data/4007775_aneurysm/visualizations"
    os.makedirs(visualization_path, exist_ok=True)

    # Load data
    dataset = CTSequenceDataset(data_path, num_frames=20)
    frames, temporal_coords = dataset.get_all_frames()
    spatial_coords = dataset.get_spatial_coords()
    voxel_spacing = dataset.get_voxel_spacing()

    # Extract initial mesh (0pct)
    initial_nifti_path = os.path.join(data_path, "0pct.nii.gz")
    nii = nib.load(initial_nifti_path)
    data = nii.get_fdata().astype(np.float32)
    mesh_vertices, mesh_faces, _, _ = measure.marching_cubes(data, level=0.5, spacing=voxel_spacing)

    # Use sklearn MinMaxScaler for mesh vertex normalization
    mesh_scaler = MinMaxScaler(feature_range=(-1, 1))
    mesh_vertices_norm = mesh_scaler.fit_transform(mesh_vertices)

    # Train model with fixed parameters
    siren_model = SIRENVelocityField(hidden_dim=256)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # # Sweep over multiple λ_cycle values
    # lambda_values = [0.01, 0.05, 0.1, 0.5, 1.0]
    # sweep_results = sweep_lambda_cycle(
    #     lambda_values,
    #     frames,
    #     spatial_coords,
    #     temporal_coords,
    #     mesh_vertices_norm,
    #     mesh_faces,
    #     voxel_spacing,
    #     mesh_scaler,
    #     num_epochs=500,    # shorter for sweep
    #     sample_points=5300,
    #     device=device,
    #     save_dir=os.path.join(visualization_path, "lambda_sweep")
    # )

    # print(sweep_results)
    
    trained_model, gt_volumes = train_inr_model(
        siren_model, frames, spatial_coords, temporal_coords,
        mesh_vertices_norm, mesh_faces, voxel_spacing, mesh_scaler,
        num_epochs=500, sample_points=5300,  # 论文实现训练
        lambda_cycle=0.01, lambda_volume=0.0,  # 使用论文的周期权重
        device=device
    )

    # Predict volumes using sklearn scaler for inverse transform
    mesh_vertices_tensor = torch.from_numpy(mesh_vertices_norm).float().to(device)
    time_tensor = torch.from_numpy(temporal_coords).float().to(device)
    with torch.no_grad():
        mesh_trajectories = integrate_velocity_to_deformation(trained_model, mesh_vertices_tensor, time_tensor)

    pred_volumes = []
    for t in range(len(frames)):
        # Use sklearn scaler for inverse transform
        verts_norm = mesh_trajectories[t].cpu().numpy()
        verts_original = mesh_scaler.inverse_transform(verts_norm)
        
        # 使用原始坐标计算体积
        pred_volumes.append(compute_mesh_volume(verts_original, mesh_faces))

    # Plot + save with paper implementation
    vol_plot_path = os.path.join(visualization_path, "21volume_comparison_paper_implementation.png")
    plot_and_save_volumes(gt_volumes, pred_volumes, vol_plot_path)

    # Compute HSD for each frame
    hsd_values = []
    for t in range(len(frames)):
        # Ground truth mesh for frame t
        verts_gt, faces_gt, _, _ = measure.marching_cubes(frames[t], level=0.5, spacing=voxel_spacing)

        # Predicted mesh (inverse transform normalized trajectory)
        verts_pred = mesh_scaler.inverse_transform(mesh_trajectories[t].cpu().numpy())

        hsd = compute_hausdorff_distance(verts_pred, mesh_faces, verts_gt, faces_gt)
        hsd_values.append(hsd)

    print(f"[INFO] HSD range: min={np.min(hsd_values):.3f} mm, max={np.max(hsd_values):.3f} mm, mean={np.mean(hsd_values):.3f} mm")
    hsd_plot_path = os.path.join(visualization_path, "21hsd.png")
    save_hsd_metrics(hsd_values, hsd_plot_path)