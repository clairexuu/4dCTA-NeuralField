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

def integrate_velocity_to_deformation(siren_model, points, time_points, method='rk4'):
    ode_func = VelocityFieldODE(siren_model)
    trajectories = odeint(ode_func, points, time_points, method=method)
    return torch.clamp(trajectories, -1.0, 1.0)

# =========================================================
# 4. Intensity Sampling
# =========================================================
def sample_intensity(volume, coords):
    coords = coords.view(1, -1, 1, 1, 3)
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

# =========================================================
# 6. Training Loop (with debug prints)
# =========================================================
def train_inr_model(
    siren_model, frames, spatial_coords, temporal_coords,
    mesh_vertices, mesh_faces, voxel_spacing, mesh_scaler,
    num_epochs=1000, sample_points=10000,
    lambda_cycle=0.1, lambda_volume=0.0,  # 完全移除体积约束
    device='cuda'
):
    print("=== 纯论文方法训练 (无体积约束) ===")
    siren_model = siren_model.to(device)
    optimizer = optim.Adam(siren_model.parameters(), lr=1e-5)

    # Convert data
    spatial_coords_torch = torch.from_numpy(spatial_coords).float().to(device)
    frames_tensor = [torch.from_numpy(frame).float().to(device) for frame in frames]
    target_frame = frames_tensor[-1]
    time_tensor = torch.from_numpy(temporal_coords).float().to(device)

    # Precompute GT volumes for final evaluation only
    gt_volumes = [compute_frame_volume_gt(f, voxel_spacing) for f in frames]
    print(f"[DEBUG] GT volume range: {min(gt_volumes):.1f}-{max(gt_volumes):.1f} mm³ (仅用于最终评估)")
    
    # Use sklearn for reproducible sampling
    def sample_coordinates_sklearn(epoch, sample_points):
        """Use sklearn shuffle for reproducible coordinate sampling."""
        total_points = spatial_coords_torch.shape[0]
        if sample_points >= total_points:
            return torch.arange(total_points, device=device)
        
        # Use epoch as random state for reproducible but varied sampling
        indices_np = np.arange(total_points)
        _, sampled_indices = shuffle(
            spatial_coords, indices_np, 
            n_samples=sample_points, 
            random_state=epoch
        )
        return torch.from_numpy(sampled_indices).to(device)

    start_time = time.time()

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # Sample points using sklearn
        idx = sample_coordinates_sklearn(epoch, sample_points)
        sampled_coords = spatial_coords_torch[idx]

        if epoch in [0, 500, 999]:
            print(f"[DEBUG] Epoch {epoch} sampled coords (first 3): {sampled_coords[:3].cpu().numpy()}")

        # Integrate forward trajectories
        trajectories = integrate_velocity_to_deformation(siren_model, sampled_coords, time_tensor)

        # Reconstruction loss - 按照论文的正确理解
        # 论文Equation 1: min Σ ||I_{t_i} ∘ φ_{t_i→T} - I_T||²
        # 理解：同一物质点在不同时刻的图像值应该一致
        recon_losses = []
        target_frame = frames_tensor[-1]  # 最后一帧作为目标 I_T
        
        for i in range(len(frames)-1):  # i = 0 to N-2
            # trajectories[i]: 采样点在时刻t_i的位置（相当于φ_{t_i}(sampled_coords)）
            # 在时刻t_i的位置采样第i帧图像
            I_ti_at_deformed = sample_intensity(frames_tensor[i].unsqueeze(0).unsqueeze(0), trajectories[i])
            
            # 在原始位置采样目标帧图像（作为参考）
            I_T_at_original = sample_intensity(target_frame.unsqueeze(0).unsqueeze(0), sampled_coords)
            
            # 损失：同一物质点在不同时刻的图像值应该相同
            recon_losses.append(torch.sum((I_ti_at_deformed - I_T_at_original) ** 2))
            
        recon_loss = torch.sum(torch.stack(recon_losses))

        # Cycle consistency loss - 按照论文的R_cycle定义
        # R_cycle = (1/n) Σ ||P_0,i - φ_T(P_0,i)||²
        # 含义：经过一个完整周期后应该回到原始位置
        final_positions = trajectories[-1]  # φ_T(P_0,i)
        cycle_loss = torch.mean((sampled_coords - final_positions) ** 2)

        # 纯论文方法：只有重建损失 + 周期一致性损失
        total_loss = recon_loss + lambda_cycle * cycle_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(siren_model.parameters(), max_norm=1.0)
        optimizer.step()

        # 简化的损失监控 - 只显示论文中的两个损失项
        if epoch % 10 == 0:
            print(f"[Epoch {epoch}] 纯论文方法 - Total={total_loss.item():.4f} "
                  f"Recon={recon_loss.item():.1f} Cycle={cycle_loss.item():.6f} λ_cyc={lambda_cycle:.4f}")

    print(f"纯论文方法训练完成，用时 {(time.time()-start_time)/60:.1f} 分钟")
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
    
    print("[INFO] ===== 纯论文方法实现 =====")
    print("[INFO] 论文Title: Neural Fields for Continuous Periodic Motion Estimation in 4D Cardiovascular Imaging")
    print("[INFO] ✅ 时间编码: f(t) = (cos(2πt), sin(2πt)) - 已正确实现")
    print("[INFO] ✅ SIREN网络: 使用正弦激活函数 - 已正确实现") 
    print("[INFO] ✅ 损失函数: Σ||I_ti ∘ φ_(ti→T) - I_T||² + λR_cycle - 已按论文修正")
    print("[INFO] ✅ 周期一致性: R_cycle = (1/n)Σ||P_0 - φ_T(P_0)||² - 已按论文实现")
    print("[INFO] ✅ ODE积分: 使用torchdiffeq进行速度场积分 - 已正确实现")
    print("[INFO] ❌ 体积约束: 已完全移除 - 原论文未提及")
    print("[INFO] 🎯 目标: 测试纯论文方法的运动估计效果")
    
    trained_model, gt_volumes = train_inr_model(
        siren_model, frames, spatial_coords, temporal_coords,
        mesh_vertices_norm, mesh_faces, voxel_spacing, mesh_scaler,
        num_epochs=60, sample_points=10000,
        lambda_cycle=0.5, lambda_volume=0.0,  # 纯论文方法：无体积约束
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

    # Plot + save with pure paper method (no volume constraints)
    vol_plot_path = os.path.join(visualization_path, "10volume_comparison.png")
    plot_and_save_volumes(gt_volumes, pred_volumes, vol_plot_path)