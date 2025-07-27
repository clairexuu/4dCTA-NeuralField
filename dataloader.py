import numpy as np
import os
import nibabel as nib

class CTSequenceDataset:
    """
    Loads a temporal sequence of 3D CT frames (segmentation or intensity).
    Matches paper's conceptual flow: treat each frame independently.
    """

    def __init__(self, dataset_path, num_frames=20):
        """
        Args:
            dataset_path: Path to folder containing frames like 0pct.nii.gz ... 95pct.nii.gz
            num_frames: Number of frames in the cycle (default 20 for 0-95pct)
        """
        self.dataset_path = dataset_path
        self.num_frames = num_frames
        self.frames = []        # list of 3D numpy arrays
        self.spatial_coords = None  # normalized [-1,1]^3 coords
        self.temporal_coords = None # normalized [0,1] time coords
        self.voxel_spacing = None   # mm scaling

        self._load_frames()
        self._compute_coords()

    def _load_frames(self):
        """Load each 3D frame independently without stacking."""
        for i in range(self.num_frames):
            filename = f"{i*5}pct.nii.gz"
            file_path = os.path.join(self.dataset_path, filename)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Missing file: {file_path}")

            nii = nib.load(file_path)
            data = nii.get_fdata().astype(np.float32)

            # Normalize intensities per paper (0-1)
            min_val, max_val = np.min(data), np.max(data)
            data = (data - min_val) / (max_val - min_val + 1e-8)

            # Save volume
            self.frames.append(data)

            # Get voxel spacing (assume all frames same)
            if self.voxel_spacing is None:
                self.voxel_spacing = nii.header.get_zooms()

    def _compute_coords(self):
        """Compute spatial and temporal normalized coordinates following paper exactly."""
        # Paper's spatial normalization: coordinates normalized to [-1,1]^3
        # Using consistent grid generation as in paper's spatial domain Ω⊂R^3
        H, W, D = self.frames[0].shape
        
        # Generate spatial coordinates in [-1,1]^3 following paper's normalization
        # Note: using consistent indexing with paper's coordinate system
        z = np.linspace(-1, 1, H)  # z-axis (depth)
        y = np.linspace(-1, 1, W)  # y-axis (height) 
        x = np.linspace(-1, 1, D)  # x-axis (width)
        
        # Create meshgrid with 'ij' indexing for consistency with paper
        zz, yy, xx = np.meshgrid(z, y, x, indexing='ij')
        
        # Stack coordinates as (x, y, z) following paper's coordinate convention
        # Paper uses P̂ = (x, y, z) ∈ R^3
        self.spatial_coords = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)

        # Temporal coords normalized to [0,1] as per paper: time horizon T
        # Paper: "time points t0, t1, ..., tN−1 were equally spaced and normalized to the [0,1] interval"
        self.temporal_coords = np.linspace(0, 1, self.num_frames, dtype=np.float32)

    def get_frame(self, idx):
        """Return frame volume (numpy) and its normalized time coordinate."""
        return self.frames[idx], self.temporal_coords[idx]

    def get_all_frames(self):
        """Return list of all frames and time coordinates."""
        return self.frames, self.temporal_coords

    def get_spatial_coords(self):
        """Return normalized spatial coordinates ([-1,1]^3) as per paper."""
        return self.spatial_coords
    
    def get_spatial_coords_grid(self):
        """Return spatial coordinates in grid format (H, W, D, 3) for volume operations."""
        H, W, D = self.frames[0].shape
        return self.spatial_coords.reshape(H, W, D, 3)
    
    def sample_points_uniformly(self, num_points):
        """Sample points uniformly from spatial domain as per paper's sampling strategy."""
        total_points = self.spatial_coords.shape[0]
        indices = np.random.choice(total_points, size=min(num_points, total_points), replace=False)
        return self.spatial_coords[indices], indices
    
    def sample_points_from_mask(self, mask, num_points, fg_ratio=0.5):
        """Sample points with foreground/background balance as per paper's strategy."""
        mask_flat = mask.flatten()
        fg_indices = np.where(mask_flat > 0)[0]
        bg_indices = np.where(mask_flat == 0)[0]
        
        if len(fg_indices) == 0:
            # No foreground, sample uniformly
            return self.sample_points_uniformly(num_points)
        
        num_fg = int(num_points * fg_ratio)
        num_bg = num_points - num_fg
        
        # Sample foreground points
        fg_sample_indices = np.random.choice(fg_indices, size=min(num_fg, len(fg_indices)), replace=True)
        
        # Sample background points
        bg_sample_indices = np.random.choice(bg_indices, size=min(num_bg, len(bg_indices)), replace=True)
        
        # Combine indices
        all_indices = np.concatenate([fg_sample_indices, bg_sample_indices])
        np.random.shuffle(all_indices)
        
        return self.spatial_coords[all_indices], all_indices

    def get_voxel_spacing(self):
        """Return voxel spacing (mm)."""
        return self.voxel_spacing