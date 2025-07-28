import numpy as np
import os
import nibabel as nib
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

class CTSequenceDataset:
    """
    Loads a temporal sequence of 3D CT frames (segmentation or intensity).
    Matches paper's conceptual flow: treat each frame independently.
    Uses sklearn for all data preprocessing and normalization.
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
        
        # sklearn scalers for consistent preprocessing
        self.intensity_scaler = MinMaxScaler(feature_range=(0, 1))
        self.spatial_scaler = MinMaxScaler(feature_range=(-1, 1))

        self._load_frames()
        self._compute_coords()

    def _load_frames(self):
        """Load each 3D frame independently without stacking, using sklearn for normalization."""
        for i in range(self.num_frames):
            filename = f"{i*5}pct.nii.gz"
            file_path = os.path.join(self.dataset_path, filename)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Missing file: {file_path}")

            nii = nib.load(file_path)
            data = nii.get_fdata().astype(np.float32)

            # Use sklearn MinMaxScaler for intensity normalization (0-1)
            original_shape = data.shape
            data_flat = data.reshape(-1, 1)
            data_normalized = self.intensity_scaler.fit_transform(data_flat)
            data = data_normalized.reshape(original_shape)

            # Save volume
            self.frames.append(data)

            # Get voxel spacing (assume all frames same)
            if self.voxel_spacing is None:
                self.voxel_spacing = nii.header.get_zooms()

    def _compute_coords(self):
        """Compute spatial and temporal normalized coordinates using sklearn scalers."""
        # Paper's spatial normalization: coordinates normalized to [-1,1]^3
        # Using consistent grid generation as in paper's spatial domain Ω⊂R^3
        H, W, D = self.frames[0].shape
        
        # Generate raw spatial coordinates
        z = np.arange(H)  # z-axis (depth)
        y = np.arange(W)  # y-axis (height) 
        x = np.arange(D)  # x-axis (width)
        
        # Create meshgrid with 'ij' indexing for consistency with paper
        zz, yy, xx = np.meshgrid(z, y, x, indexing='ij')
        
        # Stack coordinates as (x, y, z) following paper's coordinate convention
        raw_coords = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
        
        # Use sklearn MinMaxScaler to normalize to [-1,1]^3
        self.spatial_coords = self.spatial_scaler.fit_transform(raw_coords)

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
    
    def sample_points_uniformly(self, num_points, random_state=42):
        """Sample points uniformly from spatial domain using sklearn utilities."""
        total_points = self.spatial_coords.shape[0]
        if num_points >= total_points:
            return self.spatial_coords, np.arange(total_points)
        
        # Use sklearn's shuffle for consistent and reproducible sampling
        coords_shuffled, indices_shuffled = shuffle(
            self.spatial_coords, 
            np.arange(total_points), 
            n_samples=num_points,
            random_state=random_state
        )
        return coords_shuffled, indices_shuffled
    
    def sample_points_from_mask(self, mask, num_points, fg_ratio=0.5, random_state=42):
        """Sample points with foreground/background balance using sklearn sampling."""
        mask_flat = mask.flatten()
        fg_indices = np.where(mask_flat > 0)[0]
        bg_indices = np.where(mask_flat == 0)[0]
        
        if len(fg_indices) == 0:
            # No foreground, sample uniformly
            return self.sample_points_uniformly(num_points, random_state)
        
        num_fg = int(num_points * fg_ratio)
        num_bg = num_points - num_fg
        
        # Use sklearn's shuffle for consistent sampling
        if len(fg_indices) >= num_fg:
            fg_coords_sampled, fg_indices_sampled = shuffle(
                self.spatial_coords[fg_indices], 
                fg_indices, 
                n_samples=num_fg,
                random_state=random_state
            )
        else:
            # If not enough foreground points, use all with replacement
            fg_coords_sampled = self.spatial_coords[fg_indices]
            fg_indices_sampled = fg_indices
            # Repeat to reach desired count
            n_repeats = (num_fg // len(fg_indices)) + 1
            fg_coords_sampled = np.tile(fg_coords_sampled, (n_repeats, 1))[:num_fg]
            fg_indices_sampled = np.tile(fg_indices_sampled, n_repeats)[:num_fg]
        
        # Sample background points
        if len(bg_indices) >= num_bg:
            bg_coords_sampled, bg_indices_sampled = shuffle(
                self.spatial_coords[bg_indices], 
                bg_indices, 
                n_samples=num_bg,
                random_state=random_state + 1  # Different seed for background
            )
        else:
            # If not enough background points, use all with replacement
            bg_coords_sampled = self.spatial_coords[bg_indices]
            bg_indices_sampled = bg_indices
            # Repeat to reach desired count
            n_repeats = (num_bg // len(bg_indices)) + 1
            bg_coords_sampled = np.tile(bg_coords_sampled, (n_repeats, 1))[:num_bg]
            bg_indices_sampled = np.tile(bg_indices_sampled, n_repeats)[:num_bg]
        
        # Combine and shuffle the final result
        all_coords = np.vstack([fg_coords_sampled, bg_coords_sampled])
        all_indices = np.concatenate([fg_indices_sampled, bg_indices_sampled])
        
        # Final shuffle of combined results
        final_coords, final_indices = shuffle(all_coords, all_indices, random_state=random_state + 2)
        
        return final_coords, final_indices

    def get_voxel_spacing(self):
        """Return voxel spacing (mm)."""
        return self.voxel_spacing
    
    def get_intensity_scaler(self):
        """Return the fitted intensity scaler for consistent preprocessing."""
        return self.intensity_scaler
    
    def get_spatial_scaler(self):
        """Return the fitted spatial coordinate scaler."""
        return self.spatial_scaler