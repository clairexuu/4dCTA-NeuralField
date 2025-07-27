import numpy as np

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
        """Compute spatial and temporal normalized coordinates separately."""
        # Spatial coords based on first frame shape
        H, W, D = self.frames[0].shape
        z = np.linspace(-1, 1, H)
        y = np.linspace(-1, 1, W)
        x = np.linspace(-1, 1, D)
        zz, yy, xx = np.meshgrid(z, y, x, indexing='ij')
        self.spatial_coords = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)

        # Temporal coords normalized to [0,1]
        self.temporal_coords = np.linspace(0, 1, self.num_frames, dtype=np.float32)

    def get_frame(self, idx):
        """Return frame volume (numpy) and its normalized time coordinate."""
        return self.frames[idx], self.temporal_coords[idx]

    def get_all_frames(self):
        """Return list of all frames and time coordinates."""
        return self.frames, self.temporal_coords

    def get_spatial_coords(self):
        """Return normalized spatial coordinates ([-1,1]^3)."""
        return self.spatial_coords

    def get_voxel_spacing(self):
        """Return voxel spacing (mm)."""
        return self.voxel_spacing