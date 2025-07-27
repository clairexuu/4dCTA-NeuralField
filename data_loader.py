import os
import nibabel as nib
import numpy as np
from tqdm import tqdm

def load_4d_cta(dataset_path):
    """
    Load 20 frames (0pct to 95pct) of 4D CTA data into a 4D numpy array: (T, H, W, D).
    Normalize intensities to [0,1].
    """
    # Expected filenames: 0pct.nii.gz, 5pct.nii.gz, ... 95pct.nii.gz
    frames = []
    for i in range(20):
        filename = f"{i*5}pct.nii.gz"
        file_path = os.path.join(dataset_path, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Frame not found: {file_path}")

        img = nib.load(file_path)
        data = img.get_fdata().astype(np.float32)

        frames.append(data)

    # Stack frames into 4D array: (T, H, W, D)
    volume_4d = np.stack(frames, axis=0)  # Shape: (20, H, W, D)

    # Normalize intensities to [0,1]
    min_val = np.min(volume_4d)
    max_val = np.max(volume_4d)
    volume_4d = (volume_4d - min_val) / (max_val - min_val + 1e-8)

    return volume_4d


def normalize_spatial_coordinates(volume_shape):
    """
    Generate normalized spatial coordinates in [-1,1]^3 for the given volume shape (H,W,D).
    Returns an array of shape (H*W*D, 3).
    """
    H, W, D = volume_shape
    # Create coordinate grids
    z = np.linspace(-1, 1, H)
    y = np.linspace(-1, 1, W)
    x = np.linspace(-1, 1, D)

    zz, yy, xx = np.meshgrid(z, y, x, indexing='ij')  # Shape: (H, W, D)
    coords = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)  # Flatten to (N, 3)
    return coords


def normalize_temporal_coordinates(num_frames=20):
    """
    Generate normalized temporal coordinates in [0,1] for given number of frames.
    Returns array of shape (num_frames,).
    """
    return np.linspace(0, 1, num_frames, dtype=np.float32)


if __name__ == "__main__":
    # Example usage
    dataset_path = "data/4007775_aneurysm/nnunet_outputs_pp"

    print("Loading 4D CTA data...")
    volume_4d = load_4d_cta(dataset_path)
    print(f"Loaded volume shape: {volume_4d.shape}")  # (20, H, W, D)

    # Get spatial coords
    spatial_coords = normalize_spatial_coordinates(volume_4d.shape[1:])
    print(f"Spatial coordinates shape: {spatial_coords.shape}")  # (H*W*D, 3)

    # Get temporal coords
    temporal_coords = normalize_temporal_coordinates(volume_4d.shape[0])
    print(f"Temporal coordinates: {temporal_coords}")