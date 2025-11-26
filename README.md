# 4D CTA Neural Field

Implementation of **"Neural Fields for Continuous Periodic Motion Estimation in 4D Cardiovascular Imaging"** by Garzia et al. ([arXiv:2407.20728](https://arxiv.org/abs/2407.20728))

## Overview

This repository implements a neural fields-based approach to estimate continuous periodic wall deformations in 4D cardiovascular imaging. The method uses implicit neural representations (SIREN networks) to model time-dependent velocity fields, which are then integrated via ODEs to obtain deformation vector fields (DVFs).

### Key Features

- **Continuous Motion Representation**: Models arterial wall motion as a continuous velocity field rather than discrete displacements
- **Periodic Consistency**: Enforces periodicity through time encoding and DVF regularization for cardiac cycle modeling
- **4D Flow MRI Compatibility**: Designed for time-resolved 3D cardiovascular imaging data
- **Quantitative Evaluation**: Includes volume tracking and Hausdorff distance metrics

## Architecture

The implementation consists of three main components:

1. **SIREN Velocity Field**: Implicit neural representation of time-dependent velocity vectors
   - Sinusoidal activation functions (SIREN architecture)
   - Periodic time encoding: `[cos(2πt), sin(2πt)]`
   - Input: 3D spatial coordinates + time → Output: 3D velocity vector

2. **ODE Integration**: Converts velocity field to deformation field
   - Uses `torchdiffeq` for numerical integration
   - Generates continuous trajectories from t₀ to tₙ

3. **Loss Functions**:
   - **Image Reconstruction Loss**: `||I_ti ∘ φ_ti→T - I_T||²`
   - **Cycle Consistency Loss**: `||P₀,i - φ_T(P₀,i)||²`
   - Combined: `L = L_recon + λ_cycle × L_cycle`

## Installation

### Requirements

- Python 3.8+
### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/4dCTA-NeuralField.git
cd 4dCTA-NeuralField

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Data Format

The dataset should contain temporal sequences of 3D CT/MRI volumes in NIfTI format (`.nii.gz`):

```
data/
└── {patient_id}_aneurysm/
    └── nnunet_outputs_pp/
        ├── 0pct.nii.gz
        ├── 5pct.nii.gz
        ├── 10pct.nii.gz
        ...
        └── 95pct.nii.gz
```

- Files represent cardiac cycle phases (0% = end-diastole, typically)
- Default: 20 frames covering one complete cycle (0-95% in 5% increments)
- Each file contains a 3D binary segmentation or intensity volume

## Usage

### Basic Training

```python
from motion_estimation import SIRENVelocityField, train_inr_model
from dataloader import CTSequenceDataset

# Load data
dataset = CTSequenceDataset("data/patient_id/nnunet_outputs_pp", num_frames=20)
frames, temporal_coords = dataset.get_all_frames()
spatial_coords = dataset.get_spatial_coords()

# Initialize model
siren_model = SIRENVelocityField(hidden_dim=256, num_layers=3, w0=30.0)

# Train
trained_model, gt_volumes = train_inr_model(
    siren_model, frames, spatial_coords, temporal_coords,
    mesh_vertices, mesh_faces, voxel_spacing, mesh_scaler,
    num_epochs=500,
    sample_points=5300,
    lambda_cycle=0.01,
    device='cuda'
)
```

### Running the Full Pipeline

```bash
python motion_estimation.py
```

This will:
1. Load the CT sequence from the data directory
2. Extract the initial mesh using Marching Cubes
3. Train the SIREN velocity field model
4. Generate deformation trajectories
5. Compute volume predictions and Hausdorff distances
6. Save results to the visualizations directory

### Hyperparameter Sweep

To find the optimal `λ_cycle` value:

```python
lambda_values = [0.01, 0.05, 0.1, 0.5, 1.0]
sweep_results = sweep_lambda_cycle(
    lambda_values,
    frames, spatial_coords, temporal_coords,
    mesh_vertices, mesh_faces, voxel_spacing, mesh_scaler,
    num_epochs=500,
    device='cuda',
    save_dir="sweep_results"
)
```

## Project Structure

```
4dCTA-NeuralField/
├── motion_estimation.py    # Main training and inference pipeline
├── dataloader.py            # CT sequence data loading utilities
├── requirements.txt         # Python dependencies
├── .gitignore              # Git ignore rules
├── README.md               # This file
└── data/                   # Data directory (not tracked)
    └── {patient_id}_aneurysm/
        ├── nnunet_outputs_pp/     # Input NIfTI volumes
        └── visualizations/        # Output plots and metrics
```

## Output

The pipeline generates several evaluation metrics and visualizations:

### Volume Comparison
- `volume_comparison_*.png`: Multi-panel plots showing:
  - Ground truth vs predicted volumes
  - Relative error over time
  - Scatter plot (prediction vs GT)
  - Error distribution histogram
- `volume_comparison_*.csv`: Frame-by-frame volume data
- `volume_comparison_*_stats.csv`: Summary statistics (MSE, MAE, R²)

### Hausdorff Distance
- `*_hsd.csv`: Per-frame Hausdorff distance measurements
- Reports mean and max HSD in millimeters

## Evaluation Metrics

- **Volume Accuracy**:
  - Mean Squared Error (MSE)
  - Mean Absolute Error (MAE)
  - R² score

- **Surface Accuracy**:
  - Symmetric Hausdorff Distance (HSD)
  - Measured between predicted and ground truth meshes

- **Cycle Consistency**:
  - Displacement between initial and final positions
  - Enforced through `λ_cycle` regularization

## Citation

```bibtex
@article{garzia2024neural,
  title={Neural Fields for Continuous Periodic Motion Estimation in 4D Cardiovascular Imaging},
  author={Garzia, Simone and others},
  journal={arXiv preprint arXiv:2407.20728},
  year={2024}
}
```

## Acknowledgments

This implementation is based on the methodology described in the paper by Garzia et al. Key components include:

- SIREN architecture from [Implicit Neural Representations with Periodic Activation Functions](https://arxiv.org/abs/2006.09661)
- ODE integration via [torchdiffeq](https://github.com/rtqichen/torchdiffeq)