import os
import nibabel as nib
import numpy as np
from skimage import measure
import trimesh

def extract_mesh_from_nifti(nifti_path, output_path, level=0.5):
    """
    Extract triangular mesh from a binary segmentation NIfTI file and save as .ply.
    """
    # Load segmentation
    nii = nib.load(nifti_path)
    data = nii.get_fdata().astype(np.float32)

    # Get voxel spacing from NIfTI header (for physical scaling)
    spacing = nii.header.get_zooms()  # (z, y, x) or similar

    # Marching cubes to get vertices and faces
    verts, faces, normals, values = measure.marching_cubes(data, level=level, spacing=spacing)

    # Create a mesh object
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

    # Save as PLY
    mesh.export(output_path)
    print(f"Mesh saved: {output_path}")


def extract_all_meshes(input_dir, output_dir):
    """
    Extract meshes for all 20 frames (0pct to 95pct) in the input directory.
    """
    os.makedirs(output_dir, exist_ok=True)

    for i in range(20):
        filename = f"{i*5}pct.nii.gz"
        nifti_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, f"{i*5}pct.ply")

        if not os.path.exists(nifti_path):
            raise FileNotFoundError(f"Segmentation file not found: {nifti_path}")

        extract_mesh_from_nifti(nifti_path, output_path)


if __name__ == "__main__":
    input_dir = "data/4007775_aneurysm/nnunet_outputs_pp"   # segmentation frames
    output_dir = "data/4007775_aneurysm/meshes"            # where to save meshes

    extract_all_meshes(input_dir, output_dir)