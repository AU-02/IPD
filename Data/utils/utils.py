import numpy as np
import torch
from imageio import imread
from scipy.io import loadmat
import matplotlib as mpl
import matplotlib.cm as cm
from pathlib import Path

def load_as_float_img(path):
    path = str(path)  # Ensure compatibility with Pathlib
    img = imread(path).astype(np.float32)
    if len(img.shape) == 2:  # for NIR and thermal images
        img = np.expand_dims(img, axis=2)
    return img

def load_as_float_depth(path):
    path = str(path)  # Convert path to string to check for file extension
    if 'png' in path:
        depth = np.array(imread(path).astype(np.float32))
    elif 'npy' in path:
        depth = np.load(path).astype(np.float32)
    elif 'mat' in path:
        depth = loadmat(path).astype(np.float32)
    else:
        raise ValueError("Unsupported file type for depth data")
    return depth

# Temperature to raw thermal radiation value
# For the parameters (R,B,F,O), we use default values of FLIR A65C camera
def Celsius2Raw(celsius_degree):
    R = 380747
    B = 1428
    F = 1
    O = -88.539
    raw_value = R / (np.exp(B / (celsius_degree + 273.15)) - F) + O
    return raw_value

# Raw thermal radiation value to temperature
def Raw2Celsius(raw):
    R = 380747
    B = 1428
    F = 1
    O = -88.539
    celsius = B / np.log(R / (raw - O) + F) - 273.15
    return celsius

def visualize_disp_as_numpy(disp, cmap='jet'):
    """
    Args:
        disp (HxW): disparity data
        cmap: color map (inferno, plasma, jet, turbo, magma, rainbow)
    Returns:
        vis_data (HxWx3): disparity visualization (RGB)
    """
    disp = disp.cpu().numpy()
    disp = np.nan_to_num(disp)  # change nan to 0

    vmin = np.percentile(disp[disp != 0], 0)
    vmax = np.percentile(disp[disp != 0], 95)

    normalizer = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap=cmap)
    vis_data = (mapper.to_rgba(disp)[:, :, :3] * 255).astype(np.uint8)
    return vis_data

def visualize_depth_as_numpy(depth, cmap='jet', is_sparse=True):
    """
    Args:
        depth (HxW or tensor): Depth data
        cmap: color map (inferno, plasma, jet, turbo, magma, rainbow)
    Returns:
        vis_data (HxWx3): Depth visualization (RGB)
    """
    # Ensure the depth is a NumPy array
    if isinstance(depth, torch.Tensor):
        x = depth.cpu().numpy()
    elif isinstance(depth, np.ndarray):
        x = depth  # Already a NumPy array, no need for conversion
    else:
        raise TypeError(f"Expected input type torch.Tensor or np.ndarray, but got {type(depth)}")

    x = np.nan_to_num(x)  # Change NaN to 0
    inv_depth = 1 / (x + 1e-6)

    if is_sparse:
        vmax = 1 / np.percentile(x[x != 0], 5) if np.any(x != 0) else 1
    else:
        vmax = np.percentile(inv_depth, 95)

    normalizer = mpl.colors.Normalize(vmin=inv_depth.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap=cmap)
    vis_data = (mapper.to_rgba(inv_depth)[:, :, :3] * 255).astype(np.uint8)

    if is_sparse:
        vis_data[inv_depth > vmax] = 0  # Mask invalid pixels

    return vis_data
