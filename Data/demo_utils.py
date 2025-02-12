import torch
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from dataloader.MS2_dataset import DataLoader_MS2
from utils.utils import visualize_disp_as_numpy, visualize_depth_as_numpy, Raw2Celsius
from skimage.color import rgb2gray
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

# Enable Matplotlib Interactive Mode for Faster Rendering
plt.ion()

def fill_depth_colorization(imgRgb, imgDepth, alpha=1):
    """
    Implements the fill_depth_colorization method from KITTI Dense Depth.
    """
    # Ensure imgRgb has the correct shape
    if len(imgRgb.shape) == 4:  
        imgRgb = imgRgb.squeeze(0)  # Remove batch dimension
    if imgRgb.shape[-1] == 1:  
        imgRgb = np.repeat(imgRgb, 3, axis=-1)  # Convert grayscale to 3-channel RGB

    # Convert to grayscale safely
    grayImg = rgb2gray(imgRgb)
    
    # Ensure grayImg is a 2D array
    if len(grayImg.shape) != 2:
        raise ValueError(f"grayImg has an unexpected shape: {grayImg.shape}")

    imgIsNoise = (imgDepth == 0) | (imgDepth == 10)
    maxImgAbsDepth = np.max(imgDepth[~imgIsNoise])
    imgDepth = imgDepth / 10000.0
    imgDepth[imgDepth > 1] = 1
    H, W = imgDepth.shape
    numPix = H * W
    indsM = np.arange(numPix).reshape(H, W)
    knownValMask = ~imgIsNoise
    winRad = 1
    
    rows, cols, vals = [], [], []
    for j in range(W):
        for i in range(H):
            absImgNdx = indsM[i, j]
            gvals = []
            for ii in range(max(0, i - winRad), min(i + winRad + 1, H)):
                for jj in range(max(0, j - winRad), min(j + winRad + 1, W)):
                    if ii == i and jj == j:
                        continue
                    if ii >= grayImg.shape[0] or jj >= grayImg.shape[1]:  # Prevent index error
                        continue
                    rows.append(absImgNdx)
                    cols.append(indsM[ii, jj])
                    gvals.append(grayImg[ii, jj])
            
            curVal = grayImg[i, j]
            gvals.append(curVal)
            c_var = np.mean((np.array(gvals) - np.mean(gvals)) ** 2)
            csig = c_var * 0.6
            mgv = np.min((np.array(gvals[:-1]) - curVal) ** 2)
            if csig < (-mgv / np.log(0.01)):
                csig = -mgv / np.log(0.01)
            if csig < 0.000002:
                csig = 0.000002
            
            gvals[:-1] = np.exp(-(np.array(gvals[:-1]) - curVal) ** 2 / csig)
            if np.sum(gvals[:-1]) != 0:  # Prevent division by zero
                gvals[:-1] /= np.sum(gvals[:-1])
            vals.extend(-np.array(gvals[:-1]))

            rows.append(absImgNdx)
            cols.append(absImgNdx)
            vals.append(1)
    
    A = csr_matrix((vals, (rows, cols)), shape=(numPix, numPix))
    G = csr_matrix((knownValMask.ravel() * alpha, (np.arange(numPix), np.arange(numPix))), shape=(numPix, numPix))
    
    new_vals = spsolve(A + G, (knownValMask.ravel() * alpha * imgDepth.ravel()))
    denoisedDepthImg = new_vals.reshape(H, W) * maxImgAbsDepth
    return denoisedDepthImg


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default='./MS2dataset')
    parser.add_argument('--seq_name', type=str, default='_2021-08-06-10-59-33', help='sequence name')
    return parser.parse_args()

def main():
    args = parse_args()
    dataset_dir = args.dataset_dir
    seq_name = args.seq_name
    modalities = ['rgb', 'nir', 'thr']
    data_formats = ['MonoDepth', 'StereoMatch', 'MultiViewImg','Odometry']
    
    for modality in modalities:
        for data_format in data_formats:
            print(f"Processing: Modality = {modality}, Data Format = {data_format}")
            dataset = DataLoader_MS2(
                dataset_dir,
                data_split=seq_name,
                data_format=data_format,
                modality=modality,
                sampling_step=50,
                set_length=3 if data_format == 'MultiViewImg' else 1,
                set_interval=5 if data_format == 'MultiViewImg' else 1
            )
            demo_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False)
            print(f'{len(demo_loader)} samples found for evaluation.')
            
            for _, batch in enumerate(tqdm(demo_loader)):
                if data_format in ['MonoDepth', 'StereoMatch', 'MultiViewImg']:
                    # **Fixing Input Image Handling**
                    if modality == 'thr':
                        img = Raw2Celsius(batch["tgt_image"])
                    else:
                        img = batch["tgt_image"].type(torch.uint8).squeeze().cpu().numpy()

                    # **Ensure img has correct shape for visualization**
                    if len(img.shape) == 4 and img.shape[0] == 1:  
                        img = img.squeeze(0)  # Remove batch dimension
                    if img.shape[-1] == 1:  
                        img = img.squeeze(-1)  # Remove last channel if it's 1 (grayscale)
                    if len(img.shape) == 2:  
                        img = np.stack([img] * 3, axis=-1)  # Convert grayscale to RGB

                    # **Fixing Sparse Depth Handling**
                    sparse_depth = batch["tgt_depth_gt"].squeeze().cpu().numpy()
                    dense_depth = fill_depth_colorization(img, sparse_depth, alpha=1)

                    # **Matplotlib Optimization**
                    plt.clf()
                    
                    plt.subplot(2, 2, 1)
                    plt.imshow(img)
                    plt.title("Input Image")

                    plt.subplot(2, 2, 2)
                    plt.imshow(visualize_depth_as_numpy(sparse_depth), cmap='viridis')  # Depth map visualization
                    plt.title("Sparse Depth Map")

                    plt.subplot(2, 2, 3)
                    plt.imshow(visualize_depth_as_numpy(dense_depth), cmap='viridis')  # Depth map visualization
                    plt.title("Densified Depth Map")

                    plt.pause(0.1)  # Faster updates


if __name__ == '__main__':
    main()
