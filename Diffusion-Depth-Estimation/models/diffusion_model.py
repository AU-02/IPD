import torch
import torch.nn as nn
import numpy as np
import cv2
from models.unet import UNet
from skimage.color import rgb2gray
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

class UnifiedDiffusion(nn.Module):
    def __init__(self, noise_steps=1000):
        super(UnifiedDiffusion, self).__init__()
        self.noise_steps = noise_steps
        self.unet = UNet(in_channels=3, out_channels=1)  # U-Net for depth generation

    def forward_diffusion(self, x, noise_level):
        """Applies forward diffusion (adds noise)."""
        noise = torch.randn_like(x) * noise_level
        return x + noise

    def fill_depth_colorization(self, imgRgb, imgDepth, alpha=1):
        """KITTI Depth Completion Method with resizing fix."""
        
        # Fix: Ensure imgDepth is not empty
        if imgDepth is None or imgDepth.size == 0:
            raise ValueError("imgDepth is empty or None before resizing!")

        # Fix: Ensure imgDepth is valid before resizing
        if len(imgDepth.shape) == 3:
            print(f"Fixing imgDepth shape: {imgDepth.shape}")
            imgDepth = imgDepth[0]  # Extract first channel if batch size is included
        
        if imgDepth.shape[0] == 1:  # Handle single-channel depth
            imgDepth = imgDepth.squeeze(0)

        # Fix: Ensure imgDepth matches imgRgb shape
        if imgDepth.shape != imgRgb.shape[:2]:  
            print(f"Resizing imgDepth from {imgDepth.shape} to {imgRgb.shape[:2]}")
            
            #  Handle empty depth maps
            if imgDepth.size == 0:
                raise ValueError("imgDepth is empty before resizing. Check depth output from U-Net.")

            imgDepth = cv2.resize(imgDepth, (imgRgb.shape[1], imgRgb.shape[0]), interpolation=cv2.INTER_LINEAR)

        # Proceed with depth completion after ensuring correct shape
        imgIsNoise = (imgDepth == 0) | (imgDepth == 10)
        maxImgAbsDepth = np.max(imgDepth[~imgIsNoise])
        imgDepth = imgDepth / 10000.0
        imgDepth[imgDepth > 1] = 1
        H, W = imgDepth.shape
        numPix = H * W
        indsM = np.arange(numPix).reshape(H, W)
        knownValMask = ~imgIsNoise
        grayImg = rgb2gray(imgRgb)
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


    def forward(self, x, imgRgb):
        """Runs the diffusion model and applies depth completion."""
        noisy_x = self.forward_diffusion(x, noise_level=0.1)
        raw_depth = self.unet(noisy_x)

        # Convert PyTorch tensors to NumPy
        imgRgb_np = imgRgb.detach().cpu().numpy()
        raw_depth_np = raw_depth.detach().cpu().numpy()

        print(f"Before Fix - imgRgb_np shape: {imgRgb_np.shape}")  # Debugging

        # Fix: Extract single image from batch and ensure correct shape
        imgRgb_np = imgRgb_np[0].transpose(1, 2, 0)  # Convert (3, H, W) → (H, W, 3)
        raw_depth_np = raw_depth_np[0]  # Extract first depth map

        print(f"After Fix - imgRgb_np shape: {imgRgb_np.shape}")  # Debugging

        # Fix: Ensure imgDepth matches imgRgb before processing
        dense_depth = self.fill_depth_colorization(imgRgb_np, raw_depth_np, alpha=1)

        # Convert dense_depth back to tensor with requires_grad=True
        dense_depth_tensor = torch.tensor(dense_depth, dtype=torch.float32, device=x.device, requires_grad=True)

        return dense_depth_tensor.unsqueeze(0)  # Ensure shape is correct for loss function
