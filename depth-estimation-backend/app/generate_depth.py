import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from models.diffusion_model import UnifiedDiffusion
from skimage.color import rgb2gray
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get the absolute path to the backend root directory
BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # Moves one level up from 'app'
MODEL_PATH = os.path.join(BACKEND_DIR, "checkpoints", "unified_model_final.pth")  # Construct full path

# Ensure the model file exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"🔥 Model file not found at {MODEL_PATH}. Make sure the path is correct.")


# Load trained model
model = UnifiedDiffusion().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

def fill_depth_colorization(imgRgb, imgDepth, alpha=1):
    """Densify sparse depth map using the KITTI method."""
    if imgDepth.shape != imgRgb.shape[:2]:  
        print(f"Resizing imgDepth from {imgDepth.shape} to {imgRgb.shape[:2]}")
        imgDepth = cv2.resize(imgDepth, (imgRgb.shape[1], imgRgb.shape[0]), interpolation=cv2.INTER_LINEAR)

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

def generate_depth(image_path):
    # Load input image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Error: Image {image_path} not found. Check the file path.")
    
    # Convert BGR to RGB and normalize
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # Ensure the image matches the model's expected input size
    expected_height, expected_width = 256, 640  # Adjust according to your model training size
    image_resized = cv2.resize(image, (expected_width, expected_height), interpolation=cv2.INTER_LINEAR)

    # Convert to PyTorch tensor
    image_tensor = torch.tensor(image_resized, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
    
    # Run depth prediction
    with torch.no_grad():
        pred_depth = model(image_tensor, image_tensor).squeeze().cpu().numpy()
        
    # Scale depth values to check if they are too small
    pred_depth = pred_depth * 1000  # Scale depth values

    # Debugging: Check raw depth values before densification
    print(f"🔎 Raw Depth Min: {pred_depth.min()}, Max: {pred_depth.max()}")

    # Apply KITTI depth completion
    densified_depth = fill_depth_colorization(image_resized, pred_depth, alpha=1)

    #  Post-Processing Enhancements
    # 1️ aussian Blur (Smooths noise)
    densified_depth = cv2.GaussianBlur(densified_depth, (5, 5), 0)

    # 2 Bilateral Filter (Preserves Edges)
    densified_depth = cv2.bilateralFilter(densified_depth.astype(np.float32), 9, 75, 75)

    # 3️ ormalize for Visualization
    densified_depth = (densified_depth - densified_depth.min()) / (densified_depth.max() - densified_depth.min())

    # Save depth map using Matplotlib (Viridis colormap for better visualization)
    output_path = "output_depth_densified.png"
    plt.figure(figsize=(10, 5))
    plt.imshow(densified_depth, cmap="viridis", aspect='auto')
    plt.colorbar(label='Depth Value')
    plt.title("Densified Depth Map (KITTI + Post-Processing)")
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Depth map saved at: {output_path}")
    
    # Show the depth map
    plt.figure(figsize=(10, 5))
    plt.imshow(densified_depth, cmap="viridis")
    plt.colorbar()
    plt.title("Densified Depth Map (KITTI + Post-Processing)")
    plt.show()

# Run depth generation
if __name__ == "__main__":
    generate_depth("test_image1.png")  # Ensure test_image.png exists
