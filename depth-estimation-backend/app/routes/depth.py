from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import FileResponse
import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from models.diffusion_model import UnifiedDiffusion
from skimage.color import rgb2gray
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from app.core.security import get_current_user  
from app.generate_depth import generate_depth

router = APIRouter()

# Directory to save uploaded images
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Load Your Trained Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UnifiedDiffusion().to(device)
model.load_state_dict(torch.load("checkpoints/unified_model_final.pth", map_location=device))
model.eval()

def preprocess_image(image_path):
    """Preprocess image for depth estimation model."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Error: Image {image_path} not found.")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    expected_height, expected_width = 256, 640  # Adjust according to model training size
    image_resized = cv2.resize(image, (expected_width, expected_height), interpolation=cv2.INTER_LINEAR)
    image_tensor = torch.tensor(image_resized, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
    
    return image_tensor, image_resized

def fill_depth_colorization(imgRgb, imgDepth, alpha=1):
    """Densify sparse depth map using the KITTI method."""
    if imgDepth.shape != imgRgb.shape[:2]:  
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
    """Run depth estimation model and generate depth map."""
    image_tensor, image_resized = preprocess_image(image_path)

    with torch.no_grad():
        pred_depth = model(image_tensor, image_tensor).squeeze().cpu().numpy()

    pred_depth = pred_depth * 1000
    densified_depth = fill_depth_colorization(image_resized, pred_depth, alpha=1)
    densified_depth = cv2.GaussianBlur(densified_depth, (5, 5), 0)
    densified_depth = cv2.bilateralFilter(densified_depth.astype(np.float32), 9, 75, 75)
    densified_depth = (densified_depth - densified_depth.min()) / (densified_depth.max() - densified_depth.min())

    output_path = image_path.replace(".jpg", "_depth.png")
    plt.imsave(output_path, densified_depth, cmap="viridis")
    
    return output_path

@router.post("/depth-map")
async def generate_depth_map(file: UploadFile = File(...), user: dict = Depends(get_current_user)):
    """Accept an image file, process it using the trained model, and return the depth map."""
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        buffer.write(file.file.read())

    depth_map_path = generate_depth(file_path)

    return {"depth_map_url": f"http://127.0.0.1:8000/uploads/{os.path.basename(depth_map_path)}"}
