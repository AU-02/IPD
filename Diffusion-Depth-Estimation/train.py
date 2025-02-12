import sys
import os
import torch
import time
import numpy as np
from torch.utils.data import DataLoader
sys.path.append("D:/FYP-001")
from dataloader.MS2_dataset import DataLoader_MS2
from models.diffusion_model import UnifiedDiffusion
from tqdm import tqdm  # Progress bar

# Force CUDA for computations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = torch.Generator(device="cpu").manual_seed(42)

# Debugging CUDA Usage
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"CUDA Device Name: {torch.cuda.get_device_name(torch.cuda.current_device())}")

def compute_metrics(gt, pred):
    """Computes MAE, MSE, RMSE, Abs Rel, log RMSE, and Accuracy."""
    
    if len(gt.shape) == 3:
        print(f"Multiple batch samples found: {gt.shape}")
        gt = gt[0]  # Take the first sample from batch

    if len(pred.shape) == 3:
        pred = pred[0]  # Ensure it's (H, W)

    if gt.shape != pred.shape:
        raise ValueError(f"Shape mismatch: gt={gt.shape}, pred={pred.shape}")

    mask = gt > 0  # Ignore zero-depth values
    gt = gt[mask]
    pred = pred[mask]

    mae = np.mean(np.abs(gt - pred))
    mse = np.mean((gt - pred) ** 2)
    rmse = np.sqrt(mse)
    abs_rel = np.mean(np.abs(gt - pred) / (gt + 1e-6))
    log_rmse = np.sqrt(np.mean((np.log(gt + 1e-6) - np.log(pred + 1e-6)) ** 2))
    accuracy = np.mean((gt / pred < 1.25) | (pred / gt < 1.25))

    return mae, mse, rmse, abs_rel, log_rmse, accuracy

if __name__ == '__main__':  # Fix multiprocessing issue on Windows
    # Adjust num_workers based on OS (Windows needs num_workers=0)
    num_workers = 4 if os.name != 'nt' else 0  

    # Load dataset with optimized settings
    dataset = DataLoader_MS2(root="D:/FYP-001/MS2dataset")
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, pin_memory=True, num_workers=num_workers, generator=generator)

    # Initialize model on GPU
    model = UnifiedDiffusion().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = torch.nn.MSELoss()

    num_epochs = 20
    total_batches = len(dataloader)
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        total_loss = 0
        total_rmse, total_accuracy, count = 0, 0, 0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        
        for batch_idx, batch in enumerate(progress_bar):
            print(f"Batch keys: {batch.keys()}") 

            # Move data to GPU with async transfer
            image = batch["tgt_image"].to(device, non_blocking=True)
            depth_gt = batch["tgt_depth_gt"].to(device, non_blocking=True)

            print(f"Before Fix - Image shape: {image.shape}")  

            # Ensure correct shape (batch_size, 3, height, width)
            if image.shape[1] not in [1, 3]:  
                print(f"Fixing image shape: {image.shape}")  
                image = image.permute(0, 3, 1, 2)  # Move channels to first dimension

            if image.shape[1] == 1:  
                print(f"Expanding grayscale to RGB: {image.shape}")  
                image = image.repeat(1, 3, 1, 1)  

            print(f"After Fix - Image shape: {image.shape}")  

            optimizer.zero_grad()
            pred_depth = model(image, image)
            
            # Ensure pred_depth is on GPU before loss computation
            pred_depth = pred_depth.to(device)

            # Fix: Ensure loss requires gradients
            loss = criterion(pred_depth, depth_gt)
            loss.requires_grad_(True)  

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Compute evaluation metrics
            pred_depth_np = pred_depth.squeeze().detach().cpu().numpy()
            depth_gt_np = depth_gt.squeeze().detach().cpu().numpy()
            
            mae, mse, rmse, abs_rel, log_rmse, accuracy = compute_metrics(depth_gt_np, pred_depth_np)
            total_rmse += rmse
            total_accuracy += accuracy
            count += 1

            #  Estimate Time Remaining
            elapsed_time = time.time() - epoch_start_time
            avg_batch_time = elapsed_time / (batch_idx + 1)
            remaining_batches = total_batches - (batch_idx + 1)
            remaining_time = avg_batch_time * remaining_batches  

            progress_bar.set_postfix(loss=f"{loss.item():.4f}", rmse=f"{rmse:.4f}", accuracy=f"{accuracy:.4f}", remaining=f"{remaining_time/60:.2f} min")

        # End of epoch logging
        epoch_time = time.time() - epoch_start_time
        avg_rmse = total_rmse / count
        avg_accuracy = total_accuracy / count
        
        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {total_loss/total_batches:.6f} - RMSE: {avg_rmse:.4f} - Accuracy: {avg_accuracy:.4f} - Time: {epoch_time/60:.2f} min")

        # Save checkpoint
        torch.save(model.state_dict(), f"checkpoints/unified_model_epoch{epoch+1}.pth")

    # Save Final Trained Model
    torch.save(model.state_dict(), "checkpoints/unified_model_final.pth")
    print("Final trained model saved at checkpoints/unified_model_final.pth")
