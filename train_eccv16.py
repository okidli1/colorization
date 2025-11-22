import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os, glob, numpy as np
from colorizers.custom_colorizer import eccv16_custom
import time

# ---------------- Dataset ----------------
class PairedBWColorDataset(Dataset):
    def __init__(self, bw_folder, color_folder, max_samples=None):
        self.bw_files = sorted(glob.glob(os.path.join(bw_folder, "*.jpg")))
        self.color_files = sorted(glob.glob(os.path.join(color_folder, "*.jpg")))
        assert len(self.bw_files) == len(self.color_files), "Mismatch in paired dataset"

        if max_samples is not None:
            self.bw_files = self.bw_files[:max_samples]
            self.color_files = self.color_files[:max_samples]

        self.resize = transforms.Resize((256, 256))

    def __getitem__(self, idx):
        # Load black & white image
        bw = Image.open(self.bw_files[idx]).convert("L")
        bw = self.resize(bw)
        bw = np.array(bw)[:, :, np.newaxis]

        # Load color image
        color = Image.open(self.color_files[idx]).convert("RGB")
        color = self.resize(color)
        color = np.array(color)

        # Convert color to Lab
        try:
            import cv2
            lab = cv2.cvtColor(color, cv2.COLOR_RGB2LAB).astype(np.float32)
        except:
            # fallback if cv2 not installed
            from skimage import color
            lab = color.rgb2lab(color / 255.0).astype(np.float32)

        # Normalize
        L = bw / 50.0 - 1.0       # [-1,1]
        ab = lab[:,:,1:] / 110.0  # [-1,1]

        # Convert to tensors
        L = torch.from_numpy(L).permute(2,0,1).float()
        ab = torch.from_numpy(ab).permute(2,0,1).float()
        return L, ab

    def __len__(self):
        return len(self.bw_files)

# ---------------- Validation Function ----------------
def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for L, ab in val_loader:
            L, ab = L.to(device), ab.to(device)
            pred_ab = model(L)
            loss = criterion(pred_ab, ab)
            total_loss += loss.item()
    model.train()
    return total_loss / len(val_loader)

def main():
    # ---------------- Setup ----------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = eccv16_custom(pretrained=False).to(device)

    # ---------------- Quick Test Configuration ----------------
    QUICK_TEST = True  # Set to False for full training
    MAX_SAMPLES = 200  # Small dataset for quick testing
    BATCH_SIZE = 8     # Smaller batch size for quick iterations
    NUM_EPOCHS = 10     # Fewer epochs for quick test

    if QUICK_TEST:
        print("🚀 QUICK TEST MODE: Training with small sample size")
        print(f"   Max samples: {MAX_SAMPLES}")
        print(f"   Batch size: {BATCH_SIZE}")
        print(f"   Epochs: {NUM_EPOCHS}")
    else:
        print("🏋️ FULL TRAINING MODE")
        MAX_SAMPLES = None
        BATCH_SIZE = 16
        NUM_EPOCHS = 100

    # ---------------- Dataset & Loader ----------------
    # Create datasets - adjust paths as needed
    train_dataset = PairedBWColorDataset(
        "imgs/train_black", 
        "imgs/train_color", 
        max_samples=MAX_SAMPLES
    )

    # For very small datasets, use all for training
    if MAX_SAMPLES and MAX_SAMPLES <= 50:
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
        print(f"Quick test: Using {len(train_dataset)} samples for both training and validation")
    else:
        # Split into train/val (80/20)
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )
        # Remove num_workers for Windows compatibility
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")

    # ---------------- Loss & Optimizer ----------------
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    # ---------------- Training Loop ----------------
    best_val_loss = float('inf')
    save_path = "eccv16_myweights.pth"

    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()
        train_loss = 0.0
        
        model.train()
        for i, (L, ab) in enumerate(train_loader):
            L, ab = L.to(device), ab.to(device)
            
            optimizer.zero_grad()
            pred_ab = model(L)
            loss = criterion(pred_ab, ab)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Print more frequently for quick tests
            if QUICK_TEST and i % 10 == 0:
                print(f"Epoch {epoch+1}, Batch {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
            elif not QUICK_TEST and i % 50 == 0:
                print(f"Epoch {epoch+1}, Batch {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        # Validation
        val_loss = validate(model, val_loader, criterion, device)
        
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} completed in {epoch_time:.2f}s")
        print(f"Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"✅ Best model saved with val loss: {val_loss:.4f}")
        
        print("-" * 50)

    print("Training complete!")
    print(f"Best weights saved as {save_path}")
    print(f"Final best validation loss: {best_val_loss:.4f}")

if __name__ == '__main__':
    main()