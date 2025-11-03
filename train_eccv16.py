import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os, glob, numpy as np
from colorizers.custom_colorizer import eccv16_custom

# ---------------- Dataset ----------------
class PairedBWColorDataset(Dataset):
    def __init__(self, bw_folder, color_folder, max_samples=None):
        self.bw_files = sorted(glob.glob(os.path.join(bw_folder, "*.jpg")))
        self.color_files = sorted(glob.glob(os.path.join(color_folder, "*.jpg")))
        assert len(self.bw_files) == len(self.color_files), "Mismatch in paired dataset"

        if max_samples is not None:
            self.bw_files = self.bw_files[:max_samples]
            self.color_files = self.color_files[:max_samples]

        self.resize = transforms.Resize((256,256))

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

# ---------------- Setup ----------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model = eccv16_custom(pretrained=False).to(device)

# ---------------- Dataset & Loader ----------------
# Set max_samples to lower number for quick test, None for full dataset
dataset = PairedBWColorDataset("imgs/train_black", "imgs/train_color", max_samples=None)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

# ---------------- Loss & Optimizer ----------------
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# ---------------- Training Loop ----------------
num_epochs = 50  # smaller for quick test
for epoch in range(num_epochs):
    for i, (L, ab) in enumerate(loader):
        L, ab = L.to(device), ab.to(device)
        pred_ab = model(L)

        loss = criterion(pred_ab, ab)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}, Batch {i+1}/{len(loader)}, Loss: {loss.item():.4f}")

# ---------------- Save Weights ----------------
torch.save(model.state_dict(), "eccv16_myweights.pth")
print("Training complete. Weights saved as eccv16_myweights.pth")