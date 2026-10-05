# Phase-contrast to projected-thickness translation; MCNN adaptation of Wang et al. (2020).

import os
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
import re

# Dataset class with filename-matching
class Img2ImgDataset(Dataset):
    def __init__(self, input_path, target_path):
        input_files = sorted(glob(os.path.join(input_path, '*.tif')))
        target_files = sorted(glob(os.path.join(target_path, '*.tif')))

        def index_files(files):
            result = {}
            for file in files:
                ids = re.findall(r"(\d+)", os.path.basename(file))
                if not ids:
                    raise ValueError(f"No numeric ID in filename: {file}")
                key = int(ids[0])
                if key in result:
                    raise ValueError(f"Duplicate image ID {key}: {result[key]} and {file}")
                result[key] = file
            return result

        input_dict = index_files(input_files)
        target_dict = index_files(target_files)
        if not input_dict or input_dict.keys() != target_dict.keys():
            raise ValueError("Input and target folders must contain the same nonempty set of numeric IDs.")
        common_keys = sorted(input_dict)

        self.input_files = [input_dict[k] for k in common_keys]
        self.target_files = [target_dict[k] for k in common_keys]

        self.transform = transforms.Compose([
            transforms.CenterCrop(256),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        input_img = Image.open(self.input_files[idx])
        target_img = Image.open(self.target_files[idx])
        return self.transform(input_img), self.transform(target_img)

# U-Net model
class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=(32, 64, 128, 256)):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.upconvs = nn.ModuleList()
        self.pool = nn.MaxPool2d(2)

        for f in features:
            self.downs.append(self._block(in_channels, f))
            in_channels = f

        self.bottleneck = self._block(features[-1], features[-1] * 2)

        for f in reversed(features):
            self.upconvs.append(nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2))
            self.ups.append(self._block(f * 2, f))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def _block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, return_multiscale=False):
        if min(x.shape[-2:]) < 16:
            raise ValueError("Images must be at least 16 pixels on each axis.")
        outputs = []
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        skips = skips[::-1]
        for i in range(len(self.ups)):
            x = self.upconvs[i](x)
            if x.shape != skips[i].shape:
                x = F.interpolate(x, size=skips[i].shape[2:])
            x = torch.cat((skips[i], x), dim=1)
            x = self.ups[i](x)
            if hasattr(self, "coarse_heads") and i < len(self.coarse_heads):
                outputs.append(self.coarse_heads[i](x))
        full_resolution = self.final_conv(x)
        outputs.append(full_resolution)
        return outputs if return_multiscale else full_resolution

class MCNN(UNet):
    """Compact U-Net with decoder supervision inspired by Wang et al., 2020.

    Default heads produce 32, 64, 128 and 256 pixel outputs for a 256 pixel input.
    Normalization layers are omitted. This is not the authors' exact architecture.
    """
    def __init__(self, in_channels=1, out_channels=1, features=(32, 64, 128, 256)):
        super().__init__(in_channels, out_channels, features)
        self.coarse_heads = nn.ModuleList([
            nn.Conv2d(channels, out_channels, kernel_size=1)
            for channels in tuple(reversed(features))[:-1]
        ])


def multiscale_loss(outputs, target):
    """Equal-weight mean of per-scale MAE; area averaging makes coarse targets."""
    losses = [F.l1_loss(output, F.interpolate(target, size=output.shape[-2:], mode="area"))
              for output in outputs]
    return torch.stack(losses).mean()


# Training
def train_model(model, train_loader, val_loader, device, epochs=10, lr=1e-4):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1} Training", leave=False):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x, return_multiscale=True)
            loss = multiscale_loss(outputs, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)

        model.eval()
        val_loss = 0
        val_mae = 0
        val_mse = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x, return_multiscale=True)
                val_loss += multiscale_loss(outputs, y).item() * x.size(0)
                val_mae += F.l1_loss(outputs[-1], y).item() * x.size(0)
                val_mse += F.mse_loss(outputs[-1], y).item() * x.size(0)

        metrics = {"epoch": epoch + 1,
                   "train_loss": train_loss / len(train_loader.dataset),
                   "val_loss": val_loss / len(val_loader.dataset),
                   "val_mae": val_mae / len(val_loader.dataset),
                   "val_rmse": (val_mse / len(val_loader.dataset)) ** 0.5}
        history.append(metrics)
        print(metrics, flush=True)
    return history

# Visualization
def visualize_predictions(model, dataset, device, n=3):
    model.eval()
    model.to(device)
    with torch.no_grad():
        for i in range(min(n, len(dataset))):
            x, y = dataset[i]
            x = x.unsqueeze(0).to(device)
            pred = model(x).squeeze().cpu().numpy()

            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            axs[0].imshow(x.squeeze().cpu().numpy(), cmap='gray')
            axs[0].set_title("Input")
            axs[1].imshow(y.squeeze().numpy(), cmap='gray')
            axs[1].set_title("Ground Truth")
            axs[2].imshow(pred, cmap='gray')
            axs[2].set_title("Prediction")
            for ax in axs: ax.axis("off")
            plt.show()

# Main (no download step)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train phase contrast to projected thickness translation")
    parser.add_argument("--data-dir", required=True, help="Directory containing phase_contrast and projected_thickness")
    parser.add_argument("--model", choices=("mcnn", "unet"), default="mcnn")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        parser.error("epochs, batch size, and learning rate must be positive")
    torch.manual_seed(42)
    base_path = args.data_dir
    input_path = os.path.join(base_path, "phase_contrast")
    target_path = os.path.join(base_path, "projected_thickness")

    print("Input folder exists:", os.path.exists(input_path))
    print("Target folder exists:", os.path.exists(target_path))
    print("Sample input files:", glob(os.path.join(input_path, '*.tif'))[:3])
    print("Sample target files:", glob(os.path.join(target_path, '*.tif'))[:3])

    dataset = Img2ImgDataset(input_path, target_path)

    if len(dataset) == 0:
        raise ValueError("Dataset appears to be empty. Please check the image folder paths.")

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    if train_size == 0 or val_size == 0:
        raise ValueError("Train or validation split resulted in zero samples. Add more images.")

    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MCNN() if args.model == "mcnn" else UNet()

    print(f"Using device: {device} | Dataset size: {len(dataset)}")
    history = train_model(model, train_loader, val_loader, device, epochs=args.epochs, lr=args.lr)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / f"{args.model}_trained_weights.pth")
    (output_dir / "history.json").write_text(__import__("json").dumps(history, indent=2))
    (output_dir / "config.json").write_text(__import__("json").dumps(vars(args), indent=2))
    (output_dir / "split.json").write_text(__import__("json").dumps({
        "train": [dataset.input_files[i] for i in train_ds.indices],
        "validation": [dataset.input_files[i] for i in val_ds.indices]
    }, indent=2))
    if args.plot:
        visualize_predictions(model, val_ds, device)