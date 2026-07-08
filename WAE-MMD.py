import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchmetrics.image.fid import FrechetInceptionDistance
import matplotlib.pyplot as plt
import numpy as np

torch.set_float32_matmul_precision('high')


# Creation of the datasets
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

batch_size = 128
val_size = 6 * batch_size
train_size = len(train_dataset) - val_size

generator = torch.Generator().manual_seed(42)
train_ds, val_ds = random_split(train_dataset, [train_size, val_size], generator=generator)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)


class WAE(nn.Module):
    def __init__(self, z_dim=128, lam=10.0, lr=1e-4):
        super().__init__()
        self.z_dim = z_dim
        self.lam = lam
        self.lr = lr

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(4*4*256, z_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 256*4*4),
            nn.Unflatten(1, (256, 4, 4)),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def sample(self, n):
        z = torch.randn(n, self.z_dim, device=self.device)
        return self.decoder(z)


# Here is the computation of the MMD (Maximum Mean Discrepancy)
def compute_mmd(z_real, z_fake):
    def rbf_kernel(a, b, sigma):
        diff = a.unsqueeze(1) - b.unsqueeze(0)
        dist = diff.pow(2).sum(-1)
        return torch.exp(-dist / (2 * sigma**2))

    sigmas = [1.0, 2.0, 4.0, 8.0, 16.0]
    mmd = 0.0
    n = z_real.size(0)

    for sigma in sigmas:
        k_rr = rbf_kernel(z_real, z_real, sigma)
        k_ff = rbf_kernel(z_fake, z_fake, sigma)
        k_rf = rbf_kernel(z_real, z_fake, sigma)

        mmd += (k_rr.sum() - k_rr.diag().sum()) / (n*(n-1)) + \
               (k_ff.sum() - k_ff.diag().sum()) / (n*(n-1)) - \
               2 * k_rf.mean()

    return mmd


# Computation of the FID (Fréchet inception distance) to evaluate the images
@torch.no_grad()
def compute_fid(model, loader, num_samples=768, device='cuda'):
    model.eval()
    fid = FrechetInceptionDistance(feature=2048).to(device)
    generated = 0

    while generated < num_samples:
        n = min(64, num_samples - generated)
        fake = model.sample(n)
        fake = ((fake * 0.5 + 0.5).clamp(0, 1) * 255).to(torch.uint8)
        fid.update(fake, real=False)
        generated += n

    for real, _ in loader:
        real = ((real.to(device) * 0.5 + 0.5).clamp(0, 1) * 255).to(torch.uint8)
        fid.update(real, real=True)

    return fid.compute().item()


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = WAE(z_dim=128, lam=10.0, lr=1e-4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=model.lr, betas=(0.5, 0.999))

    epochs = 100
    best_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for x, _ in train_loader:
            x = x.to(device)
            z = model.encoder(x)
            x_rec = model.decoder(z)
            z_prior = torch.randn_like(z)

            recon_loss = F.mse_loss(x_rec, x)
            mmd_loss = compute_mmd(z, z_prior)
            loss = recon_loss + model.lam * mmd_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "best_wae.pth")

    return model


def show_generated(model, n=16):
    model.eval()
    with torch.no_grad():
        samples = model.sample(n).cpu()

    grid = torchvision.utils.make_grid(samples * 0.5 + 0.5, nrow=4)
    plt.figure(figsize=(8, 8))
    plt.imshow(grid.permute(1, 2, 0).clamp(0, 1))
    plt.title("Generated samples")
    plt.axis('off')
    plt.show()


def show_reconstructed(model, n=8):
    x_real, _ = next(iter(test_loader))
    x_real = x_real[:n].to('cuda')
    with torch.no_grad():
        x_rec = model(x_real)

    confronto = torch.cat([x_real.cpu(), x_rec.cpu()])  # 16 images: 8 originals + 8 reconstructed
    grid2 = torchvision.utils.make_grid(confronto * 0.5 + 0.5, nrow=8)
    plt.figure(figsize=(12, 4))
    plt.imshow(grid2.permute(1, 2, 0).clamp(0, 1).numpy())
    plt.title("Sopra: originali — Sotto: ricostruzioni")
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    model = train()
    model.load_state_dict(torch.load("best_wae.pth"))
    model.eval()

    fid_score = compute_fid(model, test_loader, num_samples=1000)
    print(f"FID on test set: {fid_score:.2f}")

    show_generated(model)
    show_reconstructed(model)
