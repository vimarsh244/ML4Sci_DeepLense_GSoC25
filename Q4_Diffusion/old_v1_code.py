import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

class LensingDataset(Dataset):
    """
    Dataset class for single-channel images stored as NumPy files.
    This loader handles different grayscale image shapes:
      - (H, W): Expanded to (1, H, W)
      - (H, W, 1): Transposed to (1, H, W)
      - (H, 1, W): Transposed to (1, H, W)
      - (1, H, W): Already in the correct format
    All images are normalized to [-1, 1].
    """
    def __init__(self, folder, transform=None):
        self.folder = folder
        self.files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.npy')]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        image = np.load(file_path).astype(np.float32)
        # Normalize to [-1, 1]
        image = image / 127.5 - 1.0
        
        # Handle different grayscale image shapes
        if image.ndim == 2:
            # (H, W) -> (1, H, W)
            image = np.expand_dims(image, axis=0)
        elif image.ndim == 3:
            # Already channel-first
            if image.shape[0] == 1:
                pass  # shape is (1, H, W)
            # If stored as (H, W, 1)
            elif image.shape[-1] == 1:
                image = np.transpose(image, (2, 0, 1))
            # If stored as (H, 1, W)
            elif image.shape[1] == 1:
                image = np.transpose(image, (1, 0, 2))
            else:
                # For safety, assume image is (H, W, C) and transpose it.
                image = np.transpose(image, (2, 0, 1))
        else:
            raise ValueError(f"Unsupported image dimensions: {image.shape}")
            
        image = torch.from_numpy(image)
        if self.transform:
            image = self.transform(image)
        return image


def visualize_random_samples(dataset, num_samples=20):
    """
    Visualizes random samples from the dataset.
    Squeezes all singleton dimensions so that grayscale images become (H, W).
    """
    indices = np.random.choice(len(dataset), size=num_samples, replace=False)
    samples = [dataset[i] for i in indices]
    samples = [((img.squeeze().numpy() + 1) / 2).clip(0, 1) for img in samples]
    ncols = 5
    nrows = num_samples // ncols if num_samples % ncols == 0 else num_samples // ncols + 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3 * nrows))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        if i < len(samples):
            ax.imshow(samples[i], cmap='gray')
            ax.set_title(f"Sample {i+1}")
            ax.axis('off')
        else:
            ax.axis('off')
    plt.tight_layout()
    plt.show()

class SinusoidalPosEmb(nn.Module):
    """
    Generates sinusoidal embeddings for scalar timesteps.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb_factor = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb_factor)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class SimpleUnet(nn.Module):
    """
    Simplified U-Net with time conditioning for 1-channel images.
    """
    def __init__(self, in_channels=1, out_channels=1, base_channels=64, time_emb_dim=128):
        super().__init__()
        # Time embedding network
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )
        self.time_emb_proj = nn.Linear(time_emb_dim, base_channels * 4)
        # Downsampling layers
        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        # Upsampling layers with skip connections
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )
        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)
        self.final_activation = nn.Tanh()

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        b, c, _, _ = x3.shape
        t_emb_proj = self.time_emb_proj(t_emb).view(b, c, 1, 1)
        x3 = x3 + t_emb_proj
        u1 = self.up1(x3)
        if u1.shape[2:] != x2.shape[2:]:
            u1 = F.interpolate(u1, size=x2.shape[2:], mode='bilinear', align_corners=False)
        u1 = u1 + x2
        u2 = self.up2(u1) + x1
        out = self.out_conv(u2)
        out = self.final_activation(out)
        return out



def linear_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)


@torch.no_grad()
def sample(model, image_size, device, timesteps, betas, alphas, alpha_hat, channels, num_samples=16):
    """
    Generate samples by performing the reverse diffusion process.
    """
    model.eval()
    x = torch.randn(num_samples, channels, image_size, image_size, device=device)

    for t in reversed(range(timesteps)):
        t_tensor = torch.full((num_samples,), t, device=device, dtype=torch.float)
        predicted_noise = model(x, t_tensor)
        beta = betas[t]
        alpha = alphas[t]
        alpha_hat_t = alpha_hat[t]
        noise = torch.randn_like(x) if t > 0 else 0

        # Reverse diffusion update step (from DDPM)
        x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / torch.sqrt(1 - alpha_hat_t)) * predicted_noise) + torch.sqrt(beta) * noise

    # Print statistics before rescaling
    # print("Generated image tensor stats BEFORE rescale:")
    # print("Mean:", x.mean().item(), "Min:", x.min().item(), "Max:", x.max().item())

    # Rescale images to [0,1]
    x = (x.clamp(-1, 1) + 1) / 2

    # Print statistics after rescaling
    # print("Generated image tensor stats AFTER rescale:")
    # print("Mean:", x.mean().item(), "Min:", x.min().item(), "Max:", x.max().item())

    return x


def save_image(tensor, filename, nrow=4):
    """
    Saves a grid of images to disk.
    Ensures single-channel images are correctly visualized.
    """
    grid = torchvision.utils.make_grid(tensor, nrow=nrow)

    # Debugging print to check saved image values
    print("Saving image with stats:")
    print("Mean:", grid.mean().item(), "Min:", grid.min().item(), "Max:", grid.max().item())

    # If single-channel, squeeze and save using grayscale colormap
    if grid.shape[0] == 1:
        grid = grid.squeeze(0).cpu().numpy()
        plt.imsave(filename, grid, cmap='gray')
    else:
        ndarr = grid.mul(255).byte().cpu().numpy()
        ndarr = np.transpose(ndarr, (1, 2, 0))
        plt.imsave(filename, ndarr)

def train_diffusion_model():
    torch.cuda.empty_cache()
    epochs = 200
    batch_size = 64
    learning_rate = 1e-4
    image_size = 150
    channels = 1
    timesteps = 100

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = LensingDataset('/home/vimarsh/Desktop/3-2/GSoC/ML4Sci/Diffusion_Samples/Samples')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    visualize_random_samples(dataset, num_samples=20)

    model = SimpleUnet(in_channels=channels, out_channels=channels, base_channels=64, time_emb_dim=128).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    betas = linear_beta_schedule(timesteps).to(device)
    alphas = 1 - betas
    alpha_hat = torch.cumprod(alphas, dim=0)

    for epoch in range(epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for images in pbar:
            images = F.interpolate(images, size=(image_size, image_size)).to(device)
            current_batch = images.shape[0]

            # Sample a random timestep for each image
            t = torch.randint(0, timesteps, (current_batch,), device=device).long()
            sqrt_alpha_hat = torch.sqrt(alpha_hat[t]).view(-1, 1, 1, 1)
            sqrt_one_minus_alpha_hat = torch.sqrt(1 - alpha_hat[t]).view(-1, 1, 1, 1)
            noise = torch.randn_like(images)

            # Create noisy images
            noisy_images = sqrt_alpha_hat * images + sqrt_one_minus_alpha_hat * noise

            # Predict the noise
            t_float = t.float()
            predicted_noise = model(noisy_images, t_float)

            # Debugging: Check what the model is predicting
            # print("Predicted Noise stats:")
            # print("Mean:", predicted_noise.mean().item(), "Min:", predicted_noise.min().item(), "Max:", predicted_noise.max().item())

            loss = F.mse_loss(predicted_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=loss.item())

        samples = sample(model, image_size, device, timesteps, betas, alphas, alpha_hat, channels, num_samples=16)
        print("Sample stats:", samples.min().item(), samples.max().item(), samples.mean().item())
        # singular_sample  = sample(model, image_size, device, timesteps, betas, alphas, alpha_hat, channels, num_samples=1)
        # plt.imshow(singular_sample.squeeze().cpu().numpy(), cmap='gray')
        # plt.show()
        save_image(samples, f"samples_epoch_{epoch+1}.png")
        print(f"Epoch {epoch+1} complete. Sample image saved.")

    print("Training complete.")

if __name__ == '__main__':
    train_diffusion_model()
