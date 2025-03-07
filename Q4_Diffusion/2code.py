import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Dataset
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class NpyImageDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir (string): Directory with all the .npy image files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = glob.glob(os.path.join(data_dir, '*.npy'))
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image = np.load(self.image_paths[idx])
        # Convert to torch tensor
        image = torch.from_numpy(image).float()
        # Apply transformations if any
        if self.transform:
            image = self.transform(image)
        return image

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(45),
    transforms.ToTensor()
])
# transforms to inc model performanc - more general 

data_dir = '/home/vimarsh/Desktop/3-2/GSoC/ML4Sci/Diffusion_Samples/Samples'
dataset = NpyImageDataset(data_dir=data_dir, transform=transform)
train_size = int(1.0 * len(dataset))
val_size = len(dataset) - train_size #no val in this (doesn;t make sense)
dataset_train, dataset_val = random_split(dataset, [train_size, val_size])
print(f"Training samples: {len(dataset_train)}")
print(f"Validation samples: {len(dataset_val)}")

batch_size = 24 # 4050mobile is smol
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)


## DIFFUSION MODEL (thanks to resources as per in readme)

# Sinusoidal positional embedding for timesteps
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        # x is a tensor of shape [batch]
        device = x.device
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb

# Residual block that incorporates time embedding
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)

        # If channel numbers differ, project input for residual addition
        self.res_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x, t):
        h = self.relu(self.conv1(x))

        # Process and add time embedding
        time_emb = self.relu(self.time_mlp(t)).unsqueeze(-1).unsqueeze(-1)

        h = h + time_emb
        h = self.conv2(self.relu(h))
        res = self.res_conv(x) if self.res_conv is not None else x
        return h + res

# Downsampling block from U-Net
class Down(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.resblock = ResidualBlock(in_channels, out_channels, time_emb_dim)
        self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x, t):
        x = self.resblock(x, t)
        skip = x  # Save for skip connection
        x = self.downsample(x)
        return skip, x

# upsampling block that accepts the skip connection channel count explicitly
class Up(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, time_emb_dim, output_padding=0):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, output_padding=output_padding)

        # input channels for the residual block is concatenation of the upsampled features and the skip connection
        self.resblock = ResidualBlock(skip_channels + out_channels, out_channels, time_emb_dim)

    def forward(self, x, skip, t):
        x = self.upsample(x)

        # concatanating skip connection along channel dimension
        x = torch.cat([x, skip], dim=1)
        x = self.resblock(x, t)
        return x

# the U-Net architecture for noise prediction
class UNet(nn.Module):
    def __init__(self, time_emb_dim=128):
        super().__init__()
        # Time embedding network
        self.time_embedding = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )

        # Initial convolution
        self.init_conv = nn.Conv2d(1, 64, kernel_size=3, padding=1)

        # Downsampling layers
        self.down1 = Down(64, 128, time_emb_dim)  # Produces skip1 with 128 channels
        self.down2 = Down(128, 256, time_emb_dim)  # Produces skip2 with 256 channels

        # Bottleneck residual block
        self.bot1 = ResidualBlock(256, 256, time_emb_dim)

        # Upsampling layers
        # For up1: input is from bottleneck (256 channels), skip2 has 256 channels, output desired is 128 channels.
        self.up1 = Up(256, skip_channels=256, out_channels=128, time_emb_dim=time_emb_dim, output_padding=1)
        # For up2: input is from up1 (128 channels), skip1 has 128 channels, output desired is 64 channels.
        self.up2 = Up(128, skip_channels=128, out_channels=64, time_emb_dim=time_emb_dim, output_padding=0)

        # Final convolution layer to predict noise (output same channels as input)
        self.out_conv = nn.Conv2d(64, 1, kernel_size=1)
    
    def forward(self, x, t):
        # Get time embeddings
        t = self.time_embedding(t)

        # Initial convolution
        x0 = self.init_conv(x)

        # Downsampling path
        skip1, x1 = self.down1(x0, t)
        skip2, x2 = self.down2(x1, t)

        # Bottleneck
        x_bot = self.bot1(x2, t)

        # Upsampling path
        x_up1 = self.up1(x_bot, skip2, t)
        x_up2 = self.up2(x_up1, skip1, t)

        # Final output: predicted noise
        out = self.out_conv(x_up2)
        return out

# hyperparameters
num_timesteps = 1500
beta_start = 1e-4
beta_end = 0.02
betas = torch.linspace(beta_start, beta_end, num_timesteps)
alphas = 1 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)


# precomputing square roots for efficiency
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)

def q_sample(x0, t, noise=None):
    """
    Add noise to x0 at timestep t.
    x0: Original images, shape [batch, channels, height, width].
    t: Timesteps for each image in batch, shape [batch].
    noise: Optional noise tensor. If None, sampled from standard normal.
    """
    if noise is None:
        noise = torch.randn_like(x0)
    
    #selecting the correct alpha and beta values for the timestep 
    sqrt_alphas_cumprod_t = sqrt_alphas_cumprod.to(x0.device)[t].reshape(-1, 1, 1, 1)
    sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod.to(x0.device)[t].reshape(-1, 1, 1, 1)
    return sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(time_emb_dim=128).to(device)
optimizer = optim.Adam(model.parameters(), lr=5e-5)
num_epochs = 25

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        batch = batch.to(device)

        # sampling random timesteps for each image in the batch
        t = torch.randint(0, num_timesteps, (batch.size(0),), device=device).long()

        noise = torch.randn_like(batch)
        # create the noised image at timestep t
        x_noisy = q_sample(batch, t, noise)

        #  noise prediction from the model
        noise_pred = model(x_noisy, t.float())
        loss = F.mse_loss(noise_pred, noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    avg_loss = epoch_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

#save model
torch.save(model.state_dict(), 'old_diffusion_model_t128_epoch50.pth')

#load model
model.load_state_dict(torch.load('old_diffusion_model_t128_epoch50.pth'))
# model = UNet(time_emb_dim=196).to(device)

## just testing (sample images)

@torch.no_grad()
def sample(model, image_size, device, num_steps=num_timesteps):
    model.eval()
    
    #some random noise
    x = torch.randn((1, 1, image_size, image_size), device=device)
    for t in reversed(range(num_steps)):
        t_batch = torch.tensor([t], device=device).float()

        # predicting noise using the model
        noise_pred = model(x, t_batch)
        beta_t = betas[t].to(device)
        sqrt_one_minus_alpha_t = sqrt_one_minus_alphas_cumprod[t].to(device)

        # the coefficient to recover x0
        sqrt_recip = 1.0 / torch.sqrt(1 - beta_t)
        x = sqrt_recip * (x - beta_t / sqrt_one_minus_alpha_t * noise_pred)
        if t > 0:
            noise = torch.randn_like(x)
            sigma_t = torch.sqrt(beta_t)
            x = x + sigma_t * noise
    return x

# to generate and display a sample image
# sampled_img = sample(model, image_size=150, device=device)
# sampled_img = sampled_img.squeeze().cpu().numpy()
# plt.imshow(sampled_img, cmap='gray')
# plt.axis('off')
# plt.title("Generated Strong Gravitational Lensing Image")
# plt.show()

#plot n images in a grid square
def plot_grid(images, rows, cols, title):
    fig, axs = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
    fig.suptitle(title)
    for i in range(rows):
        for j in range(cols):
            axs[i, j].imshow(images[i*cols + j], cmap='gray')
            axs[i, j].axis('off')
    plt.savefig('diffusion_samples.png')
    plt.show()

plot_grid([sample(model, image_size=150, device=device).squeeze().cpu().numpy() for _ in range(16)], 4, 4, "Some Strong Gravitational Lensing Images")

