import os
import glob
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score
import random

import matplotlib.pyplot as plt

# ----------------------------
# Dataset Classes
# ----------------------------

class NpyImageDataset(Dataset):
    """
    General dataset for loading .npy image files from a directory structure.
    Assumes that each class has its own subdirectory.
    """
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir (str): Root directory containing subdirectories for each class.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # List all subdirectories (each is a class)
        class_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        # Create a mapping from class name to integer index
        self.class_to_idx = {class_name: i for i, class_name in enumerate(sorted(class_dirs))}
        
        # Collect all .npy files and assign labels based on folder
        for class_name in class_dirs:
            class_path = os.path.join(data_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            for file_path in glob.glob(os.path.join(class_path, '*.npy')):
                self.image_paths.append(file_path)
                self.labels.append(class_idx)
                
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image from .npy file
        image = np.load(self.image_paths[idx])
        label = self.labels[idx]
        image = torch.from_numpy(image).float()
        # If image is 2D (H x W), add a channel dimension.
        if image.dim() == 2:
            image = image.unsqueeze(0)
        if self.transform:
            image = self.transform(image)
        return image, label

class FilteredNpyImageDataset(Dataset):
    """
    Dataset that filters the samples from NpyImageDataset to only include one target class.
    This is used for pretraining the MAE on the 'no_sub' samples.
    """
    def __init__(self, data_dir, target_class, transform=None):
        base_dataset = NpyImageDataset(data_dir, transform=transform)
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_to_idx = base_dataset.class_to_idx
        target_idx = self.class_to_idx[target_class]
        # Filter for only those samples whose label matches target_idx
        for path, label in zip(base_dataset.image_paths, base_dataset.labels):
            if label == target_idx:
                self.image_paths.append(path)
                self.labels.append(label)
                
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = np.load(self.image_paths[idx])
        label = self.labels[idx]
        image = torch.from_numpy(image).float()
        if image.dim() == 2:
            image = image.unsqueeze(0)
        if self.transform:
            image = self.transform(image)
        return image, label

class NpySuperResolutionDataset(Dataset):
    """
    Dataset for the super-resolution task.
    Expects that low-resolution (LR) and high-resolution (HR) .npy images are stored
    in separate directories but with matching filenames.
    """
    def __init__(self, lr_dir, hr_dir, transform_lr=None, transform_hr=None):
        self.lr_paths = sorted(glob.glob(os.path.join(lr_dir, '*.npy')))
        self.hr_paths = sorted(glob.glob(os.path.join(hr_dir, '*.npy')))
        assert len(self.lr_paths) == len(self.hr_paths), "Mismatch in number of LR and HR files"
        self.transform_lr = transform_lr
        self.transform_hr = transform_hr

    def __len__(self):
        return len(self.lr_paths)

    def __getitem__(self, idx):
        lr = np.load(self.lr_paths[idx])
        hr = np.load(self.hr_paths[idx])
        lr = torch.from_numpy(lr).float()
        hr = torch.from_numpy(hr).float()
        if lr.dim() == 2:
            lr = lr.unsqueeze(0)
        if hr.dim() == 2:
            hr = hr.unsqueeze(0)
        if self.transform_lr:
            lr = self.transform_lr(lr)
        if self.transform_hr:
            hr = self.transform_hr(hr)
        return lr, hr

# ----------------------------
# Utility Function: Patch Masking
# ----------------------------

def apply_patch_mask(img, mask_ratio=0.75, patch_size=16):
    """
    Applies a patch-level mask to an image.
    Divides the image (C, H, W) into patches of size patch_size x patch_size,
    randomly masks a fraction (mask_ratio) of the patches (sets them to 0),
    and returns both the masked image and the binary mask.
    """
    C, H, W = img.shape
    # Ensure H and W are divisible by patch_size
    assert H % patch_size == 0 and W % patch_size == 0, "Image dimensions must be divisible by patch_size"
    grid_h, grid_w = H // patch_size, W // patch_size
    # Create a mask for each patch (True if the patch is to be masked)
    patch_mask = (torch.rand(grid_h, grid_w) < mask_ratio)
    # Expand patch mask to pixel resolution
    mask = patch_mask.unsqueeze(0).repeat(C, 1, 1)
    mask = mask.repeat_interleave(patch_size, dim=1).repeat_interleave(patch_size, dim=2)
    # Create masked image: set masked pixels to 0
    img_masked = img.clone()
    img_masked[mask] = 0.0
    return img_masked, mask

# ----------------------------
# Model Definitions
# ----------------------------

# class MAE(nn.Module):
#     """
#     A simple Masked Autoencoder model.
#     The encoder is a small convolutional network and the decoder upsamples back to the original resolution.
#     The model is trained to reconstruct only the masked portions of the input.
#     """
#     def __init__(self, in_channels=1, latent_dim=128):
#         super(MAE, self).__init__()
#         # Encoder: 3 conv layers with downsampling
#         self.encoder = nn.Sequential(
#             nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),   # -> (32, H/2, W/2)
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),              # -> (64, H/4, W/4)
#             nn.ReLU(),
#             nn.Conv2d(64, latent_dim, kernel_size=3, stride=2, padding=1),        # -> (latent_dim, H/8, W/8)
#             nn.ReLU(),
#         )
#         # Decoder: 3 transposed conv layers for upsampling
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(latent_dim, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # -> (64, H/4, W/4)
#             nn.ReLU(),
#             nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=0),           # -> (32, H/2, W/2)
#             nn.ReLU(),
#             nn.ConvTranspose2d(32, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1),    # -> (in_channels, H, W)
#             nn.Sigmoid()  # To constrain outputs to [0, 1]
#         )
        
#     def forward(self, x):
#         latent = self.encoder(x)
#         recon = self.decoder(latent)
#         return recon, latent

# class ClassificationModel(nn.Module):
#     """
#     Classification model that re-uses the pretrained MAE encoder.
#     Uses adaptive average pooling to avoid issues with fixed image size.
#     """
#     def __init__(self, encoder, latent_dim=128, num_classes=3):
#         super(ClassificationModel, self).__init__()
#         self.encoder = encoder  # Pretrained encoder (will be fine-tuned)
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(latent_dim, num_classes)
        
#     def forward(self, x):
#         features = self.encoder(x)  # shape: (batch, latent_dim, H', W')
#         pooled = self.avgpool(features)
#         pooled = pooled.view(pooled.size(0), -1)
#         out = self.fc(pooled)
#         return out

# class SuperResolutionModel(nn.Module):
#     """
#     Super-resolution model that uses the pretrained MAE encoder and a new decoder for upsampling.
#     The model takes a low-resolution image as input and outputs a high-resolution image.
#     """
#     def __init__(self, encoder, latent_dim=128, in_channels=1):
#         super(SuperResolutionModel, self).__init__()
#         self.encoder = encoder  # Pretrained encoder from MAE
#         # Decoder for super-resolution (simple upsampling network)
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(latent_dim, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
#             nn.ReLU(),
#             nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
#             nn.ReLU(),
#             nn.ConvTranspose2d(32, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
#             nn.Sigmoid()  # Output normalized between 0 and 1
#         )
        
#     def forward(self, x):
#         features = self.encoder(x)
#         out = self.decoder(features)
#         return out

# # ----------------------------
# # Training Functions
# # ----------------------------

# def train_mae(model, dataloader, num_epochs=20, device='cuda', mask_ratio=0.75, patch_size=16):
#     """
#     Train the MAE model using masked reconstruction loss (MSE computed only on masked pixels).
#     """
#     model.train()
#     model.to(device)
#     optimizer = optim.Adam(model.parameters(), lr=1e-3)
#     # We use MSELoss but will manually weight the loss on masked regions.
#     mse_loss = nn.MSELoss(reduction='none')
    
#     for epoch in range(num_epochs):
#         running_loss = 0.0
#         for images, _ in dataloader:
#             images = images.to(device)
#             # Apply patch masking to each image in the batch
#             masked_images = []
#             masks = []
#             for img in images:
#                 img_masked, mask = apply_patch_mask(img, mask_ratio, patch_size)
#                 masked_images.append(img_masked)
#                 masks.append(mask)
#             masked_images = torch.stack(masked_images).to(device)
#             masks = torch.stack(masks).to(device)
            
#             optimizer.zero_grad()
#             recon, _ = model(masked_images)
#             # Resize reconstruction to match original image size if dimensions don't match
#             if recon.shape != images.shape:
#                 recon = F.interpolate(recon, size=(images.shape[2], images.shape[3]), mode='bilinear', align_corners=False)
#             # Compute loss only on masked pixels
#             loss_map = mse_loss(recon, images)
#             loss = (loss_map * masks).sum() / masks.sum()
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item() * images.size(0)
#         epoch_loss = running_loss / len(dataloader.dataset)
#         print(f"[MAE Pretraining] Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
        


#     # Visualize 5 random examples: original, masked, and reconstructed images
#     model.eval()
#     with torch.no_grad():
#         for images, _ in dataloader:
#             # Select 5 random images from the batch
#             indices = random.sample(range(images.shape[0]), 5)
#             originals, masked, reconstructions = [], [], []
#             for i in indices:
#                 img = images[i].to(device)
#                 img_masked, _ = apply_patch_mask(img, mask_ratio, patch_size)
#                 output, _ = model(img_masked.unsqueeze(0))
#                 # Resize if necessary to match original image dimensions
#                 if output.shape != img.unsqueeze(0).shape:
#                     output = F.interpolate(output, size=(img.shape[1], img.shape[2]), mode='bilinear', align_corners=False)
#                 originals.append(img.cpu())
#                 masked.append(img_masked.cpu())
#                 reconstructions.append(output.squeeze(0).cpu())
#             break

#     fig, axes = plt.subplots(5, 3, figsize=(12, 20))
#     for i in range(5):
#         axes[i, 0].imshow(originals[i].squeeze(), cmap='gray')
#         axes[i, 0].set_title("Original")
#         axes[i, 0].axis('off')
#         axes[i, 1].imshow(masked[i].squeeze(), cmap='gray')
#         axes[i, 1].set_title("Masked")
#         axes[i, 1].axis('off')
#         axes[i, 2].imshow(reconstructions[i].squeeze(), cmap='gray')
#         axes[i, 2].set_title("Reconstructed")
#         axes[i, 2].axis('off')
#     plt.tight_layout()
#     plt.show()
#     # plt.savefig('reconstruction.png')
#     return model

# def train_classifier(model, train_loader, val_loader, num_epochs=10, device='cuda'):
#     """
#     Fine-tune the classifier (with pretrained encoder) on the full 3-class dataset.
#     Evaluates using ROC AUC.
#     """
#     model.to(device)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
#     for epoch in range(num_epochs):
#         model.train()
#         running_loss = 0.0
#         for images, labels in train_loader:
#             images, labels = images.to(device), labels.to(device)
#             optimizer.zero_grad()
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item() * images.size(0)
            
#         epoch_loss = running_loss / len(train_loader.dataset)
#         print(f"[Classifier] Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
        
#         # Validation: compute ROC AUC over all validation samples
#         model.eval()
#         all_labels = []
#         all_probs = []
#         with torch.no_grad():
#             for images, labels in val_loader:
#                 images, labels = images.to(device), labels.to(device)
#                 outputs = model(images)
#                 probs = torch.softmax(outputs, dim=1)
#                 all_labels.extend(labels.cpu().numpy())
#                 all_probs.extend(probs.cpu().numpy())
#         # Convert ground truth to one-hot encoding for roc_auc_score
#         num_classes = model.fc.out_features
#         one_hot_labels = np.eye(num_classes)[np.array(all_labels)]
#         try:
#             auc_score = roc_auc_score(one_hot_labels, np.array(all_probs), average='macro')
#         except Exception as e:
#             auc_score = None
#         print(f"[Classifier] Validation ROC AUC: {auc_score:.4f}")
        
# def train_super_resolution(model, dataloader, num_epochs=10, device='cuda'):
#     """
#     Fine-tune the super-resolution model using MSE loss.
#     Also prints out PSNR metric for each epoch.
#     """
#     model.to(device)
#     criterion = nn.MSELoss()
#     optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
#     for epoch in range(num_epochs):
#         model.train()
#         running_loss = 0.0
#         for lr_imgs, hr_imgs in dataloader:
#             lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)
#             optimizer.zero_grad()
#             outputs = model(lr_imgs)
#             loss = criterion(outputs, hr_imgs)
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item() * lr_imgs.size(0)
#         epoch_loss = running_loss / len(dataloader.dataset)
#         # PSNR calculation (assuming images normalized in [0,1])
#         psnr = 10 * math.log10(1.0 / (epoch_loss + 1e-8))
#         print(f"[Super-Resolution] Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.6f}, PSNR: {psnr:.2f} dB")
        


# # ----------------------------
# # Main Execution
# # ----------------------------

# if __name__ == '__main__':
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     print("Using device:", device)
    
#     # Define any necessary transforms (here we assume images are already normalized)
#     transform = None  # You can add normalization/transforms if needed
    
#     # ----------------------------
#     # Task VI.A: MAE Pretraining on "no_sub" samples
#     # ----------------------------
#     # Change the path below to your training dataset directory.
#     mae_data_dir = '/home/vimarsh/Desktop/3-2/GSoC/ML4Sci/dataset/dataset/train'  # Should contain subfolders including "no_sub"
#     target_class = 'no'  # Only use samples with no substructure for pretraining
#     mae_dataset = FilteredNpyImageDataset(mae_data_dir, target_class=target_class, transform=transform)
#     mae_loader = DataLoader(mae_dataset, batch_size=64, shuffle=True, num_workers=4)
    
#     # Initialize the MAE model (assuming images are single-channel)
#     mae_model = MAE(in_channels=1, latent_dim=256)
#     print("Starting MAE pretraining...")
#     mae_model = train_mae(mae_model, mae_loader, num_epochs=50, device=device, mask_ratio=0.25, patch_size=15)
    
#     # Save the pretrained encoder for later use (optional)
#     pretrained_encoder = mae_model.encoder

#     # ----------------------------
#     # Task VI.A (continued): Classification Fine-Tuning
#     # ----------------------------
#     # Full dataset paths for training and validation (3 classes: e.g., no_sub, cdm, axion)
#     train_dir = '/home/vimarsh/Desktop/3-2/GSoC/ML4Sci/dataset/dataset/train'
#     val_dir   = '/home/vimarsh/Desktop/3-2/GSoC/ML4Sci/dataset/dataset/val'
#     dataset_train = NpyImageDataset(train_dir, transform=transform)
#     dataset_val   = NpyImageDataset(val_dir, transform=transform)
#     train_loader_cls = DataLoader(dataset_train, batch_size=160, shuffle=True, num_workers=4)
#     val_loader_cls   = DataLoader(dataset_val, batch_size=160, shuffle=False, num_workers=4)
    
#     # Build classification model using the pretrained encoder.
#     # We use adaptive pooling so that the classifier works with variable image sizes.
#     classifier_model = ClassificationModel(encoder=pretrained_encoder, latent_dim=256, num_classes=3)
#     print("Starting classifier fine-tuning...")
#     train_classifier(classifier_model, train_loader_cls, val_loader_cls, num_epochs=20, device=device)
    
#     # ----------------------------
#     # Task VI.B: Super-Resolution Fine-Tuning
#     # ----------------------------
#     # Directories for low-resolution (LR) and high-resolution (HR) images.
#     lr_dir = '/home/vimarsh/Desktop/3-2/GSoC/ML4Sci/SuperRes_Dataset/Dataset/LR'
#     hr_dir = '/home/vimarsh/Desktop/3-2/GSoC/ML4Sci/SuperRes_Dataset/Dataset/HR'
#     sr_dataset = NpySuperResolutionDataset(lr_dir, hr_dir, transform_lr=transform, transform_hr=transform)
#     sr_loader = DataLoader(sr_dataset, batch_size=64, shuffle=True, num_workers=4)
    
#     # Build super-resolution model using the same pretrained encoder.
#     sr_model = SuperResolutionModel(encoder=pretrained_encoder, latent_dim=256, in_channels=1)
#     print("Starting super-resolution fine-tuning...")
#     train_super_resolution(sr_model, sr_loader, num_epochs=5, device=device)
    
#     # Optionally, save the final models
#     torch.save(mae_model.state_dict(), 'mae_model.pth')
#     torch.save(classifier_model.state_dict(), 'classifier_model.pth')
#     torch.save(sr_model.state_dict(), 'superres_model.pth')
    
#     print("Training complete.")




class MAE(nn.Module):
    """
    A simple Masked Autoencoder model.
    The encoder is a small convolutional network and the decoder upsamples back to the original resolution.
    The model is trained to reconstruct only the masked portions of the input.
    """
    def __init__(self, in_channels=1, latent_dim=128):
        super(MAE, self).__init__()
        # Encoder: 3 conv layers with downsampling
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),   # -> (32, H/2, W/2)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),              # -> (64, H/4, W/4)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, latent_dim, kernel_size=3, stride=2, padding=1),        # -> (latent_dim, H/8, W/8)
            nn.BatchNorm2d(latent_dim),
            nn.ReLU(),
        )
        # Decoder: 3 transposed conv layers for upsampling
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # -> (64, H/4, W/4)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=0),           # -> (32, H/2, W/2)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1),    # -> (in_channels, H, W)
            nn.Sigmoid()  # To constrain outputs to [0, 1]
        )
        
    def forward(self, x):
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return recon, latent

# Improved classifier with more standard blocks and dropout
class ImprovedClassificationModel(nn.Module):
    """
    Improved classification model with standard blocks.
    Uses dropout and batch normalization for better generalization.
    """
    def __init__(self, encoder, latent_dim=128, num_classes=3):
        super(ImprovedClassificationModel, self).__init__()
        # Use the pretrained encoder
        self.encoder = encoder  
        
        # Add a few additional layers to adapt the encoder features
        self.adapt_layers = nn.Sequential(
            nn.Conv2d(latent_dim, latent_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(latent_dim),
            nn.ReLU(),
            nn.Conv2d(latent_dim, latent_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(latent_dim),
            nn.ReLU()
        )
        
        # Global average pooling followed by classifier head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier head with dropout for regularization
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(latent_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),  # Add dropout for regularization
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        features = self.encoder(x)
        features = self.adapt_layers(features)
        # Global average pooling
        pooled = self.avgpool(features)
        # Classification
        output = self.classifier(pooled)
        return output
        
# Fixed super-resolution model to handle size mismatch
class ImprovedSuperResolutionModel(nn.Module):
    """
    Improved super-resolution model that correctly upsamples to the target size.
    """
    def __init__(self, encoder, latent_dim=128, in_channels=1, lr_size=(80, 80), hr_size=(150, 150)):
        super(ImprovedSuperResolutionModel, self).__init__()
        self.encoder = encoder  # Pretrained encoder
        self.lr_size = lr_size
        self.hr_size = hr_size
        
        # Calculate required upscaling factor
        self.scale_factor = hr_size[0] / (lr_size[0] // 8)  # Encoder downsamples by 8x
        
        # Feature processing
        self.process_features = nn.Sequential(
            nn.Conv2d(latent_dim, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        
        # Progressive upsampling to target size
        self.upsampler = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 2x upscale
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # 2x upscale
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),    # 2x upscale
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        # Final layer to adjust channel count and apply final adjustments
        self.final = nn.Sequential(
            nn.Conv2d(32, in_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Encode the input
        features = self.encoder(x)
        # Process the features
        processed = self.process_features(features)
        # Upsample progressively
        upsampled = self.upsampler(processed)
        # Final layer
        out = self.final(upsampled)
        
        # Resize to target HR size if needed
        if out.shape[2:] != self.hr_size:
            out = F.interpolate(out, size=self.hr_size, mode='bilinear', align_corners=False)
            
        return out

# ----------------------------
# Training Functions
# ----------------------------

def train_mae(model, dataloader, num_epochs=20, device='cuda', mask_ratio=0.75, patch_size=16):
    """
    Train the MAE model using masked reconstruction loss (MSE computed only on masked pixels).
    """
    model.train()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # We use MSELoss but will manually weight the loss on masked regions.
    mse_loss = nn.MSELoss(reduction='none')
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, _ in dataloader:
            images = images.to(device)
            # Apply patch masking to each image in the batch
            masked_images = []
            masks = []
            for img in images:
                img_masked, mask = apply_patch_mask(img, mask_ratio, patch_size)
                masked_images.append(img_masked)
                masks.append(mask)
            masked_images = torch.stack(masked_images).to(device)
            masks = torch.stack(masks).to(device)
            
            optimizer.zero_grad()
            recon, _ = model(masked_images)
            # Resize reconstruction to match original image size if dimensions don't match
            if recon.shape != images.shape:
                recon = F.interpolate(recon, size=(images.shape[2], images.shape[3]), mode='bilinear', align_corners=False)
            # Compute loss only on masked pixels
            loss_map = mse_loss(recon, images)
            loss = (loss_map * masks).sum() / masks.sum()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        epoch_loss = running_loss / len(dataloader.dataset)
        print(f"[MAE Pretraining] Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.6f}")
        


    # Visualize 5 random examples: original, masked, and reconstructed images
    model.eval()
    with torch.no_grad():
        for images, _ in dataloader:
            # Select 5 random images from the batch
            indices = random.sample(range(images.shape[0]), 5)
            originals, masked, reconstructions = [], [], []
            for i in indices:
                img = images[i].to(device)
                img_masked, _ = apply_patch_mask(img, mask_ratio, patch_size)
                output, _ = model(img_masked.unsqueeze(0))
                # Resize if necessary to match original image dimensions
                if output.shape != img.unsqueeze(0).shape:
                    output = F.interpolate(output, size=(img.shape[1], img.shape[2]), mode='bilinear', align_corners=False)
                originals.append(img.cpu())
                masked.append(img_masked.cpu())
                reconstructions.append(output.squeeze(0).cpu())
            break

    fig, axes = plt.subplots(5, 3, figsize=(12, 20))
    for i in range(5):
        axes[i, 0].imshow(originals[i].squeeze(), cmap='gray')
        axes[i, 0].set_title("Original")
        axes[i, 0].axis('off')
        axes[i, 1].imshow(masked[i].squeeze(), cmap='gray')
        axes[i, 1].set_title("Masked")
        axes[i, 1].axis('off')
        axes[i, 2].imshow(reconstructions[i].squeeze(), cmap='gray')
        axes[i, 2].set_title("Reconstructed")
        axes[i, 2].axis('off')
    plt.tight_layout()
    plt.show()
    # save image
    plt.savefig('reconstruction_results.png')
    # plt.savefig('reconstruction.png')
    return model

def train_classifier(model, train_loader, val_loader, num_epochs=10, device='cuda'):
    """
    Improved training function for the classifier with learning rate scheduling,
    early stopping, and model checkpointing.
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Use SGD with momentum instead of Adam for better convergence
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    
    # Learning rate scheduler - reduce LR when plateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
    
    best_auc = 0.0
    best_model_state = None
    patience = 5
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            running_loss += loss.item() * images.size(0)
            
        train_loss = running_loss / len(train_loader.dataset)
        train_acc = 100.0 * correct / total
        
        # Validation phase
        model.eval()
        all_labels = []
        all_probs = []
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                
                # Calculate accuracy
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
                # For AUC calculation
                probs = torch.softmax(outputs, dim=1)
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = 100.0 * val_correct / val_total
        
        # Convert ground truth to one-hot encoding for roc_auc_score
        num_classes = len(set(all_labels))
        one_hot_labels = np.eye(num_classes)[np.array(all_labels)]
        try:
            auc_score = roc_auc_score(one_hot_labels, np.array(all_probs), average='macro')
        except Exception as e:
            print(f"Warning: Error calculating AUC: {str(e)}")
            auc_score = 0.0
        
        # Print metrics
        print(f"[Classifier] Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, "
              f"ROC AUC: {auc_score:.4f}")
        
        # Update learning rate based on AUC
        scheduler.step(auc_score)
        
        # Check if this is the best model (based on AUC)
        if auc_score > best_auc:
            best_auc = auc_score
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Load the best model weights
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model with AUC: {best_auc:.4f}")
    
    return model

# Improved super-resolution training function
def train_improved_super_resolution(model, dataloader, num_epochs=10, device='cuda'):
    """
    Improved training function for super-resolution with L1 loss component
    and proper handling of output size.
    """
    model.to(device)
    # Combine MSE and L1 loss for better results
    mse_criterion = nn.MSELoss()
    l1_criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=2e-4, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    
    for epoch in range(num_epochs):
        model.train()
        running_mse_loss = 0.0
        running_l1_loss = 0.0
        running_total_loss = 0.0
        
        for lr_imgs, hr_imgs in dataloader:
            lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)
            
            optimizer.zero_grad()
            outputs = model(lr_imgs)
            
            # Ensure output size matches target size
            if outputs.shape != hr_imgs.shape:
                outputs = F.interpolate(outputs, size=(hr_imgs.shape[2], hr_imgs.shape[3]), 
                                        mode='bilinear', align_corners=False)
                
            # Calculate losses
            mse_loss = mse_criterion(outputs, hr_imgs)
            l1_loss = l1_criterion(outputs, hr_imgs)
            # Combined loss (MSE with L1 regularization)
            loss = mse_loss + 0.5 * l1_loss
            
            loss.backward()
            optimizer.step()
            
            # Track metrics
            running_mse_loss += mse_loss.item() * lr_imgs.size(0)
            running_l1_loss += l1_loss.item() * lr_imgs.size(0)
            running_total_loss += loss.item() * lr_imgs.size(0)
            
        # Compute epoch losses
        epoch_mse_loss = running_mse_loss / len(dataloader.dataset)
        epoch_l1_loss = running_l1_loss / len(dataloader.dataset)
        epoch_total_loss = running_total_loss / len(dataloader.dataset)
        
        # PSNR calculation (assuming images normalized in [0,1])
        psnr = 10 * math.log10(1.0 / (epoch_mse_loss + 1e-8))
        
        # Update learning rate
        scheduler.step()
        
        print(f"[Super-Resolution] Epoch [{epoch+1}/{num_epochs}], "
              f"MSE Loss: {epoch_mse_loss:.6f}, L1 Loss: {epoch_l1_loss:.6f}, "
              f"Total Loss: {epoch_total_loss:.6f}, PSNR: {psnr:.2f} dB")
    
    # Visualize a few examples
    model.eval()
    with torch.no_grad():
        for lr_imgs, hr_imgs in dataloader:
            lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)
            sr_imgs = model(lr_imgs)
            
            # Ensure correct size for visualization
            if sr_imgs.shape != hr_imgs.shape:
                sr_imgs = F.interpolate(sr_imgs, size=(hr_imgs.shape[2], hr_imgs.shape[3]), 
                                        mode='bilinear', align_corners=False)
            
            # Pick a few samples to visualize
            n_samples = min(3, lr_imgs.size(0))
            
            fig, axes = plt.subplots(n_samples, 3, figsize=(15, 5*n_samples))
            for i in range(n_samples):
                # Low-resolution input
                axes[i, 0].imshow(lr_imgs[i].cpu().squeeze(0), cmap='gray')
                axes[i, 0].set_title("Low Resolution")
                axes[i, 0].axis('off')
                
                # Super-resolution output
                axes[i, 1].imshow(sr_imgs[i].cpu().squeeze(0), cmap='gray')
                axes[i, 1].set_title("Super Resolution")
                axes[i, 1].axis('off')
                
                # High-resolution ground truth
                axes[i, 2].imshow(hr_imgs[i].cpu().squeeze(0), cmap='gray')
                axes[i, 2].set_title("High Resolution")
                axes[i, 2].axis('off')
            
            plt.tight_layout()
            plt.show()
            plt.savefig('super_resolution_results.png')
            break
            
    return model




if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)
    
    # Define any necessary transforms (here we assume images are already normalized)
    transform = None  # You can add normalization/transforms if needed
    
    mae_data_dir = '/home/vimarsh/Desktop/3-2/GSoC/ML4Sci/dataset/dataset/train'  # Should contain subfolders including "no_sub"
    target_class = 'no'  # Only use samples with no substructure for pretraining
    mae_dataset = FilteredNpyImageDataset(mae_data_dir, target_class=target_class, transform=transform)
    mae_loader = DataLoader(mae_dataset, batch_size=64, shuffle=True, num_workers=4)
    
    # Initialize the MAE model (assuming images are single-channel)
    mae_model = MAE(in_channels=1, latent_dim=256)
    print("Starting MAE pretraining...")
    mae_model = train_mae(mae_model, mae_loader, num_epochs=15, device=device, mask_ratio=0.30, patch_size=15)
    
    # Get the pretrained encoder for later use
    pretrained_encoder = mae_model.encoder

    # ----------------------------
    # Task VI.A (continued): Classification Fine-Tuning with Improved Model
    # ----------------------------
    # Full dataset paths for training and validation
    train_dir = '/home/vimarsh/Desktop/3-2/GSoC/ML4Sci/dataset/dataset/train'
    val_dir   = '/home/vimarsh/Desktop/3-2/GSoC/ML4Sci/dataset/dataset/val'
    dataset_train = NpyImageDataset(train_dir, transform=transform)
    dataset_val   = NpyImageDataset(val_dir, transform=transform)
    train_loader_cls = DataLoader(dataset_train, batch_size=64, shuffle=True, num_workers=4)
    val_loader_cls   = DataLoader(dataset_val, batch_size=64, shuffle=False, num_workers=4)
    
    # Build improved classification model using the pretrained encoder
    classifier_model = ImprovedClassificationModel(encoder=pretrained_encoder, latent_dim=256, num_classes=3)
    
    # Freeze the encoder for a few epochs to allow the new layers to adapt
    for param in classifier_model.encoder.parameters():
        param.requires_grad = False
    
    print("Starting classifier fine-tuning with frozen encoder...")
    classifier_model = train_classifier(classifier_model, train_loader_cls, val_loader_cls, num_epochs=15, device=device)
    
    # Now unfreeze the encoder and continue training with a lower learning rate
    for param in classifier_model.encoder.parameters():
        param.requires_grad = True
    
    print("Fine-tuning the entire model...")
    classifier_model = train_classifier(classifier_model, train_loader_cls, val_loader_cls, num_epochs=20, device=device)
    
    # ----------------------------
    # Task VI.B: Super-Resolution Fine-Tuning with Improved Model
    # ----------------------------
    # Directories for low-resolution (LR) and high-resolution (HR) images.
    lr_dir = '/home/vimarsh/Desktop/3-2/GSoC/ML4Sci/SuperRes_Dataset/Dataset/LR'
    hr_dir = '/home/vimarsh/Desktop/3-2/GSoC/ML4Sci/SuperRes_Dataset/Dataset/HR'
    sr_dataset = NpySuperResolutionDataset(lr_dir, hr_dir, transform_lr=transform, transform_hr=transform)
    sr_loader = DataLoader(sr_dataset, batch_size=64, shuffle=True, num_workers=4)

    # Build improved super-resolution model using the pretrained encoder
    sr_model = ImprovedSuperResolutionModel(encoder=pretrained_encoder, latent_dim=256, in_channels=1)
    print("Starting super-resolution fine-tuning...")
    sr_model = train_improved_super_resolution(sr_model, sr_loader, num_epochs=20, device=device)

        
    # Optionally, save the final models
    torch.save(mae_model.state_dict(), 'mae_model.pth')
    torch.save(classifier_model.state_dict(), 'classifier_model.pth')
    torch.save(sr_model.state_dict(), 'superres_model.pth')

    print("Training complete.")
