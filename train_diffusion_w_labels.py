import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import glob
import math
from sprites import create_sample_sprites
class PixelArtDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        """
        Dataset for loading 16x16 pixel art sprites
        """
        self.image_paths = glob.glob(os.path.join(image_dir, "*.png")) + \
                          glob.glob(os.path.join(image_dir, "*.jpg"))
        self.transform = transform
        self.shapes = ['rectangle', 'circle', 'line', 'face']  # 4 types
        self.colours = ['red', 'green', 'blue', 'yellow', 'magenta', 'cyan', 'orange', 'purple']  # 8 colours
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        file_name = self.image_paths[idx]
        image = Image.open(file_name).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # get shape and colour using file name
        split = file_name.rsplit("_")
        shape = split[1]
        colour = split[2]
        shape_idx = self.shapes.index(shape)
        colour_idx = self.colours.index(colour)
        return image, shape_idx, colour_idx

class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.1):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        
        # find largest divisor <= 8
        def get_num_groups(channels):
            for groups in [8, 4, 2, 1]:
                if channels % groups == 0:
                    return groups
            return 1
        
        self.block1 = nn.Sequential(
            nn.GroupNorm(get_num_groups(in_channels), in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )
        
        self.block2 = nn.Sequential(
            nn.GroupNorm(get_num_groups(out_channels), out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x, time_emb):
        h = self.block1(x)
        
        # Add time embedding
        time_emb = self.time_mlp(time_emb)
        h = h + time_emb[:, :, None, None]
        
        h = self.block2(h)
        return h + self.shortcut(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, time_emb_dim=128, num_shape_types=4, num_colours=8):
        super().__init__()
        
        self.time_emb_dim = time_emb_dim
        self.time_embedding = SinusoidalPositionEmbedding(time_emb_dim)

        # Condition embeddings
        self.object_embedding = nn.Embedding(num_shape_types, time_emb_dim // 2)
        self.color_embedding = nn.Embedding(num_colours, time_emb_dim // 2)
        
        combined_emb_dim = time_emb_dim + time_emb_dim  # time_emb_dim for time, time_emb_dim for conditions
        # Time embedding MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim)
        )

        self.condition_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim)
        )
        
        # Encoder (downsampling)
        self.down1 = ResidualBlock(in_channels, 64, combined_emb_dim)
        self.down2 = ResidualBlock(64, 128, combined_emb_dim)
        
        # Bottleneck
        self.bottleneck = ResidualBlock(128, 128, combined_emb_dim)
        
        # Decoder (upsampling)
        self.up1 = ResidualBlock(128 + 128, 64, combined_emb_dim)  # Skip connection
        self.up2 = ResidualBlock(64 + 64, 32, combined_emb_dim)   # Skip connection
        
        # Final output
        self.final = nn.Conv2d(32, out_channels, 1)
        
        # For 16x16, we only do 2x downsampling to keep spatial information
        self.down_sample = nn.MaxPool2d(2)
        self.up_sample = nn.Upsample(scale_factor=2, mode='nearest')
    
    def forward(self, x, timestep, shape_type=None, colour=None):
        # Time embedding
        t = self.time_embedding(timestep)
        t = self.time_mlp(t)
        
        # Condition embeddings
        if shape_type is not None and colour is not None:
            sha_emb = self.object_embedding(shape_type)
            col_emb = self.color_embedding(colour)
            # concatenate shape and color embeddings
            cond_emb = torch.cat([sha_emb, col_emb], dim=-1)
            cond_emb = self.condition_mlp(cond_emb)
            # combine time and condition embeddings
            combined_emb = torch.cat([t, cond_emb], dim=-1)
        else:
            # If no conditions provided, use zeros for condition part
            batch_size = t.shape[0]
            zero_cond = torch.zeros(batch_size, self.time_emb_dim, device=t.device)
            combined_emb = torch.cat([t, zero_cond], dim=-1)

        # Encoder
        h1 = self.down1(x, combined_emb)  # 16x16 -> 16x16
        h1_down = self.down_sample(h1)  # 16x16 -> 8x8
        
        h2 = self.down2(h1_down, combined_emb)  # 8x8 -> 8x8
        h2_down = self.down_sample(h2)  # 8x8 -> 4x4
        
        # Bottleneck
        h = self.bottleneck(h2_down, combined_emb)  # 4x4 -> 4x4
        
        # Decoder with skip connections
        h = self.up_sample(h)  # 4x4 -> 8x8
        h = torch.cat([h, h2], dim=1)  # Skip connection
        h = self.up1(h, combined_emb)  # 8x8 -> 8x8
        
        h = self.up_sample(h)  # 8x8 -> 16x16
        h = torch.cat([h, h1], dim=1)  # Skip connection
        h = self.up2(h, combined_emb)  # 16x16 -> 16x16
        
        return self.final(h)

class DDPMScheduler:
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02):
        self.num_timesteps = num_timesteps
        
        # Linear beta schedule
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alpha_cumprod_prev = F.pad(self.alpha_cumprod[:-1], (1, 0), value=1.0)
        
        # Calculations for diffusion q(x_t | x_0) and others
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - self.alpha_cumprod)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1 - self.alpha_cumprod_prev) / (1 - self.alpha_cumprod)
    
    def add_noise(self, x0, timesteps, noise=None):
        """Forward diffusion process"""
        if noise is None:
            noise = torch.randn_like(x0)
        
        sqrt_alpha_cumprod_t = self.sqrt_alpha_cumprod[timesteps].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alpha_cumprod[timesteps].reshape(-1, 1, 1, 1)
        
        return sqrt_alpha_cumprod_t * x0 + sqrt_one_minus_alpha_cumprod_t * noise
    
    def sample_prev_timestep(self, xt, noise_pred, timestep):
        """Reverse diffusion step"""
        alpha_t = self.alphas[timestep]
        alpha_cumprod_t = self.alpha_cumprod[timestep]
        alpha_cumprod_t_prev = self.alpha_cumprod_prev[timestep]
        
        beta_t = self.betas[timestep]
        
        # Predict x0
        pred_x0 = (xt - torch.sqrt(1 - alpha_cumprod_t) * noise_pred) / torch.sqrt(alpha_cumprod_t)
        pred_x0 = torch.clamp(pred_x0, -1, 1)
        
        # Compute mean
        pred_mean = (torch.sqrt(alpha_cumprod_t_prev) * beta_t) / (1 - alpha_cumprod_t) * pred_x0
        pred_mean += (torch.sqrt(alpha_t) * (1 - alpha_cumprod_t_prev)) / (1 - alpha_cumprod_t) * xt
        
        if timestep == 0:
            return pred_mean
        else:
            variance = self.posterior_variance[timestep]
            noise = torch.randn_like(xt)
            return pred_mean + torch.sqrt(variance) * noise

class PixelArtDiffusion:
    def __init__(self, device='cpu', num_timesteps=1000):
        self.device = device
        self.num_timesteps = num_timesteps
        
        # Initialize model and scheduler
        self.model = UNet(num_shape_types=4, num_colours=8).to(device)
        self.scheduler = DDPMScheduler(num_timesteps)
        
        # Move scheduler tensors to device
        for attr_name in ['betas', 'alphas', 'alpha_cumprod', 'alpha_cumprod_prev',
                         'sqrt_alpha_cumprod', 'sqrt_one_minus_alpha_cumprod', 'posterior_variance']:
            setattr(self.scheduler, attr_name, getattr(self.scheduler, attr_name).to(device))
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-4)
    
        self.shapes = ['rectangle', 'circle', 'line', 'face']
        self.colours = ['red', 'green', 'blue', 'yellow', 'magenta', 'cyan', 'orange', 'purple']

    def train_step(self, batch, shapes=None, colours=None):
        """Single training step"""
        batch = batch.to(self.device)
        shapes = shapes.to(self.device)
        colours = colours.to(self.device)
        batch_size = batch.shape[0]
        
        # sample random timesteps
        timesteps = torch.randint(0, self.num_timesteps, (batch_size,), device=self.device)
        
        # add noise to image
        noise = torch.randn_like(batch)
        noisy_images = self.scheduler.add_noise(batch, timesteps, noise)
        
        # predict noise using model
        self.optimizer.zero_grad()
        noise_pred = self.model(noisy_images, timesteps, shapes, colours)
        
        loss = F.mse_loss(noise_pred, noise)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    @torch.no_grad()
    def sample(self, num_samples=16, shape_types=None, colours=None):
        """Generate samples using DDPM sampling"""
        self.model.eval()
        
        # Start with random noise
        shape = (num_samples, 3, 16, 16)
        img = torch.randn(shape, device=self.device)

        if shape_types is not None:
            if isinstance(shape_types, (int, list)):
                if isinstance(shape_types, int):
                    shape_types = [shape_types] * num_samples
                shape_types = torch.tensor(shape_types, device=self.device)
        
        if colours is not None:
            if isinstance(colours, (int, list)):
                if isinstance(colours, int):
                    colours = [colours] * num_samples
                colours = torch.tensor(colours, device=self.device)
        
        # Reverse diffusion process
        for t in reversed(range(self.num_timesteps)):
            timestep = torch.full((num_samples,), t, device=self.device, dtype=torch.long)
            
            # Predict noise
            noise_pred = self.model(img, timestep, shape_types, colours)
            
            # Remove noise
            img = self.scheduler.sample_prev_timestep(img, noise_pred, t)
        
        self.model.train()
        return img
    
    def save_samples(self, epoch, save_dir='samples', num_samples=16):
        """Save generated samples"""
        os.makedirs(save_dir, exist_ok=True)

        num_shapes = len(self.shapes)
        num_colours = len(self.colours)
        
        # Create a grid: rows = shapes, cols = colors
        shape_grid = []
        colour_grid = []
        
        for shape_idx in range(num_shapes):
            for colour_idx in range(min(4, num_colours)):  # show first 4 colors
                shape_grid.append(shape_idx)
                colour_grid.append(colour_idx)
        
        # generate samples
        samples = self.sample(
            num_samples=len(shape_grid),
            shape_types=shape_grid,
            colours=colour_grid)
        
        samples = (samples + 1) / 2
        samples = torch.clamp(samples, 0, 1)
        
        # create grid
        fig, axes = plt.subplots(num_shapes, 4, figsize=(8, 8))
        for i, (obj_idx, color_idx) in enumerate(zip(shape_grid, colour_grid)):
            row = obj_idx
            col = color_idx
            
            img = samples[i].cpu().permute(1, 2, 0).numpy()
            axes[row, col].imshow(img)
            axes[row, col].axis('off')
            axes[row, col].set_title(f'{self.shapes[obj_idx]}\n{self.colours[color_idx]}', fontsize=8)
        
        plt.suptitle(f'Generated Pixel Art - Epoch {epoch}')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/epoch_{epoch:04d}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved samples for epoch {epoch}")

def main():
    save_directory = "samples_diffusion/w_labels"
    # create sample data if needed
    # if not os.path.exists('sprites') or len(os.listdir('sprites')) < 10:
    create_sample_sprites()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"using: {device}")
    
    # data preprocessing
    transform = transforms.Compose([
        transforms.Resize((16, 16)),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
    ])
    
    # initialise dataset
    dataset = PixelArtDataset('sprites', transform=transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)  # Smaller batch for diffusion
    
    print(f"loaded {len(dataset)} sprites")
    
    # initialize diffusion model
    diffusion = PixelArtDiffusion(device=device, num_timesteps=1000)
    
    num_epochs = 500
    print("starting diffusion model training")

    # train model
    for epoch in range(num_epochs):
        losses = []
        
        for batch in dataloader:
            images, shapes, colors = batch
            loss = diffusion.train_step(images, shapes, colors)
            losses.append(loss)
        
        avg_loss = np.mean(losses)
        print(f'Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.6f}')
        
        # Save samples
        if (epoch + 1) % 20 == 0 or epoch == 0:
            diffusion.save_samples(epoch + 1, save_dir=save_directory)
    
    print("training complete")
    print(f"Generated samples saved in {save_directory} directory")
    #generate samples at the end of training
    diffusion.save_samples(epoch, save_dir=save_directory)

if __name__ == "__main__":
    main()