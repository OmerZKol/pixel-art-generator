import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import glob
from sprites import create_sample_sprites
class PixelArtDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        """
        dataset for loading 16x16 pixel art
        """
        self.image_paths = glob.glob(os.path.join(image_dir, "*.png")) + \
                          glob.glob(os.path.join(image_dir, "*.jpg"))
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

class Generator(nn.Module):
    def __init__(self, latent_dim=64, output_channels=3):
        super(Generator, self).__init__()
        
        # generator for 16x16 images
        self.latent_dim = latent_dim
        
        # start with 1x1, expand to 16x16
        self.main = nn.Sequential(
            # input shape: latent_dim x 1 x 1
            nn.ConvTranspose2d(latent_dim, 128, 4, 1, 0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # 128 x 4 x 4
            
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # 64 x 8 x 8
            
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            # 32 x 16 x 16
            
            nn.Conv2d(32, output_channels, 3, 1, 1, bias=False),
            nn.Tanh()
            # output shape: 3 x 16 x 16
        )
        
    def forward(self, noise):
        return self.main(noise)

class Discriminator(nn.Module):
    def __init__(self, input_channels=3):
        super(Discriminator, self).__init__()
        
        # discriminator for 16x16 images
        self.main = nn.Sequential(
            # input shape of: 3 x 16 x 16
            nn.Conv2d(input_channels, 32, 4, 2, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            # 32 x 8 x 8
            
            nn.Conv2d(32, 64, 4, 2, 1, bias=True),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # 64 x 4 x 4
            
            nn.Conv2d(64, 128, 4, 2, 1, bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # 128 x 2 x 2
            
            nn.Conv2d(128, 1, 2, 1, 0, bias=True),
            nn.Sigmoid()
            # output shape: 1 x 1 x 1
        )
        
    def forward(self, image):
        return self.main(image).view(-1, 1).squeeze(1)

class PixelArtGAN:
    def __init__(self, latent_dim=64, lr=0.0002, device='cpu'):
        self.device = device
        self.latent_dim = latent_dim
        
        # the 2 networks
        self.generator = Generator(latent_dim).to(device)
        self.discriminator = Discriminator().to(device)
        
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        
        # the loss func
        self.criterion = nn.BCELoss()
        
        # fixed noise for consistent sample generation
        self.fixed_noise = torch.randn(16, latent_dim, 1, 1, device=device)
        
    def train_step(self, real_images):
        batch_size = real_images.size(0)
        
        # labels
        real_labels = torch.ones(batch_size, device=self.device)
        fake_labels = torch.zeros(batch_size, device=self.device)
        
        self.d_optimizer.zero_grad()
        
        # real images
        real_output = self.discriminator(real_images)
        d_loss_real = self.criterion(real_output, real_labels)
        
        # fake images
        noise = torch.randn(batch_size, self.latent_dim, 1, 1, device=self.device)
        fake_images = self.generator(noise)
        fake_output = self.discriminator(fake_images.detach())
        d_loss_fake = self.criterion(fake_output, fake_labels)
        
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        self.d_optimizer.step()
        
        self.g_optimizer.zero_grad()
        
        fake_output = self.discriminator(fake_images)
        g_loss = self.criterion(fake_output, real_labels)
        g_loss.backward()
        self.g_optimizer.step()
        
        return d_loss.item(), g_loss.item()
    
    def generate_samples(self, num_samples=16):
        with torch.no_grad():
            noise = torch.randn(num_samples, self.latent_dim, 1, 1, device=self.device)
            fake_images = self.generator(noise)
            return fake_images
    
    def save_sample_grid(self, epoch, save_dir='samples'):
        """save a grid of generated samples"""
        os.makedirs(save_dir, exist_ok=True)
        
        with torch.no_grad():
            fake_images = self.generator(self.fixed_noise)
            
        # change range of values from [-1, 1] to [0, 1]
        fake_images = (fake_images + 1) / 2
        
        # create grid
        fig, axes = plt.subplots(4, 4, figsize=(8, 8))
        for i, ax in enumerate(axes.flat):
            if i < len(fake_images):
                img = fake_images[i].cpu().permute(1, 2, 0).numpy()
                ax.imshow(img)
                ax.axis('off')
        
        plt.suptitle(f'Generated Pixel Art - Epoch {epoch}')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/epoch_{epoch:04d}.png', dpi=150, bbox_inches='tight')
        plt.close()

def main():
    save_directory = "samples_gan"
    # Create sample data if it doesn't exist
    if not os.path.exists('sprites') or len(os.listdir('sprites')) == 0:
        print("Creating sample sprite dataset...")
        create_sample_sprites()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"using device: {device}")
    
    # Data preprocessing
    transform = transforms.Compose([
        transforms.Resize((16, 16)),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
    ])
    
    # dataset and dataloader
    dataset = PixelArtDataset('sprites', transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    print(f"loaded {len(dataset)} sprites")
    
    # initialize GAN
    gan = PixelArtGAN(latent_dim=64, lr=0.0002, device=device)
    
    # train GAN
    num_epochs = 1000
    print("starting training...")
    for epoch in range(num_epochs):
        d_losses = []
        g_losses = []
        
        for i, real_images in enumerate(dataloader):
            real_images = real_images.to(device)
            
            d_loss, g_loss = gan.train_step(real_images)
            d_losses.append(d_loss)
            g_losses.append(g_loss)
        
        avg_d_loss = np.mean(d_losses)
        avg_g_loss = np.mean(g_losses)
        
        print(f'Epoch [{epoch+1}/{num_epochs}] D_loss: {avg_d_loss:.4f} G_loss: {avg_g_loss:.4f}')

        n = 20 
        # Save samples every n epochs
        if (epoch + 1) % n == 0:
            gan.save_sample_grid(epoch + 1, save_dir=save_directory)
    
    print("Training completed!")
    print(f"Generated samples saved in {save_directory} directory")
    
    # generate final samples
    gan.save_sample_grid(epoch, save_dir=save_directory)

if __name__ == "__main__":
    main()