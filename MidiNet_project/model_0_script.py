import os
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import random_split, Dataset, DataLoader
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.amp as amp
import matplotlib.pyplot as plt

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
 
# Dataset class
class MidiDataset(Dataset):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.file_names = sorted([f for f in os.listdir(folder_path) if f.endswith('.npz')])

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_path = os.path.join(self.folder_path, self.file_names[idx])
        data = np.load(file_path)['segment']
        data = torch.from_numpy(data).view((1, data.shape[0], data.shape[1])).float()
        #data = data.to(device)         #avoid it for stability
        return data

# Model classes (Generator and Discriminator)
class Generator(nn.Module):
    def __init__(self, noise_dim=100, num_notes=87, num_tempi=1600):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim
        self.num_notes = num_notes
        self.num_tempi = num_tempi
        self.fc1 = nn.Linear(self.noise_dim, 512)
        self.fc2 = nn.Linear(512, 40*5*12)
        self.deconv0 = nn.ConvTranspose2d(40, 30, kernel_size=(2, 5), stride=(2, 3)) 
        self.deconv1 = nn.ConvTranspose2d(30, 20, kernel_size=(3, 5), stride=(2, 2)) 
        self.deconv2 = nn.ConvTranspose2d(20, 10, kernel_size=(3, 9), stride=(2, 5)) 
        self.deconv3 = nn.ConvTranspose2d(10, 1, kernel_size=(3, 8), stride=(2, 4))  

    #@torch.autocast(device_type="cuda")
    def forward(self, z):
        x = torch.relu(self.fc1(z))
        x = torch.relu(self.fc2(x))
        x = x.view(-1, 40, 5, 12)
        x = torch.relu(self.deconv0(x)) #out (30, 10, 38) 
        x = torch.relu(self.deconv1(x)) #out (20, 21, 79)
        x = torch.relu(self.deconv2(x)) #out (10, 43, 399)
        x = self.deconv3(x)             #out (1, 87, 1600)
        
        x = x.squeeze(1)                           # Remove channel dim: [batch, 87, 1600]
        x = F.gumbel_softmax(x, dim=1, hard=True)  # One-hot per column
        x = x.unsqueeze(1)                         # Restore channel dim: [batch, 1, 87, 1600]
        
        return x

class Discriminator(nn.Module):
    def __init__(self, num_notes=87, num_tempi=1600):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(num_notes, 1), stride=(1, 1))
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(1, 2), stride=(8, 8))
        self.fc1 = nn.Linear(128 * 1 * 200, 1024)
        self.fc2 = nn.Linear(1024, 1)

    #@torch.autocast(device_type="cuda")
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

def train_gan(generator, discriminator, dataloader, num_epochs, noise_dim=100, label_smoothing=0.9, device = device):
    criterion = nn.BCELoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=0.02, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    g_losses = []
    d_losses = []

    for epoch in range(num_epochs):
        g_loss_epoch = 0.0
        d_loss_epoch = 0.0

        for i, real_data in enumerate(dataloader):
            real_data = real_data.to(device)
            batch_size = real_data.size(0)
            real_labels = torch.full((batch_size, 1), label_smoothing, device=real_data.device)
            fake_labels = torch.zeros((batch_size, 1), device=real_data.device)

            optimizer_d.zero_grad(set_to_none=True)
            real_output = discriminator(real_data)
            loss_real = criterion(real_output, real_labels)
            noise = torch.randn(batch_size, noise_dim, device=real_data.device)
            fake_data = generator(noise)
            fake_output = discriminator(fake_data.detach())
            loss_fake = criterion(fake_output, fake_labels)
            loss_d = loss_real + loss_fake
            loss_d.backward()
            optimizer_d.step()

            optimizer_g.zero_grad(set_to_none=True)
            fake_output = discriminator(fake_data)
            loss_g = criterion(fake_output, real_labels)
            loss_g.backward()
            optimizer_g.step()

            optimizer_g.zero_grad(set_to_none=True)
            noise = torch.randn(batch_size, noise_dim, device=real_data.device)
            fake_data = generator(noise)
            fake_output = discriminator(fake_data.detach())
            loss_g_2 = criterion(fake_output, real_labels)
            loss_g_2.backward()
            optimizer_g.step()

            g_loss_epoch += (loss_g.item() + loss_g_2.item()) / 2
            d_loss_epoch += loss_d.item()

            if (i + 1) % 10 == 0:
                
                # Synchronize before printing to ensure correct order
                torch.cuda.synchronize()
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(dataloader)}], Loss D: {loss_d.item()}, Loss G: {loss_g.item()}")

        g_losses.append(g_loss_epoch / len(dataloader))
        d_losses.append(d_loss_epoch / len(dataloader))

    print("Training completed.")
    return g_losses, d_losses

def save_model(generator, discriminator, epoch, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    generator_path = os.path.join(output_dir, f"m0_generator_epoch_{epoch+1}.pth")
    discriminator_path = os.path.join(output_dir, f"m0_discriminator_epoch_{epoch+1}.pth")
    torch.save(generator.state_dict(), generator_path)
    torch.save(discriminator.state_dict(), discriminator_path)
    print(f"Saved models to Google Drive: {generator_path} and {discriminator_path}")

if __name__ == '__main__':
    # Ensure that the multiprocessing start method is set correctly.
    mp.set_start_method('spawn', force=True)
    
    segments_path = "/content/drive/MyDrive/NNDL_project_MidiNet/segments/"
    dataset = MidiDataset(segments_path)
    
    total_size = len(dataset)
    train_size = int(0.2 * total_size)
    
    train_dataset = torch.utils.data.Subset(dataset, range(0, train_size, 1))
    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers = True)
   
    noise_dim = 100
    num_notes = 87
    num_epochs = 100
    
    generator = Generator(noise_dim=noise_dim, num_notes=num_notes).to(device)
    discriminator = Discriminator(num_notes=num_notes).to(device)

    g_losses, d_losses = train_gan(generator, discriminator, train_loader, num_epochs=num_epochs, noise_dim=noise_dim)
    output_dir = "/content/drive/MyDrive/NNDL_project_MidiNet/model_0_state_dicts"
    save_model(generator, discriminator, epoch=num_epochs-1, output_dir=output_dir)

    plt.figure(figsize=(10, 5))
    plt.plot(g_losses, label="Generator Loss")
    plt.plot(d_losses, label="Discriminator Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Generator and Discriminator Losses")
    plt.savefig("/content/drive/MyDrive/NNDL_project_MidiNet/gan_training_loss_Model_0.pdf")

