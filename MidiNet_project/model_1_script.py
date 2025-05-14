#model_1 (with conditioner)

import os
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import random_split, Dataset, DataLoader
import random
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.multiprocessing as mp

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
        #data = data.to(device)                                     seems better not do it for stability
    
        return data

#In model 1 the generator must generate only a bar (1,87,200)
class Generator(nn.Module):
    def __init__(self, noise_dim=100, num_notes=87, length_bar=200):
        super(Generator, self).__init__()
        
        self.noise_dim = noise_dim
        self.num_notes = num_notes
        self.length_bar = length_bar

        self.fc1 = nn.Linear(noise_dim, 512)
        self.fc2 = nn.Linear(512, 40*5*12)
        
        self.deconv0 = nn.ConvTranspose2d(80, 30, kernel_size=(2, 3), stride=(2, 2))  # 40 + 40 = 80
        self.deconv1 = nn.ConvTranspose2d(60, 20, kernel_size=(3, 2), stride=(2, 2))  # 30 + 30 = 60
        self.deconv2 = nn.ConvTranspose2d(40, 10, kernel_size=(3, 2), stride=(2, 2))  # 20 + 20 = 40
        self.deconv3 = nn.ConvTranspose2d(20, 1, kernel_size=(3, 2), stride=(2, 2))   # 10 + 10 = 20

    def forward(self, z, cond):
        x = torch.relu(self.fc1(z))
        x = torch.relu(self.fc2(x))
        x = x.view(-1, 40, 5, 12)  # Initial shape: (batch, 40, 5, 12)
        
        # Concatenate conditioner features and process
        x = torch.cat([x, cond[3]], dim=1)       # 40 + 40 = 80 channels
        x = torch.relu(self.deconv0(x))          # Out: (30, 10, 25)
        
        x = torch.cat([x, cond[2]], dim=1)       # 30 + 30 = 60 channels
        x = torch.relu(self.deconv1(x))          # Out: (20, 21, 50)
        
        x = torch.cat([x, cond[1]], dim=1)       # 20 + 20 = 40 channels
        x = torch.relu(self.deconv2(x))          # Out: (10, 43, 100)
        
        x = torch.cat([x, cond[0]], dim=1)       # 10 + 10 = 20 channels
        x = self.deconv3(x)                      # Out: (1, 87, 200)
        
        x = x.squeeze(1)                           # Remove channel dim: [batch, 87, 200]
        x = F.gumbel_softmax(x, dim=1, hard=True)  # One-hot per column
        x = x.unsqueeze(1)                         # Restore channel dim: [batch, 1, 87, 200]
        
        
        return x
        
#conditioner
class Conditioner(nn.Module):
    def __init__(self, num_notes = 87, length_bar = 200):
        super(Conditioner, self).__init__()
        
        #variables
        self.num_notes = num_notes
        self.length_bar = length_bar
        
        #structure
        self.conv0 = nn.Conv2d(30, 40, kernel_size=(3, 2), stride=(2, 2), padding=(1, 0))
        self.conv1 = nn.Conv2d(20, 30, kernel_size=(2, 3), stride=(2, 2), padding=(0, 1))
        self.conv2 = nn.Conv2d(10, 20, kernel_size=(2, 3), stride=(2, 2), padding=(0, 1))
        self.conv3 = nn.Conv2d(1, 10,  kernel_size=(2, 3), stride=(2, 2), padding=(0, 1))
       
    def forward(self, x):   
        x = x.view(-1, 1, 87, 200)        # (batch_size, 1, 87, 200)           
        x0 = torch.relu(self.conv3(x))    # out (1, 10, 43, 100)
        x1 = torch.relu(self.conv2(x0))   # out (1, 20, 21, 50)
        x2 = torch.relu(self.conv1(x1))   # out (1, 30, 10, 25)
        x3 = torch.relu(self.conv0(x2))   # out (1, 40,  5, 12) 
        
        return [x0,x1,x2,x3]  

#discriminator
class Discriminator(nn.Module):
    def __init__(self, num_notes=87, length_bar=200):
        super(Discriminator, self).__init__()
        
        self.num_notes = num_notes
        self.length_bar = length_bar
        
        # Modified architecture for (1,87,200) inputs
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(5, 5), stride = 2, padding = 2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(5, 5), stride = 2, padding = 2) 
        self.fc1 = nn.Linear(128 * 22 * 50, 1) 
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))  # Out: (64, 44, 100)
        x = torch.relu(self.conv2(x))  # Out: (128, 22, 50)
        x = x.view(x.size(0), -1)      # Flatten to 128*22*50
        x = torch.sigmoid(self.fc1(x))
        return x
        
def train_gan(generator, conditioner, discriminator, dataloader, num_epochs, noise_dim=100, label_smoothing=0.9, device = device):
    criterion = nn.BCELoss()
    optimizer_g = optim.Adam(list(generator.parameters()) + list(conditioner.parameters()), lr=0.02, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    g_losses = []
    d_losses = []
    # I do not think I need also the conditioner loss: they work together

    for epoch in range(num_epochs):
        g_loss_epoch = 0.0
        d_loss_epoch = 0.0

        for i, data in enumerate(dataloader):
            
            batch_size = data.size(0)
            zeros = torch.zeros(batch_size,1,87,200, dtype = torch.int)
            data = torch.cat((zeros,data), dim = -1)     #(batch_size, 1, 87, 1800) 9 bars, the first composed of zeros
            first_idxs = random.choices([i for i in range(0, 8)], k = batch_size)  #[0,7]  (batch_size)
            second_idxs = [i + 1 for i in first_idxs]
        
       
            # Extract first bars using indexing, it is either real data or all zeros. Input of the conditioner
            first_bars = torch.stack([data[b, :, :, 200 * idx : 200 * (idx + 1)] for b, idx in enumerate(first_idxs)]).to(device)

	    # Extract second bars (next in sequence). Real data, used as real data for input of discriminator
            second_bars = torch.stack([data[b, :, :, 200 * idx : 200 * (idx + 1)] for b, idx in enumerate(second_idxs)]).to(device)
            
            real_labels = torch.full((batch_size, 1), label_smoothing, device = device)
            fake_labels = torch.zeros((batch_size, 1), device = device )
            
            #Forward discriminator
            optimizer_d.zero_grad(set_to_none=True)
            real_output = discriminator(second_bars)
            loss_real = criterion(real_output, real_labels)
            
            #forward conditioner
            conditions = conditioner(first_bars)
            
            #forward generator 
            noise = torch.randn(batch_size, noise_dim, device = device)
            fake_data = generator(noise, conditions)
            
            #backward discriminator
            fake_output = discriminator(fake_data.detach())
            loss_fake = criterion(fake_output, fake_labels)
            loss_d = loss_real + loss_fake
            loss_d.backward()
            optimizer_d.step()
            
            #backward generator and conditioner
            optimizer_g.zero_grad(set_to_none=True)
            fake_output = discriminator(fake_data)     
            loss_g = criterion(fake_output, real_labels)
            loss_g.backward()
            optimizer_g.step()

            #forward conditioner (implied), TODO:maybe here I can optimize and avoid to repeat variables
            conditions = conditioner(first_bars)      #computed two times, I have to keep it because I use different noise vectors
            
            #backward generator and conditioner,            
            optimizer_g.zero_grad(set_to_none=True)
            noise = torch.randn(batch_size, noise_dim, device = device) 
            fake_data = generator(noise, conditions)
            fake_output = discriminator(fake_data.detach())
            loss_g_2 = criterion(fake_output, real_labels)
            loss_g_2.backward()
            optimizer_g.step()

            g_loss_epoch += (loss_g.item() + loss_g_2.item()) / 2
            d_loss_epoch += loss_d.item()

            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(dataloader)}], Loss D: {loss_d.item()}, Loss G: {loss_g.item()}")

        g_losses.append(g_loss_epoch / len(dataloader))
        d_losses.append(d_loss_epoch / len(dataloader))

    print("Training completed.")
    return g_losses, d_losses

def save_model(generator, conditioner, discriminator, epoch, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    generator_path = os.path.join(output_dir, f"m1_generator_epoch_{epoch+1}.pth")
    conditioner_path = os.path.join(output_dir, f"m1_conditioner_epoch_{epoch+1}.pth")
    discriminator_path = os.path.join(output_dir, f"m1_discriminator_epoch_{epoch+1}.pth")
    torch.save(generator.state_dict(), generator_path)
    torch.save(conditioner.state_dict(), conditioner_path)
    torch.save(discriminator.state_dict(), discriminator_path)
    print(f"Saved models to Google Drive: {generator_path} and {conditioner_path} and {discriminator_path}")
       
if __name__ == '__main__':
    # Ensure that the multiprocessing start method is set correctly.
    mp.set_start_method('spawn', force=True)
    
    segments_path = "/content/drive/MyDrive/NNDL_project_MidiNet/segments/"
    dataset = MidiDataset(segments_path)
    
    total_size = len(dataset)
    train_size = int(0.2*total_size)
    
    train_dataset = torch.utils.data.Subset(dataset, range(0, train_size, 1))
    batch_size = 200
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers = True)    #num_workers decrease to 1 for now (it was 2)
   
    noise_dim = 100
    num_notes = 87
    num_epochs = 150
    
    generator = Generator(noise_dim=noise_dim, num_notes=num_notes).to(device)
    conditioner = Conditioner().to(device)
    discriminator = Discriminator(num_notes=num_notes).to(device)

    g_losses, d_losses = train_gan(generator, conditioner, discriminator, train_loader, num_epochs=num_epochs, noise_dim=noise_dim)
    output_dir = "/content/drive/MyDrive/NNDL_project_MidiNet/model_1_state_dicts"
    save_model(generator, conditioner, discriminator, epoch=num_epochs-1, output_dir=output_dir)

    plt.figure(figsize=(10, 5))
    plt.plot(g_losses, label="Generator Loss")
    plt.plot(d_losses, label="Discriminator Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Generator and Discriminator Losses")
    plt.savefig("/content/drive/MyDrive/NNDL_project_MidiNet/gan_training_loss_model_1_150e200b.pdf")
    
