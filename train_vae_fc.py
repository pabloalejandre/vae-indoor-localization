import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from utils.data_utils import MyDataset, get_MDP_from_data, pad_and_sort_MDPs, find_shortest_sequence_length, truncate_batch
from vae_models.vae_model_fc import VariationalAutoencoder as VAE, vae_loss

#Training parameters
latent_dim = 2
number_of_transmitters = 1
split_ratio = 0.9

num_epochs = 128
batch_size = 32
learning_rate = 1e-3
use_gpu = True

#Get training data from simulator output, extract the MDPs, and preprocess them for VAE input
receiver_MDPs = get_MDP_from_data(path='data/BoxRoom_1Transmitter_x0_y9p5/training/training_Order2_DiffOff.json', receiver_id=1, number_of_transmitters=1, timestamps=1500)
receiver_MDPs_padded = pad_and_sort_MDPs(receiver_MDPs)
shortest_seq_length = find_shortest_sequence_length(receiver_MDPs_padded)
shortest_seq_length = 14 #Shortest sequence in fingerprints/testing points
print("Truncated MDPs to: ", shortest_seq_length)
receiver_MDPs_padded = truncate_batch(receiver_MDPs_padded, shortest_seq_length)
receiver_MDPs_padded = np.squeeze(receiver_MDPs_padded)

#Define training and validation sets for VAE
dataset = MyDataset(receiver_MDPs_padded)
split_index = int(len(dataset)*split_ratio)
training_dataset, val_dataset = random_split(dataset, [split_index, len(dataset) - split_index])

train_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


#Initialize VAE model and set optimizer
device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
vae = VAE(feature_length=shortest_seq_length, latent_dim=latent_dim)
vae = vae.to(device)
num_params = sum(p.numel() for p in vae.parameters() if p.requires_grad)
print('Number of parameters: %d' % num_params)
optimizer = torch.optim.Adam(params=vae.parameters(), lr=learning_rate, weight_decay=1e-5) #optimizer
vae.train() #Set model to training mode

#Training loop
train_loss_avg = []
bce_avg = []
kld_avg = []
val_loss_avg = []

print('Training ...')
for epoch in range(num_epochs):
    train_loss_avg.append(0)
    bce_avg.append(0)
    kld_avg.append(0)

    num_batches = 0
    
    for batch in train_dataloader:
        #Truncate batch
        batch = batch.to(device)

        # vae reconstruction
        batch_recon, latent_mu, latent_logvar = vae(batch)
        
        # reconstruction error
        loss, bce, kld = vae_loss(batch_recon, batch, latent_mu, latent_logvar)
        
        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        
        # one step of the optmizer (using the gradients from backpropagation)
        optimizer.step()
        
        train_loss_avg[-1] += loss.item()
        
        bce_avg[-1] += bce.item()
        kld_avg[-1] += kld.item()
        
        num_batches += 1
        
    train_loss_avg[-1] /= num_batches
    bce_avg[-1] /= num_batches
    kld_avg[-1] /= num_batches
    print('Epoch [%d / %d] average reconstruction error: %f' % (epoch+1, num_epochs, train_loss_avg[-1]))


    #Validation step
    vae.eval()
    val_loss_avg.append(0)
    num_val_batches = 0

    with torch.no_grad():
        for val_batch in val_dataloader:
            val_batch = val_batch.to(device)

            #vae reconstrunction
            val_batch_recon, val_latent_mu, val_latent_logvar = vae(val_batch)

            #reconstruction error
            val_loss, _, _ = vae_loss(val_batch_recon, val_batch, val_latent_mu, val_latent_logvar)
           
            val_loss_avg[-1] += val_loss.item()
            num_val_batches +=1

    val_loss_avg[-1] /= num_val_batches
    print('                average reconstruction error (validation): %f' % (val_loss_avg[-1]))
    print('--------------------------------------------------------------------')

    #Set model back to training mode
    vae.train()

# Saving the model after training
model_save_path = "trained_models/vae_model_fc.pth"
torch.save({
    'model_state_dict': vae.state_dict(),
    'latent_dim' : latent_dim,
    'feature_length' : shortest_seq_length
}, model_save_path)
print(f"Model saved to {model_save_path}")

# Setup for subplots
fig, axs = plt.subplots(3, 1, figsize=(10, 15)) # 3 rows, 1 column
# Total Loss Plot
axs[0].plot(train_loss_avg, label='Training Loss')
axs[0].plot(val_loss_avg, label='Validation Loss')
axs[0].set_xlabel('Epochs')
axs[0].set_ylabel('Total Loss')
axs[0].grid(True)

# KLD Plot
axs[1].plot(bce_avg, label='BCE Loss')
axs[1].set_xlabel('Epochs')
axs[1].set_ylabel('BCE')
axs[1].grid(True)

# BCE Plot
axs[2].plot(kld_avg, label='KLD Loss')
axs[2].set_xlabel('Epochs')
axs[2].set_ylabel('KLD')
axs[2].grid(True)

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()
