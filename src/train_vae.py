import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from vae_model_fc import VariationalAutoencoder as VAE_FC, vae_loss
from vae_model_conv_1D import VariationalAutoencoder as VAE_Conv1D
from vae_model_conv_2D import VariationalAutoencoder as VAE_Conv2D
from data_utils import MyDataset, load_data, getReceiverCSI, getMDPsfromReceiverCSI, pad_MDP

#2-d latent space, parameter count in same order of magnitude
# as in the original VAE paper (VAE paper has about 3x as many)
num_epochs = 4 #100
batch_size = 1 #128
learning_rate = 1e-3
use_gpu = True

# # 10-d latent space, for comparison with non-variational auto-encoder
# num_epochs = 100
# batch_size = 128
# learning_rate = 1e-3
# use_gpu = True

#Get training data from simulator output
dicts = load_data("data/BoxLectureRoomOffline.json")
rcv_CSI = getReceiverCSI(dicts, receiver_number=3, transmitterRange=3)
rcv_MDP = getMDPsfromReceiverCSI(rcv_CSI, timestamps=4)
rcv_MDP = pad_MDP(rcv_MDP)
rcv_MDP = np.array(rcv_MDP, dtype=np.float32)
print("Dimensions of MDPs: " + str(rcv_MDP.shape))

#Initialize VAE model and set optimizer
model_choice = input("Choose model. 0 for FC, 1 for Conv1D, 2 for Conv2D: ")

#Fully connected VAE
if(model_choice == '0'):
    rcv_MDP = rcv_MDP.reshape(rcv_MDP.shape[0], -1) #vae_model_fc
    print("Dimensions of flattened MDPs for fully connected VAE: " + str(rcv_MDP.shape))
    vae = VAE_FC(input_shape = rcv_MDP.shape[1], latent_dim= 20) #vae_model_fc
#Convolutional VAE (1D)
elif(model_choice == '1'):
    vae = VAE_Conv1D(num_feature_vectors=rcv_MDP.shape[1], feature_vector_length=rcv_MDP.shape[2], latent_dim=10) #vae_model_conv_1D
#Convolutional VAE (2D)
elif(model_choice == '2'):
    vae = VAE_Conv2D() #vae_model_conv_2D
else:
    print('Invalid choice')
    quit()

#Define training set for VAE
training_set = MyDataset(rcv_MDP)
train_dataloader = DataLoader(training_set, batch_size=batch_size, shuffle=True)

device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
vae = vae.to(device)

num_params = sum(p.numel() for p in vae.parameters() if p.requires_grad)
print('Number of parameters: %d' % num_params)

optimizer = torch.optim.Adam(params=vae.parameters(), lr=learning_rate, weight_decay=1e-5)


#Set to training mode and perform training loop
vae.train()

train_loss_avg = []

print('Training ...')
for epoch in range(num_epochs):
    train_loss_avg.append(0)
    num_batches = 0
    
    for image_batch in train_dataloader:
        
        image_batch = image_batch.to(device)

        # vae reconstruction
        image_batch_recon, latent_mu, latent_logvar = vae(image_batch)
        
        # reconstruction error
        loss = vae_loss(image_batch_recon, image_batch, latent_mu, latent_logvar)
        
        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        
        # one step of the optmizer (using the gradients from backpropagation)
        optimizer.step()
        
        train_loss_avg[-1] += loss.item()
        num_batches += 1
        
    train_loss_avg[-1] /= num_batches
    print('Epoch [%d / %d] average reconstruction error: %f' % (epoch+1, num_epochs, train_loss_avg[-1]))


#Plot the loss after every epoch
fig = plt.figure()
plt.plot(train_loss_avg)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig('vae_loss')
plt.show()
