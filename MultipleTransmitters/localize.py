import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from scipy.spatial.distance import cdist
from vae_models.vae_model import VariationalAutoencoder as VAE
from utils.data_utils import MyDataset, get_MDP_from_data, get_latent_representation

#Set evaluation parameters
batch_size = 1
use_gpu = True
device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")

#Paths to get fingerprint and testing point data
fingerprint_data_path = 'data/BoxRoom_4Transmitters/fingerprints.json'
testing_points__data_path = 'data/BoxRoom_4Transmitters/testing1.json'

#MDP parameters
number_of_transmitters = 4
receiver_id = 4
timestamps = 12
padlength = None
trunclength = 17

# Load the saved model and input dimensions
model_name = "vae_model_conv.pth"
model_load_path = os.path.join("trained_models/", model_name)
checkpoint = torch.load(model_load_path)
number_of_transmitters = checkpoint['number_of_transmitters']
latent_dim = checkpoint['latent_dim']

vae = VAE(number_of_transmitters=number_of_transmitters, latent_dim=latent_dim)    
vae.load_state_dict(checkpoint['model_state_dict'])
vae.eval()

#Get MDPs of reference and testing points from simulator output
fingerprints_MDP = get_MDP_from_data(fingerprint_data_path, receiver_id, number_of_transmitters, timestamps, padlength, trunclength) 
testing_points_MDP = get_MDP_from_data(testing_points__data_path, receiver_id, number_of_transmitters, timestamps, padlength, trunclength)


#Prepare the data for VAE input (dataset and dataloader)
fingerprints_dataset = MyDataset(fingerprints_MDP)
testing_points_dataset = MyDataset(testing_points_MDP)
fingerprints_dataloader = DataLoader(fingerprints_dataset, batch_size=batch_size, shuffle=False)
testing_points_dataloader = DataLoader(testing_points_dataset, batch_size=batch_size, shuffle=False)


#Get latent variables
fingerprints_latent = get_latent_representation(vae, device, fingerprints_dataloader)
testing_points_latent = get_latent_representation(vae, device, testing_points_dataloader)

# Convert PyTorch tensors to NumPy arrays for distance computation
fingerprints_latent_np = fingerprints_latent.numpy()
testing_points_latent_np = testing_points_latent.numpy()

# Calculate distances between each localizing point and each reference point according to different metrics
distances_euclidean = cdist(testing_points_latent_np, fingerprints_latent_np, metric='euclidean')
distances_cosine = cdist(testing_points_latent_np, fingerprints_latent_np, metric='cosine')

# Find the index of the closest reference point for each localizing point
pairings_euclidean = np.argmin(distances_euclidean, axis=1)
print("---Computed pairings with euclidean norm---")
print(pairings_euclidean)
print("------------------------")

pairings_cosine =  np.argmin(distances_cosine, axis=1)
print("---Computed pairings with euclidean norm---")
print(pairings_cosine)
print("------------------------")


#Real pairings:
#Testing1
print("---Real pairings:---")
print([4, 9, 9, 11, 1, 6, 1, 9, 6, 3, 7, 9])

#Testing2
#print("---Real pairings:---")
#print([2, 10, 0, 9, 3, 7, 9, 6, 4, 6, 6, 2])