import torch
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from vae_models.vae_model_fc import VariationalAutoencoder as VAE
from utils.data_utils import MyDataset, get_MDP_from_data, pad_and_sort_MDPs, truncate_batch

#Set evaluation parameters
batch_size = 1
use_gpu = True
device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")

#Paths to get fingerprint and testing point data
fingerprint_data_path = 'data/BoxRoom_1Transmitter_x0_y9p5/testing/testingPoints_Order2_DiffOff.json'
testing_points__data_path = 'data/BoxRoom_1Transmitter_x0_y9p5/testing/testingPoints_Order2_DiffOff.json'

#MDP parameters
number_of_transmitters = 1
receiver_id = 1
timestamps = 12

# Load the saved model and input dimensions
model_load_path = 'trained_models/vae_model_fc_x0_y9p5.pth'
checkpoint = torch.load(model_load_path)
feature_length = checkpoint['feature_length']
latent_dim = checkpoint['latent_dim']

vae = VAE(feature_length=feature_length, latent_dim=latent_dim)    
vae.load_state_dict(checkpoint['model_state_dict'])
vae.eval()

#Get MDPs of reference and testing points from simulator output
fingerprints_MDP = get_MDP_from_data(fingerprint_data_path, receiver_id, number_of_transmitters, timestamps) 
testing_points_MDP = get_MDP_from_data(testing_points__data_path, receiver_id, number_of_transmitters, timestamps)

fingerprints_MDP = pad_and_sort_MDPs(fingerprints_MDP)
fingerprints_MDP = truncate_batch(fingerprints_MDP, feature_length)
fingerprints_MDP = np.squeeze(fingerprints_MDP)

testing_points_MDP = pad_and_sort_MDPs(testing_points_MDP)
testing_points_MDP = truncate_batch(testing_points_MDP, feature_length)
testing_points_MDP = np.squeeze(testing_points_MDP)

#Prepare the data for VAE input (dataset and dataloader)
fingerprints_dataset = MyDataset(fingerprints_MDP)
testing_points_dataset = MyDataset(testing_points_MDP)
fingerprints_dataloader = DataLoader(fingerprints_dataset, batch_size=batch_size, shuffle=False)
testing_points_dataloader = DataLoader(testing_points_dataset, batch_size=batch_size, shuffle=False)

with torch.no_grad():
        for batch in fingerprints_dataloader:
                batch.to(device)
                batch_recon, _, _ = vae(batch)
                input = batch.numpy()
                reconstruction = batch_recon.numpy()
                print('---------------------------------------')
                print(input)
                print(reconstruction)
