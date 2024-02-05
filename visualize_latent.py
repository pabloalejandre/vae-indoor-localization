import torch
import numpy as np
from torch.utils.data import DataLoader
from vae_models.vae_model_fc import VariationalAutoencoder as VAE
from utils.data_utils import MyDataset, get_MDP_from_data, pad_and_sort_MDPs, find_shortest_sequence_length, truncate_batch, get_latent_representation
from utils.visualize_utils import visualize_2D_latent_representations

#Set evaluation parameters
batch_size = 1
use_gpu = True
device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")

#Paths to get fingerprint and testing point data
fingerprint_data_path = 'data/BoxRoom_1Transmitter_x0_y9p5/testing/testingPoints_Order2_DiffOff.json'
#testing_points_data_path = 'data/BoxRoom_1Transmitter/testing/testingPoints_Order2_DiffOff.json'

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
fingerprints_MDP = pad_and_sort_MDPs(fingerprints_MDP)
fingerprints_MDP = truncate_batch(fingerprints_MDP, feature_length)
print(find_shortest_sequence_length(fingerprints_MDP))
fingerprints_MDP = np.squeeze(fingerprints_MDP)
#testing_points_MDP = get_MDP_from_data(testing_points_data_path, receiver_id, number_of_transmitters, timestamps)
# testing_points_MDP = pad_and_sort_MDPs(testing_points_MDP)
# testing_points_MDP = truncate_batch(testing_points_MDP, feature_length)
# testing_points_MDP = np.squeeze(testing_points_MDP)


#Prepare the data for VAE input (dataset and dataloader)
fingerprints_dataset = MyDataset(fingerprints_MDP)
fingerprints_dataloader = DataLoader(fingerprints_dataset, batch_size=batch_size, shuffle=False)
#testing_points_dataset = MyDataset(testing_points_MDP)
#testing_points_dataloader = DataLoader(testing_points_dataset, batch_size=batch_size, shuffle=False)

#Get latent variables and convert PyTorch tensors to Numpy arrays
fingerprints_latent = get_latent_representation(vae, device, fingerprints_dataloader)
fingerprints_latent_np = fingerprints_latent.numpy()
#testing_points_latent = get_latent_representation(vae, device, testing_points_dataloader)
#testing_points_latent_np = testing_points_latent.numpy()

#Visualize latent representation
visualize_2D_latent_representations(latent_array=fingerprints_latent_np)