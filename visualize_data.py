import torch
from torch.utils.data import DataLoader
from vae_model import VariationalAutoencoder as VAE
from vae_model_conv import VariationalAutoencoder as VAE_Conv
from utils.data_utils import MyDataset, get_MDP_from_data, preprocess_mdps, evaluate_model, evaluate_model_modified
from utils.visualize_utils import visualize_mdps, visualize_reconstructions, visualize_2D_latent_space

#Set evaluation parameters
batch_size = 1
use_gpu = True
device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
#MDP parameters
scenario_name = 'BoxLectureRoomCorner'
number_of_transmitters = 1
receiver_id = 1
timestamps = 12

#Paths to signal data and coordinates
fingerprint_data_path = 'scenarios/' + scenario_name + '/data/fingerprints.json'
testing_data_path = 'scenarios/' + scenario_name + '/data/testing.json'
testing_obstacles_data_path = 'scenarios/' + scenario_name + '/data/testing_obstacles1.json'

# Load the saved model and input dimensions
#model_load_path = 'trained_models/vae_model_' + scenario_name + '.pth'
model_load_path = 'trained_models/vae_model_conv_goodperf2.pth'
checkpoint = torch.load(model_load_path)
feature_length = checkpoint['feature_length']
latent_dim = checkpoint['latent_dim']

vae = VAE_Conv(feature_length=feature_length, latent_dim=latent_dim)
#vae = VAE(feature_length=feature_length, latent_dim=latent_dim)
vae.load_state_dict(checkpoint['model_state_dict'])
vae.eval()

#Get MDPs of reference and testing points from simulator output
fingerprints_MDP = get_MDP_from_data(fingerprint_data_path, receiver_id, number_of_transmitters, timestamps) 
fingerprints_MDP_processed = preprocess_mdps(fingerprints_MDP, feature_length)
testing_MDP = get_MDP_from_data(testing_data_path, receiver_id, number_of_transmitters, timestamps)
testing_MDP_processed = preprocess_mdps(testing_MDP, feature_length)
testing_obstacles_MDP = get_MDP_from_data(testing_obstacles_data_path, receiver_id, number_of_transmitters, timestamps)
testing_obstacles_MDP_processed = preprocess_mdps(testing_obstacles_MDP, feature_length)

#Prepare the data for VAE input (dataset and dataloader)
fingerprints_dataset = MyDataset(fingerprints_MDP_processed)
fingerprints_dataloader = DataLoader(fingerprints_dataset, batch_size=batch_size, shuffle=False)
testing_dataset = MyDataset(testing_MDP_processed)
testing_dataloader = DataLoader(testing_dataset, batch_size=batch_size, shuffle=False)
testing_obstacles_dataset = MyDataset(testing_obstacles_MDP_processed)
testing_obstacles_dataloader = DataLoader(testing_obstacles_dataset, batch_size=batch_size, shuffle=False)

#Get VAE Outputs: Reconstructions and Latent variables
#Fingerprints
fingerprints_latent, fingerprints_recon = evaluate_model(vae, device, fingerprints_dataloader)
fingerprints_modified_latent, fingerprints_modified_recon, fingerprints_modified = evaluate_model_modified(vae, device, fingerprints_dataloader)
#Testing points
testing_latent, testing_recon = evaluate_model(vae, device, testing_dataloader)
testing_modified_latent, testing_modified_recon, testing_modified = evaluate_model_modified(vae, device, testing_dataloader)
#Testing points with obstacles
testing_obstacles_latent, testing_obstacles_recon = evaluate_model(vae, device, testing_obstacles_dataloader)

visualize_mdps(fingerprints_MDP_processed, fingerprints_modified, title='Effect of Modification')
visualize_mdps(fingerprints_recon, fingerprints_modified_recon, title='Reconstruction Comparison')

#Visualize VAE outputs
visualize_reconstructions(fingerprints_MDP_processed, fingerprints_recon, title='Fingerprints')
visualize_reconstructions(testing_MDP_processed, testing_recon, title='Testing Points')

visualize_mdps(testing_MDP_processed, testing_obstacles_MDP_processed, title='Effect of Obstacles')
visualize_mdps(testing_recon, testing_obstacles_recon, title='Reconstruction Comparison')

visualize_2D_latent_space(fingerprints_latent, testing_latent, title='No obstacles')
visualize_2D_latent_space(fingerprints_latent, testing_obstacles_latent, title='With obstacles')
