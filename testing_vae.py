import torch
from vae_model import VariationalAutoencoder as VAE
from utils.data_utils import get_MDP, get_latent_variable

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
training_csv_path = 'scenarios/' + scenario_name + '/csv_grids/training.csv'
training_data_path = 'scenarios/' + scenario_name + '/data/training.json'
fingerprint_data_path = 'scenarios/' + scenario_name + '/data/fingerprints.json'
testing_points_data_path = 'scenarios/' + scenario_name + '/data/testing_points.json'

#Load the saved model and input dimensions
model_load_path = 'trained_models/vae_model_' + scenario_name + '.pth'
checkpoint = torch.load(model_load_path)
feature_length = checkpoint['feature_length']
latent_dim = checkpoint['latent_dim']

vae = VAE(feature_length=feature_length, latent_dim=latent_dim)    
vae.load_state_dict(checkpoint['model_state_dict'])
vae.eval()

#Get MDP and Latent Variable
target_x = 10
target_y = 19.5

mdp, latent, (x,y) = get_latent_variable(training_csv_path, training_data_path, target_x, target_y, vae, device)
print(x, y)
print(mdp)
print(latent)