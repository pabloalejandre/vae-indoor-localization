from torch.utils.data import DataLoader
from vae_model import VariationalAutoencoder as VAE
from utils.data_utils import MyDataset, get_MDP_from_data, preprocess_mdps, evaluate_model
from utils.visualize_utils import visualize_mdps, visualize_reconstructions, visualize_2D_latent_space

scenario_name = 'BoxLectureRoom'
#scenario_name = 'LShapedRoom'
#scenario_name = 'LivingRoom'

#Paths to signal data and coordinates
fingerprint_data_path = f'scenarios/{scenario_name}/data/fingerprints.json'
testing_data_path = f'scenarios/{scenario_name}/data/testing.json'
#testing_obstacles_data_path = f'scenarios/{scenario_name}/data/testing_obstacles2.json'

#Load the trained model
model_load_path = f'trained_models/{scenario_name}.ckpt'
vae = VAE.load_from_checkpoint(model_load_path)
feature_length = vae.feature_length
vae.eval()

#Get MDPs of reference and testing points from simulator output
fingerprints_MDP = get_MDP_from_data(fingerprint_data_path)
fingerprints_MDP_processed = preprocess_mdps(fingerprints_MDP, feature_length)
testing_MDP = get_MDP_from_data(testing_data_path)
testing_MDP_processed = preprocess_mdps(testing_MDP, feature_length)
#testing_obstacles_MDP = get_MDP_from_data(testing_obstacles_data_path)
#testing_obstacles_MDP_processed = preprocess_mdps(testing_obstacles_MDP, feature_length)

#Prepare the data for VAE input (dataset and dataloader)
fingerprints_dataset = MyDataset(fingerprints_MDP_processed)
fingerprints_dataloader = DataLoader(fingerprints_dataset, batch_size=1, shuffle=False)

testing_dataset = MyDataset(testing_MDP_processed)
testing_dataloader = DataLoader(testing_dataset, batch_size=1, shuffle=False)

#testing_obstacles_dataset = MyDataset(testing_obstacles_MDP_processed)
#testing_obstacles_dataloader = DataLoader(testing_obstacles_dataset, batch_size=1, shuffle=False)

#Get VAE Outputs: Reconstructions and Latent variables
#Fingerprints
fingerprints_latent, fingerprints_recon = evaluate_model(vae, feature_length, fingerprints_dataloader)
#Testing points
testing_latent, testing_recon = evaluate_model(vae, feature_length, testing_dataloader)
#Testing points with obstacles
#testing_obstacles_latent, testing_obstacles_recon = evaluate_model(vae, feature_length, testing_obstacles_dataloader)

#Visualize VAE outputs
visualize_reconstructions(fingerprints_MDP_processed, fingerprints_recon, title='Reference Points')
visualize_reconstructions(testing_MDP_processed, testing_recon, title='Testing Points')

#visualize_mdps(testing_MDP_processed, testing_obstacles_MDP_processed, label1='Testing Points', label2='Testing Points (with obst.)', title='Effect of Obstacles')
#visualize_mdps(testing_recon, testing_obstacles_recon, label1='Reconstructed Testing Points', label2='Reconstructed Testing Points (with obst.)', title='Reconstruction Comparison')

visualize_2D_latent_space(fingerprints_latent, testing_latent, title='No obstacles')
#visualize_2D_latent_space(fingerprints_latent, testing_obstacles_latent, title='With obstacles')
