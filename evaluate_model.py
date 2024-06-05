import time
from torch.utils.data import DataLoader
from vae_model import VariationalAutoencoder as VAE
from utils.data_utils import MyDataset, get_MDP_from_data, preprocess_mdps, evaluate_model, find_k_nearest_neighbors, estimate_positions_from_nearest_neighbors, calculate_position_errors
from utils.visualize_utils import read_point_coordinates, visualize_mdps, visualize_reconstructions, visualize_2D_latent_space, visualize_error_cdf

scenario_name = 'BoxLectureRoom'
#scenario_name = 'LShapedRoom'
#scenario_name = 'LivingRoom'

#Paths to signal data and point coordinates
fingerprint_coord_path = f'scenarios/{scenario_name}/csv/reference.csv'
testing_coord_path = f'scenarios/{scenario_name}/csv/testing_points.csv'
fingerprint_data_path = f'scenarios/{scenario_name}/data/reference.json'
testing_data_path = f'scenarios/{scenario_name}/data/testing.json'

#Get ground truth coordinates of each reference and testing point
reference_coords = read_point_coordinates(fingerprint_coord_path)
testing_coords = read_point_coordinates(testing_coord_path)

#Load the trained model
model_load_path = f'trained_models/{scenario_name}.ckpt'
vae = VAE.load_from_checkpoint(model_load_path)
feature_length = vae.feature_length
vae.eval()

#Offline phase: Get MDPs of reference points and their VAE output (would be stored in database)
reference_MDP = get_MDP_from_data(fingerprint_data_path)
reference_MDP_processed = preprocess_mdps(reference_MDP, feature_length)
reference_dataset = MyDataset(reference_MDP_processed)
reference_dataloader = DataLoader(reference_dataset, batch_size=1, shuffle=False)
reference_latent, reference_recon = evaluate_model(vae, feature_length, reference_dataloader)

#Online Phase: Get VAE outputs of testing points and localize them using fingerprint map
testing_MDP = get_MDP_from_data(testing_data_path)
testing_MDP_processed = preprocess_mdps(testing_MDP, feature_length)
testing_dataset = MyDataset(testing_MDP_processed)
testing_dataloader = DataLoader(testing_dataset, batch_size=1, shuffle=False)
testing_latent, testing_recon = evaluate_model(vae, feature_length, testing_dataloader)

#Localize testing points
#Get k-nearest neighbors in the latent space
nearest_neighbors_indices, nearest_neighbors_distances = find_k_nearest_neighbors(testing_latent, reference_latent, k=2)
estimated_positions = estimate_positions_from_nearest_neighbors(nearest_neighbors_indices, reference_coords)
errors = calculate_position_errors(estimated_positions, testing_coords)
print(nearest_neighbors_indices)
print(errors)
print(errors.mean())

#Choose whether to plot results and which ones to plot
plot_results = 1
if plot_results:
    visualize_reconstructions(testing_MDP_processed, testing_recon, scenario=scenario_name, title='Testing Points', obstacles='(No obstacles)')
    visualize_2D_latent_space(reference_latent, testing_latent, scenario=scenario_name, title='(No obstacles)')
    visualize_error_cdf(errors, k=1, title_addition='(No obstacles)')
    

#Include obstacles
if scenario_name == 'BoxLectureRoom' or scenario_name == 'LShapedRoom':
    testing_obstacles_data_path = f'scenarios/{scenario_name}/data/testing_obstacles.json'
    testing_obstacles_MDP = get_MDP_from_data(testing_obstacles_data_path)
    testing_obstacles_MDP_processed = preprocess_mdps(testing_obstacles_MDP, feature_length)
    testing_obstacles_dataset = MyDataset(testing_obstacles_MDP_processed)
    testing_obstacles_dataloader = DataLoader(testing_obstacles_dataset, batch_size=1, shuffle=False)
    testing_obstacles_latent, testing_obstacles_recon = evaluate_model(vae, feature_length, testing_obstacles_dataloader)

    nearest_neighbors_indices_obstacles, nearest_neighbors_distances_obst = find_k_nearest_neighbors(testing_obstacles_latent, reference_latent, k=5)
    estimated_positions_obstacles = estimate_positions_from_nearest_neighbors(nearest_neighbors_indices_obstacles, reference_coords)
    errors_obstacles = calculate_position_errors(estimated_positions_obstacles, testing_coords)
    print(nearest_neighbors_indices_obstacles)
    print(errors_obstacles)
    print(errors_obstacles.mean())

    visualize_mdps(testing_MDP_processed, testing_obstacles_MDP_processed, label1='Testing Points', label2='Testing Points (with obst.)', scenario=scenario_name, title='Effect of Obstacles on MDPs')
    visualize_mdps(testing_recon, testing_obstacles_recon, label1='Reconstructed Testing Points', label2='Reconstructed Testing Points (with obst.)', scenario=scenario_name, title='Testing Points Reconstruction Comparison')
    visualize_2D_latent_space(reference_latent, testing_obstacles_latent, title='With obstacles')
    visualize_error_cdf(errors_obstacles, k=1, title_addition='(With obstacles)')