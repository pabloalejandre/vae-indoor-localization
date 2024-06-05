import numpy as np
import time
from utils.data_utils import get_MDP_from_data, calculate_position_errors
from utils.visualize_utils import read_point_coordinates, visualize_error_cdf

def similarity_score(D, F, epsilon):
    score = 0
    for dj in D:
        fl = min(F, key=lambda fl: abs(dj - fl))
        if abs(dj - fl) < epsilon:
            score += (epsilon - abs(dj - fl)) ** 2
    return score

def mca_localization(MDP, fingerprint_map, epsilon):
    best_score = -np.inf
    best_location = None
    mdp = MDP[0]
    for Yi, Fi in fingerprint_map:
        Fi = Fi[0]
        current_score = similarity_score(mdp, Fi, epsilon)
        if current_score > best_score:
            best_score = current_score
            best_location = Yi
    
    return best_location

scenario_name = 'BoxLectureRoom'
#scenario_name = 'LShapedRoom'
#scenario_name = 'LivingRoom'

#Paths to signal data and point coordinates
fingerprint_coord_path = f'scenarios/{scenario_name}/csv/reference.csv'
testing_coord_path = f'scenarios/{scenario_name}/csv/testing_points.csv'
fingerprint_data_path = f'scenarios/{scenario_name}/data/reference.json'
testing_data_path = f'scenarios/{scenario_name}/data/testing.json'

#Get coords of reference and testing points
reference_coords = read_point_coordinates(fingerprint_coord_path)
testing_coords = read_point_coordinates(testing_coord_path)

#Offline phase: prepare fingerprint map
reference_MDP = get_MDP_from_data(fingerprint_data_path)
fingerprint_map = list(zip(reference_coords, reference_MDP))

#Get testing point data
testing_MDP = get_MDP_from_data(testing_data_path)
if scenario_name == 'BoxLectureRoom' or scenario_name == 'LShapedRoom':
    testing_obstacles_data_path = f'scenarios/{scenario_name}/data/testing_obstacles.json'
    testing_obstacles_MDP = get_MDP_from_data(testing_obstacles_data_path)

#Localize testing points using mca
chosen_e = 0
maxerror = 100
for i in np.linspace(0.1, 2, 100):
    estimated = []
    for MDP in testing_MDP:
        estimated.append(mca_localization(MDP, fingerprint_map, epsilon=i))
    estimated = np.array(estimated)
    errors = calculate_position_errors(estimated, testing_coords)
    current_error = max(errors)
    if current_error < maxerror:
        chosen_e = i
        maxerror = current_error

print(f'Chosen threshold: {chosen_e}')
estimated_positions = []
for MDP in testing_MDP:
    estimated_positions.append(mca_localization(MDP, fingerprint_map, epsilon=chosen_e))
estimated_positions = np.array(estimated_positions)

#Calculate and visualize the localization erros 
errors = calculate_position_errors(estimated_positions, testing_coords)
print(errors)
print(errors.mean())
visualize_error_cdf(errors, k=1, title_addition='(No obstacles)')


#Repeat for scenario with obstacles
if scenario_name == 'BoxLectureRoom' or scenario_name == 'LShapedRoom':
    estimated_positions_obstacles = []
    for MDP_obs in testing_obstacles_MDP:
        estimated_positions_obstacles.append(mca_localization(MDP_obs, fingerprint_map, epsilon=chosen_e))
    estimated_positions_obstacles = np.array(estimated_positions_obstacles)
    errors_obstacles = calculate_position_errors(estimated_positions_obstacles, testing_coords)
    print(errors_obstacles)
    print(errors_obstacles.mean())
    visualize_error_cdf(errors_obstacles, k=1, title_addition='(With obstacles)')
