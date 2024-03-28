from utils.data_utils import get_MDP_from_data, preprocess_mdps
from utils.visualize_utils import visualize_mdps

#scenario_name, feature_length = 'BoxLectureRoom', 14
#scenario_name, feature_length = 'LShapedRoom', 15
scenario_name, feature_length = 'LivingRoom', 16

#Paths to signal data and coordinates
testing_data_path = f'scenarios/{scenario_name}/data/testing.json'
testing_obstacles_data_path = f'scenarios/{scenario_name}/data/testing.json'

#Raw Data
testing_MDP = get_MDP_from_data(testing_data_path)
testing_obstacles_MDP = get_MDP_from_data(testing_obstacles_data_path)

#Processed Data
testing_MDP_processed = preprocess_mdps(testing_MDP, feature_length)
testing_obstacles_MDP_processed = preprocess_mdps(testing_obstacles_MDP, feature_length)

visualize_mdps(testing_MDP, testing_obstacles_MDP)
visualize_mdps(testing_MDP_processed, testing_obstacles_MDP_processed)

