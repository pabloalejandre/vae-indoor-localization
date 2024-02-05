#This file gets the coordinates and MDPs and visualizes them.
#Visualizes the distribution of fingerprints and testing points by displaying the coordinates on the whole grid.
#Visualizes the MDPS using a heat map for each transmitter

from utils.data_utils import get_MDP_from_data
from utils.visualize_utils import read_point_coordinates, draw_rectangle_with_grid_and_indexed_points, visualize_MDPS
import matplotlib.pyplot as plt

#Load fingerprints and testing points (coords and MDPs)
fingerprint_data_path = 'data/BoxRoom_4Transmitters/fingerprints.json'
fingerprint_csv_path = 'csv/BoxRoom_4Transmitters/testing/fingerprints.csv'
fingerprints_coords = read_point_coordinates(fingerprint_csv_path)
fingerprints_MDP = get_MDP_from_data(path=fingerprint_data_path, receiver_id=4, number_of_transmitters=4, timestamps=12, length=None) 

testing1_data_path = 'data/BoxRoom_4Transmitters/testing1.json'
testing1_csv_path = 'csv/BoxRoom_4Transmitters/testing/testing1.csv'
testing1_coords = read_point_coordinates(testing1_csv_path)
testing1_MDP = get_MDP_from_data(path=testing1_data_path, receiver_id=4, number_of_transmitters=4, timestamps=12, length=None)

testing2_data_path = 'data/BoxRoom_4Transmitters/testing2.json'
testing2_csv_path = 'csv/BoxRoom_4Transmitters/testing/testing2.csv'
testing2_coords = read_point_coordinates(testing2_csv_path)
testing2_MDP = get_MDP_from_data(path=testing2_data_path, receiver_id=4, number_of_transmitters=4, timestamps=12, length=None)


#Visualize coordinates and MDPS of fingerprints and testing points
#draw_rectangle_with_grid_and_indexed_points(10, 19, fingerprints_coords, testing1_coords)
#draw_rectangle_with_grid_and_indexed_points(10, 19, fingerprints_coords, testing2_coords)

#visualize_MDPS(fingerprints_MDP)
#visualize_MDPS(testing1_MDP)
#visualize_MDPS(testing2_MDP)