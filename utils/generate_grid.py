import csv
import math
import random

#Functions to generate grids with different parameters for simulator input:
    #Number of cells grids for receiver nodes: either for training or fingerprinting
    #Fixed grid for transmitter nodes
    #Zero grid for rotation values of simulator
    #Random grid for testing points for localization / for training

def generate_grid_with_min_distance(width, length, min_distance):
    # Calculate the number of rows and columns based on the minimum distance
    num_columns = int(width // min_distance)
    num_rows = int(length // min_distance)
    # Adjust the actual distance between points to distribute them evenly
    actual_distance_x = width / num_columns
    actual_distance_y = length / num_rows
    # Generate the grid points
    grid = []
    for row in range(num_rows):
        for col in range(num_columns):
            point_x = (col + 0.5) * actual_distance_x
            point_y = (row + 0.5) * actual_distance_y
            grid.append((point_x, point_y))
    return grid

def generate_grid_number_of_cells(width, length, height, num_quadrants):
    # Calculate the number of rows and columns to make the grid as square as possible
    num_columns = int(math.ceil(math.sqrt(num_quadrants)))
    num_rows = int(math.ceil(num_quadrants / num_columns))
    # Calculate the width and height of each quadrant
    quadrant_width = width / num_columns
    quadrant_height = length / num_rows
    # Calculate the center coordinates of each quadrant
    grid = []
    for row in range(num_rows):
        for col in range(num_columns):
            center_x = (col + 0.5) * quadrant_width
            center_y = (row + 0.5) * quadrant_height
            grid.append((center_x, center_y, height/2))
    return grid

def create_fixed_grid(x, y, z, entries):
    return [[x, y, z] for _ in range(entries)]

def create_zero_grid(entries):
    return [[0, 0, 0] for _ in range(entries)]

def generate_random_testing_points(width, length, height, number_of_points):
    points = []
    for i in range(number_of_points):
        point = [random.uniform(0.0, float(width)), random.uniform(0.0,float(length)), height/2]
        points.append(point)
    return points

def save_to_csv(grid, filename):
    with open(filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerows(grid)


#Create grids     
#Training grids
transmitter_grid_train = create_fixed_grid(x=3.75, y=16, z=1.5, entries=1500)
transmitter_grid_test = create_fixed_grid(x=3.75, y=16, z=1.5, entries=12)

#Savee grids
save_to_csv(transmitter_grid_train, 'csv/transmitter1_train.csv')
save_to_csv(transmitter_grid_test, 'csv/transmitter1_test.csv')