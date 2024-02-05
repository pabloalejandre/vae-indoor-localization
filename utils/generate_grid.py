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
transmitter1_grid = create_fixed_grid(x=0, y=9.5, z=1.5, entries=1500)
#transmitter2_grid = create_fixed_grid(x=10, y=9.5, z=1.5, entries=1000)
#transmitter3_grid = create_fixed_grid(x=5, y=0, z=1.5, entries=1000)
#transmitter4_grid = create_fixed_grid(x=5, y=19, z=1.5, entries=1000)

#training_grid = generate_grid_number_of_cells(width=10, length=19, height= 3, num_quadrants=1000)
training_random_grid = generate_random_testing_points(width=10, length=19, height=3, number_of_points=1500)

rotation_grid = create_zero_grid(entries=1500) 

#Evaluation Grids
# transmitter1_grid = create_fixed_grid(x=0, y=9.5, z=1.5, entries=12)
# transmitter2_grid = create_fixed_grid(x=10, y=9.5, z=1.5, entries=12)
# transmitter3_grid = create_fixed_grid(x=5, y=0, z=1.5, entries=12)
# transmitter4_grid = create_fixed_grid(x=5, y=19, z=1.5, entries=12)

# fingerprint_grid = generate_grid_number_of_cells(width=10, length=19, height=3, num_quadrants=10)
# testing_points_grid = generate_random_testing_points(width=10, length=19, height=3, number_of_points=12)
# rotation_grid = create_zero_grid(entries=12) 

#Save the grids to csv files
save_to_csv(transmitter1_grid, 'csv/BoxRoom_1Transmitter/training/transmitter1.csv')
#save_to_csv(transmitter2_grid, 'csv/1TransmitterGrids/testing/transmitter2.csv')
#save_to_csv(transmitter3_grid, 'csv/1TransmitterGrids/testing/transmitter3.csv')
#save_to_csv(transmitter4_grid, 'csv/1TransmitterGrids/testing/transmitter4.csv')
#save_to_csv(fingerprint_grid, 'csv/1TransmitterGrids/testing/fingerprints.csv')
#save_to_csv(testing_points_grid, 'csv/1TransmitterGrids/testing/testing_points.csv')
save_to_csv(training_random_grid, 'csv/BoxRoom_1Transmitter/training/training.csv')
save_to_csv(rotation_grid, 'csv/BoxRoom_1Transmitter/training/rotation.csv')
