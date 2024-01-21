import csv
import math

def generate__grid_granularity(width, length, granularity):
    grid = []
    for i in range(0, width, granularity):
        for j in range(0, length, granularity):
            grid.append((i, j))
    return grid

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


def generate_grid_quadrants(width, length, num_quadrants):
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
            grid.append((center_x, center_y))
    return grid

def save_to_csv(grid, filename):
    with open(filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        #csv_writer.writerow(['X', 'Y'])  # Header row
        csv_writer.writerows(grid)


grid = generate_grid_quadrants(width=10, length=2, num_quadrants=20)
grid2 = generate_grid_with_min_distance(width=10, length=2, min_distance=1)

#Save the grid to a CSV file
csv_filename1 = 'grid1.csv'
csv_filename2 = 'grid2.csv'
save_to_csv(grid, csv_filename1)
save_to_csv(grid2, csv_filename2)
