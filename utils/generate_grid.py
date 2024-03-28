import csv
import math
import random

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

def generate_random_points(width, length, height, number_of_points):
    points = []
    for i in range(number_of_points):
        point = [random.uniform(0.0, float(width)), random.uniform(0.0,float(length)), height/2]
        points.append(point)
    return points

def generate_random_L_points(width, length, height, number_of_points):
    points = []
    while True:
        point = [random.uniform(0.0, float(width)), random.uniform(0.0,float(length)), height/2]
        if 0 <= point[0] <= 6 and 6 <= point[1] <= 19:
            continue  # If it is in the subtracted area, regenerate the coordinates
        else:
            points.append(point)
        if len(points) >= number_of_points:
            break
    return points

def divide_L_shaped_room_into_cells(x_size, y_size, x_cells, y_cells):
    cell_width = x_size / x_cells
    cell_height = y_size / y_cells
    grid = []

    for x_cell in range(x_cells):
        for y_cell in range(y_cells):
            # Calculate cell's center coordinates
            x_center = (x_cell * cell_width) + (cell_width / 2)
            y_center = (y_cell * cell_height) + (cell_height / 2)

            # Check if the cell falls within the subtracted area
            if 0 <= x_center <= 6 and 6 <= y_center <= 19:
                # This cell's center is in the excluded area, so we skip it
                continue
            else:
                # Add the cell to the grid, including its center and size
                grid.append([x_center, y_center, 1.5])
    
    return grid


def save_to_csv(grid, filename):
    with open(filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerows(grid)
