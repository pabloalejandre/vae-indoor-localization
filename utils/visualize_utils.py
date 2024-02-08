import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import csv
import numpy as np

#######---------------------------------------------------------###########
#######   Visualizing spatial coords of points in indoor env.   ###########
#######---------------------------------------------------------###########

def read_point_coordinates(csv_file_path):
    """
    Reads point coordinates from a CSV file and returns them as a list of tuples.

    Parameters:
    csv_file_path (str): The file path of the CSV file.

    Returns:
    list of tuples: A list containing the (x, y) coordinates.
    """
    points = []
    with open(csv_file_path, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if len(row) >= 2:
                try:
                    x, y = float(row[0]), float(row[1])
                    points.append((x, y))
                except ValueError:
                    # Skip the row if it cannot be converted to floats
                    pass

    return points


def draw_rectangle_with_grid_and_indexed_points(length, width, points1, points2):
    """
    Draws a rectangle, divides it into a grid based on the number of points in the first list,
    and plots two sets of points with their indices on top, starting at 0. Points from the first list are plotted at the
    center of each grid cell, while points from the second list are plotted at their respective coordinates.

    Parameters:
    length (float): The length of the rectangle.
    width (float): The width of the rectangle.
    points1 (list of tuples): A list of (x, y) coordinates for the first set of points.
    points2 (list of tuples): A list of (x, y) coordinates for the second set of points.
    """

    # Calculate the number of rows and columns for the grid based on the first list of points
    num_points = len(points1)
    num_cols = int(math.ceil(math.sqrt(num_points)))
    num_rows = int(math.ceil(num_points / num_cols))

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Set the aspect of the plot to be equal
    ax.set_aspect('equal')

    # Create a rectangle
    rect = patches.Rectangle((0, 0), length, width, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

    # Calculate the size of each grid cell
    cell_width = length / num_cols
    cell_height = width / num_rows

    # Draw grid lines
    for col in range(1, num_cols):
        ax.axvline(x=col * cell_width, color='gray', linestyle='--')
    for row in range(1, num_rows):
        ax.axhline(y=row * cell_height, color='gray', linestyle='--')

    # Plot the first set of points in their respective grid cells with index on top (starting at 0)
    for i, point in enumerate(points1):
        row = i // num_cols
        col = i % num_cols
        center_x = (col + 0.5) * cell_width
        center_y = (row + 0.5) * cell_height

        # Plot the point
        ax.plot(center_x, center_y, 'bo')  # blue circle markers
        ax.text(center_x, center_y, str(i), ha='center', va='bottom', color='blue')

    # Plot the second set of points at their respective coordinates with index on top (starting at 0)
    for i, point in enumerate(points2):
        # Plot the point
        ax.plot(point[0], point[1], 'go')  # green circle markers
        ax.text(point[0], point[1], str(i), ha='center', va='bottom', color='green')

    # Set the limits of the plot to the size of the rectangle
    ax.set_xlim([0, length])
    ax.set_ylim([0, width])

    plt.show()

#######---------------------------------------------------------###########
#######         Visualizing MDP and latent features             ###########
#######---------------------------------------------------------###########

def visualize_mdps(mdp1, mdp2):
    # Prepare the data
    y1 = np.array(mdp1)
    y2 = np.array(mdp2)
    x1_points = np.full_like(y1, 1)
    x2_points = np.full_like(y2, 2)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(x1_points, y1, color='blue', label='Point 2')
    plt.scatter(x2_points, y2, color='red', label='Point 1')
    
    # Annotate each point with its value
    for i, value in enumerate(y1):
        plt.text(1, value, str(value), fontsize=9, ha='right')
    for i, value in enumerate(y2):
        plt.text(2, value, str(value), fontsize=9, ha='left')
    
    # Labels and legend
    plt.ylabel('Distances')
    plt.legend()
    plt.axis('equal')
    plt.grid()
    plt.tight_layout()
    plt.show()

def visualize_2D_latent_representations(array1, array2=None):
    # Define offset values
    horizontal_offset = 0.05
    vertical_offset = 0.05

    diagonal_offset_x = 0.05
    diagonal_offset_y = 0.05

    slight_horizontal_offset = 0.05  # Slightly less offset for horizontal variation
    slight_vertical_offset = 0.05   # Slightly less offset for vertical variation

    # Expand the offsets list to include more positions
    offsets = [
        (0, vertical_offset),                     # top
        (0, -vertical_offset),                    # bottom
        (horizontal_offset, 0),                   # right
        (-horizontal_offset, 0),                  # left
        (diagonal_offset_x, diagonal_offset_y),   # top-right
        (-diagonal_offset_x, diagonal_offset_y),  # top-left
        (diagonal_offset_x, -diagonal_offset_y),  # bottom-right
        (-diagonal_offset_x, -diagonal_offset_y), # bottom-left
        # Additional positions
        (slight_horizontal_offset, slight_vertical_offset),   # slight top-right
        (-slight_horizontal_offset, slight_vertical_offset),  # slight top-left
        (slight_horizontal_offset, -slight_vertical_offset),  # slight bottom-right
        (-slight_horizontal_offset, -slight_vertical_offset), # slight bottom-left
        (0, 2*vertical_offset),                    # further top
        (0, -2*vertical_offset),                   # further bottom
        (2*horizontal_offset, 0),                  # further right
        (-2*horizontal_offset, 0)                  # further left
    ]
    
    plt.figure(figsize=(8, 6))
    # Create a scatter plot for the fingerprints
    plt.scatter(array1[:, 0], array1[:, 1], marker='o', color='blue', label='Fingerprints')
    for i, (x, y) in enumerate(array1):
        offset = offsets[i % len(offsets)]  # Cycle through offsets
        # Adjust position with offset
        adjusted_x = x + offset[0]
        adjusted_y = y + offset[1]
        plt.text(adjusted_x, adjusted_y, f'{i}', ha='center', va='center', color='navy')
    
    if array2 is not None:
        # Create a scatter plot for the testing points if provided
        plt.scatter(array2[:, 0], array2[:, 1], marker='o', color='green', label='Testing Points')
        for i, (x, y) in enumerate(array2):
            offset = offsets[i % len(offsets)]  # Cycle through offsets
            # Adjust position with offset
            adjusted_x = x + offset[0]
            adjusted_y = y + offset[1]
            plt.text(adjusted_x, adjusted_y, f'{i}', ha='center', va='center', color='darkgreen')
    
    plt.axis('equal')
    plt.grid()
    plt.tight_layout()
    plt.show()


