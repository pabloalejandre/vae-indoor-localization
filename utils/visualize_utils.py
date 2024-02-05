import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import csv
import numpy as np

#######---------------------------------------------------------###########
#######     Visualizing spatial features of points              ###########
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

    # Show the plot
    plt.show()

#######---------------------------------------------------------###########
#######     Visualizing MDP and latent features                 ###########
#######---------------------------------------------------------###########

def visualize_MDPS(mdps):
    nrows = 3
    ncols = 4
    fig, axs = plt.subplots(nrows, ncols, figsize=(15, 10))
    for i in range(len(mdps)):
            row_idx = nrows - 1 - (i // ncols)
            col_idx = i % ncols
            ax = axs[row_idx, col_idx]
            im = ax.imshow(mdps[i], aspect='auto', cmap='viridis')
            ax.set_title(f'Fingerprint {i}')
            plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.show()
    

def visualize_1D_latent_representations(latent_array):
    # Create a scatter plot with the values as coordinates on the number line
    plt.scatter(latent_array, np.zeros_like(latent_array), marker='o', color='blue')
    # Add text labels for each point with their index, adjusting position for overlap
    for i, value in enumerate(latent_array):
        offset = (0.005 * (-1) ** i)  # Adjust the offset to alternate above and below the points
        plt.text(value, offset, f'{i}', ha='center', va='center')
    plt.grid()
    plt.tight_layout()
    plt.show()


def visualize_2D_latent_representations(latent_array):
    # Extract x and y coordinates from the 2D array
    x_coords = latent_array[:, 0]
    y_coords = latent_array[:, 1]
    
    # Create a scatter plot with the 2D points
    plt.scatter(x_coords, y_coords, marker='o', color='blue')
    
    # Add text labels for each point with their index
    for i, (x, y) in enumerate(zip(x_coords, y_coords)):
        plt.text(x, y, f'{i}', ha='center', va='bottom')
    
    plt.grid()
    plt.tight_layout()
    plt.show()
