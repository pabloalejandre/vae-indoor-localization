import matplotlib.pyplot as plt
import matplotlib.patches as patches
import csv
import numpy as np
import pandas as pd

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

def draw_points_in_room(length, width, transmitter, points1, points2=None, show_index=True, LRoom=False, draw_grid=False):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Set the aspect of the plot to be equal and the limits
    ax.set_aspect('equal')
    ax.set_xlim([0, length])
    ax.set_ylim([0, width])

    # Create and add the rectangle representing the room
    room_rect = patches.Rectangle((0, 0), length, width, linewidth=1, edgecolor='black', facecolor='none')
    ax.add_patch(room_rect)

    # Optionally draw the L-shape
    if LRoom:
        L_shape = patches.Polygon([[0, 6], [6, 6], [6, 19], [0, 19], [0, 6]], closed=True, linewidth=1, edgecolor='black', facecolor='none')
        ax.add_patch(L_shape)

    # Optionally draw the grid based on points1
    if draw_grid and points1:
        x_coords, y_coords = zip(*points1)
        grid_size_x = np.min(np.diff(sorted(set(x_coords))))
        grid_size_y = np.min(np.diff(sorted(set(y_coords))))
        
        for x in np.arange(0, length + grid_size_x, grid_size_x):
            ax.axvline(x, color='lightgrey', linewidth=0.5)
        for y in np.arange(0, width + grid_size_y, grid_size_y):
            ax.axhline(y, color='lightgrey', linewidth=0.5)

    # Plot transmitter with increased size and higher zorder
    for i, (x, y) in enumerate(transmitter):
        ax.plot(x, y, 'r^', zorder=4, clip_on=False, label='Transmitter' if i == 0 else "")
        ax.text(x+0.25, y + 0.25, 'tx', color='red', ha='center', va='bottom', zorder=5)

    # Plot points from the first list and label them
    for i, (x, y) in enumerate(points1):
        ax.plot(x, y, 'bo', zorder=3, label='Ref. Points' if i == 0 else "")
        if show_index:
            ax.text(x, y + 0.25, str(i), color='blue', ha='center', va='bottom', zorder=3)

    # Plot points from the second list and label them, if provided
    if points2:
        for i, (x, y) in enumerate(points2):
            ax.plot(x, y, 'go', zorder=3, label='Testing Points' if i == 0 else "")
            if show_index:
                ax.text(x, y + 0.25, str(i), color='green', ha='center', va='bottom', zorder=3)

    # Display the legend outside the plot area
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

    # Adjust the title position
    fig.suptitle('Indoor Geometry', y=0.95)  # Adjust the y value as needed for optimal positioning

    plt.tight_layout()
    plt.show()


#######---------------------------------------------------------###########
#######                   Visualizing MDPs                      ###########
#######---------------------------------------------------------###########

def visualize_mdps(list_mdp1, list_mdp2, label1='', label2='', title=''):
    num_pairs = len(list_mdp1)
    
    fig, axs = plt.subplots(1, num_pairs, figsize=(4*num_pairs, 8))
    
    global_min = min(min(np.array(mdp1).min(), np.array(mdp2).min()) for mdp1, mdp2 in zip(list_mdp1, list_mdp2))
    global_max = max(max(np.array(mdp1).max(), np.array(mdp2).max()) for mdp1, mdp2 in zip(list_mdp1, list_mdp2))
    
    for i, (mdp1, mdp2) in enumerate(zip(list_mdp1, list_mdp2)):
        ax = axs[i]
        y1 = np.array(mdp1)
        y2 = np.array(mdp2)
        x1_points = np.full_like(y1, 0.9) + np.random.normal(0, 0.01, size=y1.shape) 
        x2_points = np.full_like(y2, 1.1) + np.random.normal(0, 0.01, size=y2.shape)

        ax.scatter(x1_points, y1, color='royalblue', label=label1 if i == 0 else "", s=50, alpha=0.5)
        ax.scatter(x2_points, y2, color='darkblue', label=label2 if i == 0 else "", s=50, alpha=0.5)
        
        ax.set_xlim(0.7, 1.3)
        ax.set_ylim(global_min-1, global_max+1)
        ax.set_xlabel(f'Point {i}')
        if i == 0:
            ax.set_ylabel('Distance (m)')
            ax.legend(loc='lower left', bbox_to_anchor=(0, 1), ncol=1, fancybox=True, shadow=True)
        else:
            ax.set_yticklabels([])
        
        ax.set_xticks([])
        ax.grid()

    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.9, wspace=0.2)
    fig.suptitle(title, fontsize=16, y=0.95)
    plt.show()

#######---------------------------------------------------------###########
#######       Visualizing VAE latent and decoder outputs        ###########
#######---------------------------------------------------------###########

def visualize_reconstructions(list_mdp1, list_mdp2, title=''):
    num_pairs = len(list_mdp1)  # Assuming list_mdp1 and list_mdp2 are of the same length
    
    # Adjust the height of the figure to reduce vertical padding and ensure legend is visible
    fig, axs = plt.subplots(1, num_pairs, figsize=(4*num_pairs, 8))  # Reduced height from 15 to 8
    
    # Find global y-axis limits
    global_min = min(min(np.array(mdp1).min(), np.array(mdp2).min()) for mdp1, mdp2 in zip(list_mdp1, list_mdp2))
    global_max = max(max(np.array(mdp1).max(), np.array(mdp2).max()) for mdp1, mdp2 in zip(list_mdp1, list_mdp2))
    
    for i, (mdp1, mdp2) in enumerate(zip(list_mdp1, list_mdp2)):
        ax = axs[i]
        y1 = np.array(mdp1)
        y2 = np.array(mdp2)
        x1_points = np.full_like(y1, 0.9) + np.random.normal(0, 0.01, size=y1.shape)
        x2_points = np.full_like(y2, 1.1) + np.random.normal(0, 0.01, size=y1.shape) 

        ax.scatter(x1_points, y1, color='royalblue', label='Original MPCs' if i == 0 else "", s=50, alpha=0.5)
        ax.scatter(x2_points, y2, color='darkblue', label='Reconstructed MPCs' if i == 0 else "", s=50, alpha = 0.5)
        
        ax.set_xlim(0.7, 1.3)
        ax.set_ylim(global_min-1, global_max+1)
        ax.set_xlabel(f'Point {i}')
        if i == 0:
            ax.set_ylabel('Distance (m)')
            # Place the legend inside the plot area at the top left corner
            ax.legend(loc='lower left', bbox_to_anchor=(0, 1), ncol=1, fancybox=True, shadow=True)
        else:
            ax.set_yticklabels([])
        
        ax.set_xticks([])
        ax.grid()

    # Adjust subplot parameters to use available space more effectively
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.9, wspace=0.2)
    fig.suptitle('VAE MDP Reconstructions: ' + title, fontsize=16, y=0.95)  # Adjust y parameter to position suptitle
    plt.show()


def visualize_2D_latent_space(array1, array2=None, title=''):
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
    plt.scatter(array1[:, 0], array1[:, 1], marker='o', color='blue', label='Reference Points')
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
    plt.xlabel('$z_1$')
    plt.ylabel('$z_2$')
    plt.title('2-Dimensional Latent Space: ' + title)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

