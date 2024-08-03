#
#
#      0===========================================================0
#      |    TP1 Basic structures and operations on point clouds    |
#      0===========================================================0
#
#
# ------------------------------------------------------------------------------------------
#
#      Second script of the practical session. Subsampling of a point cloud
#
# ------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 13/12/2017
#


# ------------------------------------------------------------------------------------------
#
#          Imports and global variables
#      \**********************************/
#


# Import numpy package and name it "np"
import numpy as np

# Import functions to read and write ply files
from ply import write_ply, read_ply

# Import time package
import time


# ------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define useful functions to be used in the main
#


def cloud_decimation(points, colors, labels, factor):

    # YOUR CODE
    decimated_points = points[::factor]
    decimated_colors = colors[::factor]
    decimated_labels = labels[::factor]

    return decimated_points, decimated_colors, decimated_labels


def grid_subsampling(points, colors, labels, voxel_size):
    """
    Subsample a point cloud using a grid approach. Each voxel keeps only one point.
    The representative point can be chosen as the barycenter of the points in the voxel.
    The color can be the average color of the points in the voxel.
    The label can be the most common label of the points in the voxel.
    """
    # Compute the indices of the grid cell each point falls into
    grid_indices = np.floor(points / voxel_size).astype(np.int32)

    # Create dictionaries to hold points, colors, and labels for each grid cell
    voxel_dict = {}
    color_dict = {}
    label_dict = {}

    for point, color, label, index in zip(points, colors, labels, grid_indices):
        # Convert the index array to a tuple so it can be used as a dictionary key
        key = tuple(index)
        if key not in voxel_dict:
            voxel_dict[key] = []
            color_dict[key] = []
            label_dict[key] = []
        voxel_dict[key].append(point)
        color_dict[key].append(color)
        label_dict[key].append(label)

    # For each voxel, compute the representative point, color, and label
    subsampled_points = []
    subsampled_colors = []
    subsampled_labels = []

    for key in voxel_dict.keys():
        # Compute the barycenter for the points
        subsampled_points.append(np.mean(voxel_dict[key], axis=0))
        # Compute the average color
        subsampled_colors.append(np.mean(color_dict[key], axis=0).astype(np.uint8))
        # Choose the most common label
        subsampled_labels.append(np.bincount(label_dict[key]).argmax())

    return (np.array(subsampled_points), 
            np.array(subsampled_colors), 
            np.array(subsampled_labels))





# ------------------------------------------------------------------------------------------
#
#           Main
#       \**********/
#
# 
#   Here you can define the instructions that are called when you execute this file
#

if __name__ == '__main__':

    # Load point cloud
    # ****************
    #
    #   Load the file '../data/indoor_scan.ply'
    #   (See read_ply function)
    #

    # Path of the file
    file_path = '../data/indoor_scan.ply'

    # Load point cloud
    data = read_ply(file_path)

    # Concatenate data
    points = np.vstack((data['x'], data['y'], data['z'])).T
    colors = np.vstack((data['red'], data['green'], data['blue'])).T
    labels = data['label']

    # Decimate the point cloud
    # ************************
    #
    if False: # decimation
        # Define the decimation factor
        factor = 300

        # Decimate
        t0 = time.time()
        decimated_points, decimated_colors, decimated_labels = cloud_decimation(points, colors, labels, factor)
        t1 = time.time()
        print('decimation done in {:.3f} seconds'.format(t1 - t0))

        # Save
        write_ply('../data/decimated.ply', [decimated_points, decimated_colors, decimated_labels], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])
        print('Done')

    if True: # grid subsampling
        # Define the size of the grid
        voxel_size = 0.05

        # Subsample
        t0 = time.time()
        subsampled_points, subsampled_colors, subsampled_labels = grid_subsampling(points, colors, labels, voxel_size)
        t1 = time.time()
        print('Subsampling done in {:.3f} seconds'.format(t1 - t0))

        # Cast colors to unsigned 8-bit integers
        subsampled_colors = subsampled_colors.astype(np.uint8)
        # Cast labels to integers (if they are not already)
        subsampled_labels = subsampled_labels.astype(np.int32)
        # Save
        write_ply('../data/grid_subsampled.ply', [subsampled_points, subsampled_colors, subsampled_labels], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])
        print('Done')
