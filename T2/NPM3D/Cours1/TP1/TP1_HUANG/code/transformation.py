#
#
#      0===========================================================0
#      |    TP1 Basic structures and operations on point clouds    |
#      0===========================================================0
#
#
# ------------------------------------------------------------------------------------------
#
#      First script of the practical session. Transformation of a point cloud
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


# ------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define useful functions to be used in the main
#


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

    # Path of the file
    file_path = '../data/bunny.ply'

    # Load point cloud
    data = read_ply(file_path)

    # Concatenate x, y, and z in a (N*3) point matrix
    points = np.vstack((data['x'], data['y'], data['z'])).T

    # Concatenate R, G, and B channels in a (N*3) color matrix
    colors = np.vstack((data['red'], data['green'], data['blue'])).T

    # Transform point cloud
    # *********************
    #
    #   Follow the instructions step by step
    #

    # Center the cloud on its centroid(centroid = mean of points in x, y, z)
    centroid = np.mean(points, axis=0)
    transformed_points = points - centroid
    # Divide its scale by a factor 2
    transformed_points = transformed_points / 2
    # Recenter the cloud at the origianl position
    transformed_points = transformed_points + centroid
    # Apply a -10cm translation along y-axis
    transformed_points = transformed_points + np.array([0, -0.1, 0])

    # Save point cloud
    # *********************
    #
    #   Save your result file
    #   (See write_ply function)
    #

    # Save point cloud
    write_ply('../data/little_bunny.ply', [transformed_points, colors], ['x', 'y', 'z', 'red', 'green', 'blue'])

    print('Done')
