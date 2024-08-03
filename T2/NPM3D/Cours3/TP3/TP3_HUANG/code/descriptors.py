#
#
#      0=============================0
#      |    TP4 Point Descriptors    |
#      0=============================0
#
#
# ------------------------------------------------------------------------------------------
#
#      Script of the practical session
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

# Import library to plot in python
from matplotlib import pyplot as plt

# Import functions from scikit-learn
from sklearn.neighbors import KDTree

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
#   Here you can define usefull functions to be used in the main
#



def PCA(points):

    # Calculate the centroid of the points
    centroid = np.mean(points, axis=0)
    
    # Center the points around the centroid
    centered_points = points - centroid
    
    # Calculate the covariance matrix
    cov_matrix = np.cov(centered_points.T)
    
    # Compute the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # Ensure that eigenvalues and eigenvectors are real
    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)

    # Sort the eigenvalues and eigenvectors in ascending order by eigenvalue
    idx = eigenvalues.argsort()
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:,idx]
    
    return eigenvalues, eigenvectors




"""
def compute_local_PCA(query_points, cloud_points, k):

    # This function needs to compute PCA on the neighborhoods of all query_points in cloud_points

    all_eigenvalues = np.zeros((query_points.shape[0], 3))
    all_eigenvectors = np.zeros((query_points.shape[0], 3, 3))

    # Create a KDTree for efficient neighbor search
    tree = KDTree(cloud_points)

    # Search for all neighbors once

    _, neighbors_list = tree.query(query_points, k)


    for i, neighbors in enumerate(neighbors_list):
        if len(neighbors) < 2:  # Cannot compute PCA with less than 2 points
            # Handle the edge case by setting eigenvalues to zero and eigenvectors to identity
            all_eigenvalues[i, :] = np.zeros(3)
            all_eigenvectors[i, :, :] = np.eye(3)
        else:
            neighborhood_points = cloud_points[neighbors]
            eigenvalues, eigenvectors = PCA(neighborhood_points)
            all_eigenvalues[i, :] = eigenvalues
            all_eigenvectors[i, :, :] = eigenvectors

    return all_eigenvalues, all_eigenvectors
"""


def compute_local_PCA(query_points, cloud_points, radius):

    # This function needs to compute PCA on the neighborhoods of all query_points in cloud_points

    all_eigenvalues = np.zeros((query_points.shape[0], 3))
    all_eigenvectors = np.zeros((query_points.shape[0], 3, 3))

    # Create a KDTree for efficient neighbor search
    tree = KDTree(cloud_points)

    # Search for all neighbors once

    neighbors_list = tree.query_radius(query_points, radius)


    for i, neighbors in enumerate(neighbors_list):
        if len(neighbors) < 2:  # Cannot compute PCA with less than 2 points
            # Handle the edge case by setting eigenvalues to zero and eigenvectors to identity
            all_eigenvalues[i, :] = np.zeros(3)
            all_eigenvectors[i, :, :] = np.eye(3)
        else:
            neighborhood_points = cloud_points[neighbors]
            eigenvalues, eigenvectors = PCA(neighborhood_points)
            all_eigenvalues[i, :] = eigenvalues
            all_eigenvectors[i, :, :] = eigenvectors

    return all_eigenvalues, all_eigenvectors




def compute_features(query_points, cloud_points, radius):

    # Compute local PCA to get eigenvalues for each point
    all_eigenvalues, all_eigenvectors = compute_local_PCA(query_points, cloud_points, radius)
    
    # Initialize feature arrays
    verticality = np.zeros(query_points.shape[0])
    linearity = np.zeros(query_points.shape[0])
    planarity = np.zeros(query_points.shape[0])
    sphericity = np.zeros(query_points.shape[0])
    
    # Compute features for each point
    for i, eigenvalues in enumerate(all_eigenvalues):
        # Avoid division by zero by adding a small epsilon to denominators
        epsilon = 1e-6
        lambda_1, lambda_2, lambda_3 = eigenvalues + epsilon

        # Verticality
        # n_z is the Z component (third component) of the normal (first eigenvector)
        n_z = all_eigenvectors[i, 2, 0]  # Corrected component selection
        verticality[i] = 1 - (2 * np.arcsin(np.abs(n_z)) / np.pi)
        
        # Linearity
        linearity[i] = (lambda_1 - lambda_2) / (lambda_1 + epsilon)
        
        # Planarity
        planarity[i] = (lambda_2 - lambda_3) / (lambda_1 + epsilon)
        
        # Sphericity
        sphericity[i] = lambda_3 / (lambda_1 + epsilon)

    return verticality, linearity, planarity, sphericity



# ------------------------------------------------------------------------------------------
#
#           Main
#       \**********/
#
# 
#   Here you can define the instructions that are called when you execute this file
#

if __name__ == '__main__':

    # PCA verification
    # ****************
    if False:

        # Load cloud as a [N x 3] matrix
        cloud_path = '../data/Lille_street_small.ply'
        cloud_ply = read_ply(cloud_path)
        cloud = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T

        # Compute PCA on the whole cloud
        eigenvalues, eigenvectors = PCA(cloud)

        # Print your result
        print(eigenvalues)

        # Expected values :
        #
        #   [lambda_3; lambda_2; lambda_1] = [ 5.25050177 21.7893201  89.58924003]
        #
        #   (the convention is always lambda_1 >= lambda_2 >= lambda_3)
        #

		
    # Normal computation
    # ******************
    if False:

        # Load cloud as a [N x 3] matrix
        cloud_path = '../data/Lille_street_small.ply'
        cloud_ply = read_ply(cloud_path)
        cloud = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T

        # Compute PCA on the whole cloud
        #all_eigenvalues, all_eigenvectors = compute_local_PCA(cloud, cloud, 0.50)
        all_eigenvalues, all_eigenvectors = compute_local_PCA(cloud, cloud, 30)

        # The normals are the eigenvectors corresponding to the smallest eigenvalue, i.e. the first eigenvector for each point
        normals = all_eigenvectors[:, :, 0]

        # Save cloud with normals
        write_ply('../data/Lille_street_small_normals.ply', (cloud, normals), ['x', 'y', 'z', 'nx', 'ny', 'nz'])
		

    # Features computation
    # ********************
    if True:

        # Load cloud as a [N x 3] matrix
        cloud_path = '../data/Lille_street_small.ply'
        cloud_ply = read_ply(cloud_path)
        cloud = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T

        # Compute the 4 features for all points
        verticality, linearity, planarity, sphericity = compute_features(cloud, cloud, 0.50)

        # Save cloud with the 4 features
        write_ply('../data/Lille_street_small_features.ply', [cloud, verticality, linearity, planarity, sphericity], ['x', 'y', 'z', 'verticality', 'linearity', 'planarity', 'sphericity'])
        print('Done')