#
#
#      0===========================================================0
#      |                      TP6 Modelisation                     |
#      0===========================================================0
#
#
#------------------------------------------------------------------------------------------
#
#      Plane detection with RANSAC
#
#------------------------------------------------------------------------------------------
#
#      Xavier ROYNARD - 19/02/2018
#


#------------------------------------------------------------------------------------------
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


# Import functions from scikit-learn
from sklearn.neighbors import KDTree



#------------------------------------------------------------------------------------------
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



def compute_plane(points):

    point_plane = np.zeros((3,1))
    normal_plane = np.zeros((3,1))
    
    # TODO:

    # Extract three points from the input
    p0 = points[0]
    p1 = points[1]
    p2 = points[2]
    
    # Vectors representing edges of the plane triangle
    v0 = p1 - p0
    v1 = p2 - p0
    
    # Compute the normal of the plane using the cross product
    normal_plane = np.cross(v0, v1)
    # Normalize the normal vector
    normal_plane /= np.linalg.norm(normal_plane)
    
    # Choose one of the points as a point on the plane
    point_plane = p0
    
    return point_plane, normal_plane



def in_plane_normals(points, normals, pt_plane, normal_plane, threshold_in=0.1, threshold_angle=np.deg2rad(10)):
    
    indexes = np.zeros(len(points), dtype=bool)
    
    # TODO:

    # Calculate the distance from each point to the plane
    distances = np.abs(np.dot(points - pt_plane, normal_plane))
    
    # Calculate the angular difference between normals of points and plane normal
    angular_diff = np.arccos(np.abs(np.dot(normals, normal_plane)))
    
    # Determine inliers based on both distance and angular difference
    indexes = (distances < threshold_in) & (angular_diff < threshold_angle)
    
    return indexes



def RANSAC_normals(points, normals, nb_draws=100, threshold_in=0.1, threshold_angle=np.deg2rad(10)):
    
    best_vote = 0
    best_pt_plane = np.zeros((3,))
    best_normal_plane = np.zeros((3,))
    
    for _ in range(nb_draws):
        # Randomly sample 3 unique points and their normals
        sample_indices = np.random.choice(points.shape[0], 3, replace=False)
        sample_points = points[sample_indices]
        
        # Compute the plane from these points
        pt_plane, normal_plane = compute_plane(sample_points)
        
        # Determine inliers
        inliers = in_plane_normals(points, normals, pt_plane, normal_plane, threshold_in, threshold_angle)
        
        # Count votes
        vote_count = np.sum(inliers)
        
        # Update best plane if current vote count is higher
        if vote_count > best_vote:
            best_vote = vote_count
            best_pt_plane = pt_plane
            best_normal_plane = normal_plane
                
    return best_pt_plane, best_normal_plane, best_vote



def recursive_RANSAC_normals(points, normals, nb_draws=100, threshold_in=0.1, threshold_angle=np.deg2rad(10), nb_planes=2):
    
    nb_points = len(points)
    plane_inds = np.arange(0,0)
    plane_labels = np.arange(0,0)
    remaining_inds = np.arange(0,nb_points)
	
    # TODO:

    for i in range(nb_planes):
        # Use RANSAC to detect the best plane from the remaining points
        pt_plane, normal_plane, _ = RANSAC_normals(points[remaining_inds], normals[remaining_inds], nb_draws, threshold_in, threshold_angle)
        # Find inliers for the current plane
        inliers = in_plane_normals(points[remaining_inds], normals[remaining_inds], pt_plane, normal_plane, threshold_in, threshold_angle)
        
        # Store the indices of inliers relative to the original points array
        plane_inds = np.append(plane_inds, remaining_inds[inliers])
        
        # Store the plane's label
        plane_labels = np.append(plane_labels, np.full(np.sum(inliers), i))
        
        # Update the remaining indices by removing the inliers
        remaining_inds = remaining_inds[~inliers]
        
        # Break the loop if there are no remaining points
        if len(remaining_inds) == 0:
            break

    
    return plane_inds, remaining_inds, plane_labels



#------------------------------------------------------------------------------------------
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
    nb_points = len(points)


    # Compute PCA on the whole cloud
    all_eigenvalues, all_eigenvectors = compute_local_PCA(points, points, 30)

    # The normals are the eigenvectors corresponding to the smallest eigenvalue, i.e. the first eigenvector for each point
    normals = all_eigenvectors[:, :, 0]
    
    # Find "all planes" in the cloud
    # ***********************************
    #
    #
    
    print('\n--- 4) ---\n')
    
    # Define parameters of recursive_RANSAC_normals
    nb_draws = 100
    threshold_in = 0.10
    threshold_angle=np.deg2rad(10)
    nb_planes = 5
    
    # Recursively find best plane by RANSAC_normals
    t0 = time.time()
    plane_inds, remaining_inds, plane_labels = recursive_RANSAC_normals(points, normals, nb_draws, threshold_in, threshold_angle, nb_planes)
    t1 = time.time()
    print('recursive RANSAC done in {:.3f} seconds'.format(t1 - t0))
                
    # Save the best planes and remaining points
    write_ply('../data/best_planes_normals.ply', [points[plane_inds], colors[plane_inds], labels[plane_inds], plane_labels.astype(np.int32)], ['x', 'y', 'z', 'red', 'green', 'blue', 'label', 'plane_label'])
    write_ply('../data/remaining_points_best_planes_normals.ply', [points[remaining_inds], colors[remaining_inds], labels[remaining_inds]], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])
    
    
    print('Done')
    