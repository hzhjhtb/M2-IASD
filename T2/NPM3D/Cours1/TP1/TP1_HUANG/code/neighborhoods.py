#
#
#      0===========================================================0
#      |    TP1 Basic structures and operations on point clouds    |
#      0===========================================================0
#
#
# ------------------------------------------------------------------------------------------
#
#      Third script of the practical session. Neighborhoods in a point cloud
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

# Import functions from scikit-learn
from sklearn.neighbors import KDTree

# Import functions to read and write ply files
from ply import write_ply, read_ply

# Import time package
import time

# Import matplotlib colormaps
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define useful functions to be used in the main
#


def brute_force_spherical(queries, supports, radius):
    """
    For a chosen point p, the spherical neighborhood comprises the points situated less than a fixed radius from p
    """
    # YOUR CODE
    neighborhoods = []
    # For each query point, compute the distance to all support points
    for q in queries:
        dist = np.linalg.norm(supports - q, axis=1)
        # Select the points that are within the radius
        neighborhoods.append(supports[dist < radius].tolist())
    
    return neighborhoods


def brute_force_KNN(queries, supports, k):
    """
    For a chosen point p, the k-nearest neighbors comprises a fixed number of closest points to p
    """

    # YOUR CODE
    neighborhoods = []
    # For each query point, compute the distance to all support points
    for q in queries:
        dist = np.linalg.norm(supports - q, axis=1)
        # Select the k points that are closest
        neighborhoods.append(supports[np.argsort(dist)[:k]].tolist())

    return neighborhoods





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

    # Brute force neighborhoods
    # *************************
    #

    # If statement to skip this part if you want
    if False:

        # Define the search parameters
        neighbors_num = 100
        radius = 0.2
        num_queries = 10

        # Pick random queries
        random_indices = np.random.choice(points.shape[0], num_queries, replace=False)
        queries = points[random_indices, :]

        # Search spherical
        t0 = time.time()
        neighborhoods = brute_force_spherical(queries, points, radius)
        t1 = time.time()

        # Search KNN      
        neighborhoods = brute_force_KNN(queries, points, neighbors_num)
        t2 = time.time()

        # Print timing results
        print('{:d} spherical neighborhoods computed in {:.3f} seconds'.format(num_queries, t1 - t0))
        print('{:d} KNN computed in {:.3f} seconds'.format(num_queries, t2 - t1))

        # Time to compute all neighborhoods in the cloud
        total_spherical_time = points.shape[0] * (t1 - t0) / num_queries
        total_KNN_time = points.shape[0] * (t2 - t1) / num_queries
        print('Computing spherical neighborhoods on whole cloud : {:.0f} hours'.format(total_spherical_time / 3600))
        print('Computing KNN on whole cloud : {:.0f} hours'.format(total_KNN_time / 3600))

 



    # KDTree neighborhoods
    # ********************
    #

    # If statement to skip this part if wanted
    if False:

        # Define the search parameters
        num_queries = 1000
        radius = 0.2

        # Array of different leaf sizes to test
        leaf_sizes = [30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140]

        # Pick random queries
        random_indices = np.random.choice(points.shape[0], num_queries, replace=False)
        queries = points[random_indices, :]

        # Dictionary to store timing results for each leaf size
        timing_results = {}

        # Test each leaf size
        for leaf_size in leaf_sizes:
            t0 = time.time()

            # Initialize KDTree with the current leaf size
            kdtree = KDTree(points, leaf_size=leaf_size)

            # Search spherical neighborhoods
            neighborhoods_indices = kdtree.query_radius(X=queries, r=radius)

            t1 = time.time()
            
            # Calculate and print the time taken
            time_taken = t1 - t0
            print('Leaf size {:d}:{:d} spherical neighborhoods computed in {:.3f} seconds'.format(leaf_size, num_queries, time_taken))
            
            # Store the timing results
            timing_results[leaf_size] = time_taken

        # Find the leaf size with the minimum time
        optimal_leaf_size = min(timing_results, key=timing_results.get)
        print('Optimal leaf size is {}'.format(optimal_leaf_size))

    if True:

        # Define the search parameters
        num_queries = 1000
        leaf_size = 110  # Use the optimal leaf size found previously

        radiuses = np.linspace(0.05, 0.5, 10)

        # Pick random queries
        random_indices = np.random.choice(points.shape[0], num_queries, replace=False)
        queries = points[random_indices, :]

        # Initialize KDTree with the optimal leaf size
        kdtree = KDTree(points, leaf_size=leaf_size)

        # Dictionary to store timing results for each radius
        timing_results = {}

        # Test each radius
        for radius in radiuses:
            t0 = time.time()

            # Search spherical neighborhoods
            neighborhoods_indices = kdtree.query_radius(X=queries, r=radius)

            t1 = time.time()
            
            # Calculate and store the time taken
            time_taken = t1 - t0
            timing_results[radius] = time_taken

        # Plot the timing results
        plt.plot(radiuses, list(timing_results.values()), marker='o')
        plt.xlabel('Radius')
        plt.ylabel('Time Taken (seconds)')
        plt.title('KDTree Spherical Neighborhood Search Timing by Radius')
        plt.show()

        # Time to compute all neighborhoods in the cloud for a radius of 20cm
        all_points_time = points.shape[0] * timing_results[0.2] / num_queries
        print('Estimated time to search 20cm neighborhoods for all points: {:.2f} seconds'.format(all_points_time))

        
        
        
        
        