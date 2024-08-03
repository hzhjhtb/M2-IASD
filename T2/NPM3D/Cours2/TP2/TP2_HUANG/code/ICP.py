#
#
#      0===================================0
#      |    TP2 Iterative Closest Point    |
#      0===================================0
#
#
#------------------------------------------------------------------------------------------
#
#      Script of the practical session
#
#------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 17/01/2018
#


#------------------------------------------------------------------------------------------
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
from visu import show_ICP

import sys


#------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define usefull functions to be used in the main
#


def best_rigid_transform(data, ref):
    '''
    Computes the least-squares best-fit transform that maps corresponding points data to ref.
    Inputs :
        data = (d x N) matrix where "N" is the number of points and "d" the dimension
         ref = (d x N) matrix where "N" is the number of points and "d" the dimension
    Returns :
           R = (d x d) rotation matrix
           T = (d x 1) translation vector
           Such that R * data + T is aligned on ref
    '''
    # YOUR CODE
    R = np.eye(data.shape[0])
    T = np.zeros((data.shape[0],1))

    # Calculate barycenters
    Pm = np.mean(ref, axis=1).reshape(-1,1)
    Pm_prime = np.mean(data, axis=1).reshape(-1,1)

    # Compute centered clouds Q and Q_prime
    Q = ref - Pm
    Q_prime = data - Pm_prime

    # Get matrix H = Q_prime * Q^T
    H = Q_prime @ Q.T

    # Find the singular value decomposition USV^T of H
    U, S, Vt = np.linalg.svd(H)

    # Compute R = V * U^T
    R = Vt.T @ U.T
    # Ensure R is a rotation matrix (orthogonal with determinant 1)
    if np.linalg.det(R) < 0:
        # Multiply last column of U by -1
        U[:, -1] *= -1
        # Re-compute R = V * U^T
        R = Vt.T @ U.T
    
    # Compute T = Pm - R * Pm_prime
    T = Pm - R @ Pm_prime

    return R, T



def icp_point_to_point(data, ref, max_iter, RMS_threshold):
    '''
    Iterative closest point algorithm with a point to point strategy.
    Inputs :
        data = (d x N_data) matrix where "N_data" is the number of points and "d" the dimension
        ref = (d x N_ref) matrix where "N_ref" is the number of points and "d" the dimension
        max_iter = stop condition on the number of iterations
        RMS_threshold = stop condition on the distance
    Returns :
        data_aligned = data aligned on reference cloud
        R_list = list of the (d x d) rotation matrices found at each iteration
        T_list = list of the (d x 1) translation vectors found at each iteration
        neighbors_list = At each iteration, you search the nearest neighbors of each data point in
        the ref cloud and this obtain a (1 x N_data) array of indices. This is the list of those
        arrays at each iteration
        RMS_list = list of the RMS found at each iteration
           
    '''

    # Variable for aligned data
    data_aligned = np.copy(data)

    # Initiate lists
    R_list = []
    T_list = []
    neighbors_list = []
    RMS_list = []

    # YOUR CODE

    # Build KDTree on reference cloud
    tree = KDTree(ref.T)


    # Iterate
    for i in range(max_iter):

        # Find closest neighbors
        _, neighbors = tree.query(data_aligned.T, k=1)
        # neighours should be a (1 x N_data) array of indices
        neighbors = neighbors.reshape(1,-1)
        neighbors_list.append(neighbors)
        
        # Compute best transformation
        R, T = best_rigid_transform(data_aligned, ref[:, neighbors].squeeze())

        # Apply transformation
        data_aligned = R @ data_aligned + T

        # Save transformation
        R_list.append(R)
        T_list.append(T)

        # Compute RMS on matched points
        distances2 = np.sum(np.power(data_aligned - ref[:, neighbors].squeeze(), 2), axis=0)
        RMS = np.sqrt(np.mean(distances2))
        RMS_list.append(RMS)

        # Check stop condition
        if RMS < RMS_threshold:
            break

    return data_aligned, R_list, T_list, neighbors_list, RMS_list



def icp_point_to_point_fast(data, ref, max_iter, RMS_threshold, sampling_limit):
    '''
    Iterative closest point algorithm with a point to point strategy using random sampling.
    Inputs :
        data = (d x N_data) matrix where "N_data" is the number of points and "d" the dimension
        ref = (d x N_ref) matrix where "N_ref" is the number of points and "d" the dimension
        max_iter = stop condition on the number of iterations
        RMS_threshold = stop condition on the distance
        sampling_limit = number of points to randomly sample at each iteration for neighbor search
    Returns :
        data_aligned = data aligned on the reference cloud
        R_list = list of the (d x d) rotation matrices found at each iteration
        T_list = list of the (d x 1) translation vectors found at each iteration
        neighbors_list = At each iteration, you search the nearest neighbors of each data point in
        the ref cloud and this obtains a (1 x N_data) array of indices. This is the list of those
        arrays at each iteration
        RMS_list = list of the RMS found at each iteration
    '''

    # Variable for aligned data
    data_aligned = np.copy(data)

    # Initiate lists
    R_list = []
    T_list = []
    neighbors_list = []
    RMS_list = []

    # Build KDTree on reference cloud
    tree = KDTree(ref.T)

    # Iterate
    for i in range(max_iter):

        # Randomly sample data points for nearest neighbor search
        random_indices = np.random.choice(data_aligned.shape[1], size=sampling_limit, replace=False)
        sampled_data = data_aligned[:, random_indices]

        # Find closest neighbors
        _, neighbors = tree.query(sampled_data.T, k=1)
        # neighours should be a (1 x N_data) array of indices
        neighbors = neighbors.reshape(1,-1)
        neighbors_list.append(neighbors)

        # Compute best transformation
        R, T = best_rigid_transform(sampled_data, ref[:, neighbors].squeeze())

        # Apply transformation to all data points
        data_aligned = R @ data_aligned + T

        # Save transformation
        R_list.append(R)
        T_list.append(T)

        # Compute RMS on all data points
        distances, _ = tree.query(data_aligned.T, k=1)
        RMS = np.sqrt(np.mean(distances ** 2))
        RMS_list.append(RMS)

        # Check stop condition
        if RMS < RMS_threshold:
            break

    return data_aligned, R_list, T_list, neighbors_list, RMS_list




#------------------------------------------------------------------------------------------
#
#           Main
#       \**********/
#
#
#   Here you can define the instructions that are called when you execute this file
#


if __name__ == '__main__':
   
    # Transformation estimation
    # *************************
    #

    # If statement to skip this part if wanted
    if False:

        # Cloud paths
        bunny_o_path = '../data/bunny_original.ply'
        bunny_r_path = '../data/bunny_returned.ply'

		# Load clouds
        bunny_o_ply = read_ply(bunny_o_path)
        bunny_r_ply = read_ply(bunny_r_path)
        bunny_o = np.vstack((bunny_o_ply['x'], bunny_o_ply['y'], bunny_o_ply['z']))
        bunny_r = np.vstack((bunny_r_ply['x'], bunny_r_ply['y'], bunny_r_ply['z']))

        # Find the best transformation
        R, T = best_rigid_transform(bunny_r, bunny_o)

        # Apply the tranformation
        bunny_r_opt = R.dot(bunny_r) + T

        # Save cloud
        write_ply('../data/bunny_r_opt.ply', [bunny_r_opt.T], ['x', 'y', 'z'])

        # Compute RMS
        distances2_before = np.sum(np.power(bunny_r - bunny_o, 2), axis=0)
        RMS_before = np.sqrt(np.mean(distances2_before))
        distances2_after = np.sum(np.power(bunny_r_opt - bunny_o, 2), axis=0)
        RMS_after = np.sqrt(np.mean(distances2_after))

        print('Average RMS between points :')
        print('Before = {:.3f}'.format(RMS_before))
        print(' After = {:.3f}'.format(RMS_after))


   

    # Test ICP and visualize
    # **********************
    #

    # If statement to skip this part if wanted
    if False:

        # Cloud paths
        ref2D_path = '../data/ref2D.ply'
        data2D_path = '../data/data2D.ply'
        
        # Load clouds
        ref2D_ply = read_ply(ref2D_path)
        data2D_ply = read_ply(data2D_path)
        ref2D = np.vstack((ref2D_ply['x'], ref2D_ply['y']))
        data2D = np.vstack((data2D_ply['x'], data2D_ply['y']))        

        # Apply ICP
        data2D_opt, R_list, T_list, neighbors_list, RMS_list = icp_point_to_point(data2D, ref2D, 10, 1e-4)
        
        # Show ICP
        show_ICP(data2D, ref2D, R_list, T_list, neighbors_list)
        
        # Plot RMS
        plt.plot(RMS_list)
        plt.show()
        

    # If statement to skip this part if wanted
    if False:

        # Cloud paths
        bunny_o_path = '../data/bunny_original.ply'
        bunny_p_path = '../data/bunny_perturbed.ply'
        
        # Load clouds
        bunny_o_ply = read_ply(bunny_o_path)
        bunny_p_ply = read_ply(bunny_p_path)
        bunny_o = np.vstack((bunny_o_ply['x'], bunny_o_ply['y'], bunny_o_ply['z']))
        bunny_p = np.vstack((bunny_p_ply['x'], bunny_p_ply['y'], bunny_p_ply['z']))

        # Apply ICP
        bunny_p_opt, R_list, T_list, neighbors_list, RMS_list = icp_point_to_point(bunny_p, bunny_o, 25, 1e-4)
        
        # Show ICP
        show_ICP(bunny_p, bunny_o, R_list, T_list, neighbors_list)
        
        # Plot RMS
        plt.plot(RMS_list)
        plt.show()


    # If statement to skip this part if wanted
    if True:
        # Cloud paths
        notre_dame_path = '../data/Notre_Dame_Des_Champs_1.ply'
        notre_dame_p_path = '../data/Notre_Dame_Des_Champs_2.ply'

        # Load clouds
        notre_dame_ply = read_ply(notre_dame_path)
        notre_dame_p_ply = read_ply(notre_dame_p_path)
        notre_dame = np.vstack((notre_dame_ply['x'], notre_dame_ply['y'], notre_dame_ply['z']))
        notre_dame_p = np.vstack((notre_dame_p_ply['x'], notre_dame_p_ply['y'], notre_dame_p_ply['z']))

        # Apply ICP with 1000 points at each iteration
        _, _, _, _, RMS_list_1000 = icp_point_to_point_fast(notre_dame_p, notre_dame, 25, 1e-4, 1000)
        
        # Apply ICP with 10000 points at each iteration
        _, _, _, _, RMS_list_10000 = icp_point_to_point_fast(notre_dame_p, notre_dame, 25, 1e-4, 10000)

        # Plot RMS curves
        plt.plot(RMS_list_1000, label='1000 points per iteration')
        plt.plot(RMS_list_10000, label='10000 points per iteration')
        plt.legend()
        plt.title('RMS Convergence for Different Sampling Limits')
        plt.show()