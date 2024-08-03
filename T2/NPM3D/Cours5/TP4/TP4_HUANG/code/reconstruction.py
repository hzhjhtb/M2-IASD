#
#
#      0===========================================================0
#      |              TP4 Surface Reconstruction                   |
#      0===========================================================0
#
#
# ------------------------------------------------------------------------------------------
#
#      Jean-Emmanuel DESCHAUD - 06/02/2022
#


# Import numpy package and name it "np"
import numpy as np

# Import functions from scikit-learn
from sklearn.neighbors import KDTree

# Import functions to read and write ply files
from ply import write_ply, read_ply

# Import time package
import time

from skimage import measure

import trimesh


# Hoppe surface reconstruction
def compute_hoppe(points, normals, scalar_field, grid_resolution, min_grid, size_voxel):
    
    # Assuming scalar_field is a 3D grid of zeros (initialized outside this function)

    # Create a KD-tree for the input point cloud
    kd_tree = KDTree(points)

    # Compute the scalar field values for each voxel in the grid
    for i in range(grid_resolution):
        for j in range(grid_resolution):
            for k in range(grid_resolution):
                # Compute the world coordinates of the voxel
                voxel_center = min_grid + np.array([i, j, k]) * size_voxel
                # Reshape voxel_center to a 2D array with a single sample
                voxel_center_2d = voxel_center.reshape(1, -1)
                
                # Find the nearest point in the point cloud
                distance, nearest_idx = kd_tree.query(voxel_center_2d, k=1)
                p_i = points[nearest_idx[0][0]]
                n_i = normals[nearest_idx[0][0]]
                # Compute the Hoppe function value for the voxel center
                scalar_field[i, j, k] = np.dot(n_i, (voxel_center - p_i))


    

# IMLS surface reconstruction
def compute_imls(points, normals, scalar_field, grid_resolution, min_grid, size_voxel, knn):
    
    # Assuming scalar_field is a 3D grid of zeros (initialized outside this function)

    # Create a KD-tree for the input point cloud
    kd_tree = KDTree(points)

    h = 0.0008 # is a good trade-off for the bunny point cloud.

    # Compute the scalar field values for each voxel in the grid
    for i in range(grid_resolution):
        for j in range(grid_resolution):
            for k in range(grid_resolution):
                # Compute the world coordinates of the voxel
                voxel_center = min_grid + np.array([i, j, k]) * size_voxel
                # Reshape voxel_center to a 2D array with a single sample
                voxel_center_2d = voxel_center.reshape(1, -1)
                
                # Find the k nearest neighbors in the point cloud
                distances, indices = kd_tree.query(voxel_center_2d, k=knn)
                
                # Initialize numerator and denominator of the IMLS function
                numerator = 0.0
                denominator = 0.0
                
                # Loop over all nearest neighbors
                for idx, point_idx in enumerate(indices[0]):
                    p_i = points[point_idx]  # The ith point of the point cloud
                    n_i = normals[point_idx]  # The normal at the ith point
                    r = distances[0][idx]  # The distance from voxel to the ith point
                    
                    # Compute the weight for the ith point
                    theta = np.exp(-(r*r) / (h*h))
                    
                    # Update the sums for the numerator and denominator
                    numerator += theta * np.dot(n_i, (voxel_center - p_i))
                    denominator += theta
                
                # Compute the scalar field value at the voxel center using IMLS function
                scalar_field[i, j, k] = numerator / denominator if denominator != 0 else 0





if __name__ == '__main__':

    t0 = time.time()
    
    # Path of the file
    file_path = '../data/bunny_normals.ply'

    # Load point cloud
    data = read_ply(file_path)

    # Concatenate data
    points = np.vstack((data['x'], data['y'], data['z'])).T
    normals = np.vstack((data['nx'], data['ny'], data['nz'])).T

	# Compute the min and max of the data points
    min_grid = np.amin(points, axis=0)
    max_grid = np.amax(points, axis=0)
				
	# Increase the bounding box of data points by decreasing min_grid and inscreasing max_grid
    min_grid = min_grid - 0.10*(max_grid-min_grid)
    max_grid = max_grid + 0.10*(max_grid-min_grid)

	# grid_resolution is the number of voxels in the grid in x, y, z axis
    grid_resolution = 128 #16
    size_voxel = np.array([(max_grid[0]-min_grid[0])/(grid_resolution-1),(max_grid[1]-min_grid[1])/(grid_resolution-1),(max_grid[2]-min_grid[2])/(grid_resolution-1)])
	
	# Create a volume grid to compute the scalar field for surface reconstruction
    scalar_field = np.zeros((grid_resolution,grid_resolution,grid_resolution),dtype = np.float32)

	# Compute the scalar field in the grid
    #compute_hoppe(points,normals,scalar_field,grid_resolution,min_grid,size_voxel)
    compute_imls(points,normals,scalar_field,grid_resolution,min_grid,size_voxel,30)

	# Compute the mesh from the scalar field based on marching cubes algorithm
    verts, faces, normals_tri, values_tri = measure.marching_cubes(scalar_field, level=0.0, spacing=(size_voxel[0],size_voxel[1],size_voxel[2]))
	
    # Export the mesh in ply using trimesh lib
    mesh = trimesh.Trimesh(vertices = verts, faces = faces)
    #mesh.export(file_obj='../data/bunny_mesh_hoppe_16.ply', file_type='ply')
    mesh.export(file_obj='../data/bunny_mesh_hoppe_128.ply', file_type='ply')

	
    print("Total time for surface reconstruction : ", time.time()-t0)
	


