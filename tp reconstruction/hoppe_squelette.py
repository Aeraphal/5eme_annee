import numpy as np
import mcubes
from scipy import spatial

p = np.loadtxt('armadillo_sub.xyz')
points = p[:, :3]
normals = p[:, 3:6] 
kdtree = spatial.KDTree(points)

X, Y, Z = np.mgrid[np.min(p[:,0]):np.max(p[:,0]),np.min(p[:,1]):np.max(p[:,1]),np.min(p[:,2]):np.max(p[:,2])]
u = np.zeros_like(X)

for i in range(len(X)):
    for j in range(len(Y)):
        for k in range(len(Z)):
            query = [X[i,j,k],Y[i,j,k],Z[i,j,k]]
            dist, idx = kdtree.query(query)
            
            sign = np.sign(np.dot(query - points[idx], normals[idx]))
            u[i,j,k] = sign * dist

vertices, triangles = mcubes.marching_cubes(u,0)

mcubes.export_obj(vertices, triangles, 'result.obj')
