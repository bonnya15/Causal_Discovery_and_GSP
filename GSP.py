# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 12:41:10 2023

@author: shiuli Subhra Ghosh
"""
import numpy as np
from pygsp import graphs, filters
import matplotlib.pyplot as plt
from numpy import linalg as LA
from scipy import sparse

W = np.array([[0,5,0,5,0,0],[5,0,4,10,3.33,0],[0,4,0,0,3.84,10],[5,10,0,0,2.5,0],[0,3.33,3.84,2.5,0,3.33],[0,0,10,0,3.33,0]])
W = np.array([[0,0,0,5,0,0],[0,0,4,10,3.33,0],[0,4,0,0,3.84,10],[5,10,0,0,2.5,0],[0,3.33,3.84,2.5,0,3.33],[0,0,10,0,3.33,0]])
W = np.array([[0,5,0,5,0,0],[5,0,4,10,3.33,0],[0,4,0,0,3.84,10],[5,10,0,0,2.5,0],[0,3.33,3.84,2.5,0,0],[0,0,10,0,0,0]])
G = graphs.Graph(W)
L_shape = G.L.shape
G.compute_laplacian('combinatorial')
G.compute_fourier_basis()
G.set_coordinates()
fig, axes = plt.subplots(1, 6, figsize=(10, 3))
for i, ax in enumerate(axes):
    G.plot_signal(G.U[:, i], vertex_size=50, ax=ax)
    _ = ax.set_title('Eigenvector {}'.format(i+1))
    ax.set_axis_off()


D = sparse.diags(np.ravel(W.sum(1)), 0)
L = (D - G.W).toarray()
w1, v1 = LA.eig(L)
d = LA.det(L)
