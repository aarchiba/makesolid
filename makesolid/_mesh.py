# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np

class Mesh(object):
    def __init__(self):
        self.vertices = []
        self.vertex_index = {}
        self.triangles = []

    def add_vertex(v):
        if v not in self.vertex_index:
            self.vertex_index[v] = len(self.vertices)
            self.vertices.append(v)
        return self.vertex_index[v]

    def add_triangle(i,j,k):
        if i<len(self.vertices) and j<len(self.vertices) and k<len(vertices):
            if i==j or j==k or i==k:
                # Degenerate; suppress
                return
            self.triangles.append((i,j,k))
        else:
            raise ValueError("Triangle (%d,%d,%d) contains undefined vertex;"
                             " only %d available" % (i,j,k,len(self.vertices)))


def uv_surface(xyz):
    xyz = np.asarray(xyz)
    t, n_u, n_v = xyz.shape
    if t!=3:
        raise ValueError("xyz must have shape (3,n_u,n_v) but is (%d,%d,%d)"
                         % xyz.shape)

    M = Mesh()
    ix = np.zeros((n_u,n_v),dtype=int)
    for i in range(n_u):
        for j in range(n_v):
            ix[i,j] = M.add_vertex(xyz[:,i,j])

    for i in range(n_u):
        for j in range(n_v):
            M.add_triangle(ix[i,j], ix[i,j+1], ix[i+1,j])
            M.add_triangle(ix[i+1,j], ix[i,j+1], ix[i+1,j+1])

    return M

