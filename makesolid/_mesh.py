# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np
import solid

class Mesh(object):
    def __init__(self):
        self.vertices = []
        self.vertex_index = {}
        self.triangles = []

    def add_vertex(self, v):
        v = tuple(v)
        if v not in self.vertex_index:
            self.vertex_index[v] = len(self.vertices)
            self.vertices.append(v)
        return self.vertex_index[v]

    def add_triangle(self, i,j,k):
        if (i<len(self.vertices)
                and j<len(self.vertices)
                and k<len(self.vertices)):
            if i==j or j==k or i==k:
                # Degenerate; suppress
                return
            self.triangles.append((i,j,k))
        else:
            raise ValueError("Triangle (%d,%d,%d) contains undefined vertex;"
                             " only %d available" % (i,j,k,len(self.vertices)))

    def as_scad(self):
        return solid.polyhedron(points=self.vertices, faces=self.triangles)

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

    for i in range(n_u-1):
        for j in range(n_v-1):
            M.add_triangle(ix[i,j], ix[i,j+1], ix[i+1,j])
            M.add_triangle(ix[i+1,j], ix[i,j+1], ix[i+1,j+1])

    return M


def helix_extrude(points, turns, vertical_motion_per_turn,
                  steps_per_turn=16):
    """Spiral the shape specified by points around the z axis"""
    if turns<=0:
        raise ValueError("Number of turns must be positive but got %s" % turns)
    n = max(int(np.ceil(turns*steps_per_turn)+1),2)
    points = np.asarray(points, dtype=float)
    nps, d = points.shape
    if d!=3:
        raise ValueError("Only three-dimensional points currently supported; "
                             "got shape %s" % (points.shape,))
    ts = np.linspace(0,turns,n)
    xyz_base = points.T[:,:,None]+0*ts[None,None,:]
    xyz_base[2,:,:] += (ts*vertical_motion_per_turn)[None,:]
    for i in range(n):
        th = 2*np.pi*ts[i]
        rm = np.array([[np.cos(th), np.sin(th)],
                       [-np.sin(th), np.cos(th)]])
        xyz_base[:2,:,i] = np.dot(rm, xyz_base[:2,:,i])
    # add end caps
    xyz = np.zeros((3,nps+1,n+2))
    xyz[:,:-1,1:-1] = xyz_base
    xyz[:,:-1,0] = np.mean(xyz[:,:-1,1],axis=1)[:,None]
    xyz[:,:-1,-1] = np.mean(xyz[:,:-1,-2],axis=1)[:,None]
    # add wrap
    xyz[:,-1,:] = xyz[:,0,:]

    return uv_surface(xyz).as_scad()
