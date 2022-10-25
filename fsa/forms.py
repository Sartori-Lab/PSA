"""
This file contains functions that create different elemantal deformations. The
goal is to compare them with analytical results
"""

# Numerical
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares

# Biological
from Bio import PDB

# Internal
from . import load 


def noise(xyz, sigma):
    """
    asd
    """

    # Create noise data structure
    xyz_n = np.zeros_like(xyz)

    # Add normal noise
    xyz_n[:, :] = xyz[:, :] + np.random.normal(0, sigma, (len(xyz), 3))

    return xyz_n


def generate_rod(r0=20., z0=180., theta0=2*np.pi, ds=4.):
    """
    Generate a cylindrical rod with the provided dimensions (Radius height, in
    A), and the provided spacing among atoms (ds)
    """

    # Then, assign coordiante values
    rtz = []
    Nr0, Nz0 = int(r0 / ds), int(z0 / ds)

    for z in np.linspace(0, z0, Nz0):
        for r in np.linspace(0, r0, Nr0):
            Ntheta0 = int(2 * np.pi * r / ds)
            for theta in np.linspace(0, theta0, Ntheta0):
                rtz.append([r, theta, z])

    # Transform to cartesian
    rtz = np.array(rtz)
    xyz = coordinates_to_polar(rtz, inv=True)

    return xyz


def deform(xyz, method='twist', d=0.01):
    """
    Take as input a set of coordinates, a method, and a deformation parameter;
    and select the corresponding deformation function
    """
    cases = {'twist': twist,
             'spin': spin,
             'radial': radial_extension}

    return cases[method](xyz, d)


def twist(xyz, d=0.01):
    """
    Apply a twist transformation to the given (euclidean) coordinates
    """
    # Transform reference coordinates to cylindrical
    rtz = coordinates_to_polar(xyz)
    rtz_d = []

    # Apply twist transformation
    for rtz_a in rtz:
        r_d = rtz_a[0]
        t_d = rtz_a[1] + d * rtz_a[2]
        z_d = rtz_a[2]
        rtz_d.append([r_d, t_d, z_d])

    # Transform deformed coordinates to euclidean
    rtz_d = np.array(rtz_d)
    xyz_d = coordinates_to_polar(rtz_d, inv=True)

    return xyz_d


def radial_extension(xyz, d=.5):
    """
    Apply a unuform radial extension to the given (euclidean) coordinates
    """
    # Transform reference coordinates to cylindrical
    rtz = coordinates_to_polar(xyz)
    rtz_d = []

    # Apply radaial extensiomn transformation
    for rtz_a in rtz:
        r_d = rtz_a[0] + d * rtz_a[0]
        t_d = rtz_a[1]
        z_d = rtz_a[2]
        rtz_d.append([r_d, t_d, z_d])

    # Transform deformed coordinates to euclidean
    rtz_d = np.array(rtz_d)
    xyz_d = coordinates_to_polar(rtz_d, inv=True)

    return xyz_d


def spin(xyz, d):
    """
    Deform angles in the radial direction
    """
    # Transform reference coordinates cylindrical
    rtz = coordinates_to_polar(xyz)
    rtz_d = []

    # Apply spin
    for rtz_a in rtz:
        r_d = rtz_a[0]
        t_d = rtz_a[1] + d * rtz_a[0]
        z_d = rtz_a[2]
        rtz_d.append([r_d, t_d, z_d])

    # Transform deformed coordinates to euclidean
    rtz_d = np.array(rtz_d)
    xyz_d = coordinates_to_polar(rtz_d, inv=True)

    return xyz_d
