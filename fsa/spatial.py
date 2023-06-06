"""
This file contains functions that assist in calculating spatial properties. For
example, they assist in spatial orientation and alignment of the structures.
"""

# Numerical
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares

# Biological
from Bio import PDB

# Internal
from . import load 

def align_structures(rel_pps, def_pps, al_chain_ids=None, common_res=None,
                     rel_dict=None, def_dict=None):
    """
    Perform spatial alignment of deformed polypeptides to reference polypep-
    tides for the given chains. If no chain id is given, use all structure
    """    
    if not al_chain_ids:
        al_chain_ids = [load.get_chain_name(chain) for chain in rel_pps]
    
    # Load relaxed/deformed chains to align
    rel_al_ch = load.choose_chains(rel_pps, al_chain_ids)
    def_al_ch = load.choose_chains(def_pps, al_chain_ids)

    # Load relaxed/deformed common atoms to align
    rel_al_atm, _ = load.atoms(rel_al_ch, common_res, rel_dict)
    def_al_atm, _ = load.atoms(def_al_ch, common_res, def_dict)

    # Load all deformed atoms
    def_all_atom, _ = load.atoms(def_pps, common_res, def_dict)

    # Load alignment tool, align selected atoms, and apply to all
    super_imposer = PDB.Superimposer()
    super_imposer.set_atoms(rel_al_atm, def_al_atm)
    super_imposer.apply(def_all_atom)

    return  # need to specify which atoms, otherwise can not perform alignment!
    
    
def cylinder_axis(xyz, phi_ini=.1):

    """
    Fit the provided coordinates to a cylinder of variable center, orientation
    and radius (5 parameters). Returns orientation axis of the best fit
    cylinder, as well as the whole fit results. We also provide an initial
    angle bias for the fit, useful to set at 0.1 or -3 to bias fully inverted
    Z-axis
    """
    # Initialize fit parameters
    xc, yc = np.mean(xyz[:, 0]), np.mean(xyz[:, 1])  # cylinder center
    theta, phi = 0., phi_ini  # angles about x and y axis, phi>0 biases Z>0
    r = np.std(xyz)  # radius

    # Least square fit using cylinder distance
    result = least_squares(cylinder_error,
                           [xc, yc, theta, phi, r],
                           args=(xyz[:, 0], xyz[:, 1], xyz[:, 2]),
                           max_nfev=10000)
    [xc, yc, theta, phi, r] = result.x

    # Calculate new axis
    r = R.from_euler('xy', [theta, phi])
    axis = r.apply([0, 0, 1])

    return axis, result


def cylinder_error(p, x, y, z):
    """
    Error from the data to a cylinderical sheet centerd in xc, yc = p[0], p[1]
    of radius r = p[4], and with x/y orientation theta = p[2] and phi = p[3].
    """
    # Create xyz array
    xyz = np.transpose(np.array([x, y, z]))

    # Position vectors to cylinder center
    dxyz = xyz - np.array([p[0], p[1], 0.])

    # Cylinder axis
    r = R.from_euler('xy', [p[2], p[3]])
    cyl_axis = r.apply([0, 0, 1])

    # Distance to axis
    dxyz_projected = dxyz * cyl_axis
    axis_distance = np.sqrt(np.sum(dxyz**2, 1) - np.sum(dxyz_projected, 1)**2)
    fit_error = np.sum((axis_distance - p[4])**2)

    return fit_error


def center_and_align(pps, cyl_chain_ids, phi_ini=0.1, Nrep=2):
    """
    This function aligns the given polypeptides to the  z axis by using the
    chains in cyl_chain_ids as a cylinder that will be centerd and aligned. We
    repeat the mean subtraction Nrep times, as after a finite rotation the
    center gets displaced. We provide as parameter an initial tilt angle
    phi_ini for the fit.
    """

    # Load coordinates of cylinder chains
    cyl_pps = load.choose_chains(pps, cyl_chain_ids)
    cyl_xyz, _ = load.coordinates(cyl_pps)

    # Calculate displacement and axis
    displacement = np.mean(cyl_xyz, 0)
    axis, _ = cylinder_axis(cyl_xyz - displacement, phi_ini)
    rigid_body(pps, displacement, axis)

    # Center again
    # cyl_pps = load.choose_chains(pps, cyl_chain_ids)
    cyl_xyz, _ = load.coordinates(cyl_pps)
    displacement = np.mean(cyl_xyz, 0)
    rigid_body(pps, displacement)

    return axis


def rigid_body(pps, displacement, old_axis=[0, 0, 1], new_axis=[0, 0, 1]):
    """
    Move polypeptides by displacement and rotate them so that the
    input axis matches the Z axis.
    """
    # Define rotation matrix
    rotation = PDB.rotmat(PDB.Vector(new_axis),
                          PDB.Vector(old_axis))

    # Perform rotation of residues
    for chains in pps:
        for pp in chains:
            for res in pp:
                if load.test_residue(res):
                    res.transform(rotation,
                                  -displacement)

    return


def coordinates_to_polar(coordinates, inv=False):
    """
    Transform the given coordinates from cartesian to polar, or from polar to
    cartesian (inverse) if inv=True
    """
    if not inv:
        xyz = coordinates
        r = np.sqrt(xyz[:, 0]**2 + xyz[:, 1]**2)
        t = np.arctan2(xyz[:, 1], xyz[:, 0])
        z = xyz[:, 2]
        rtz = np.transpose(np.array([r, t, z]))
        return rtz
    else:
        rtz = coordinates
        x = rtz[:, 0] * np.cos(rtz[:, 1])
        y = rtz[:, 0] * np.sin(rtz[:, 1])
        z = rtz[:, 2]
        xyz = np.transpose(np.array([x, y, z]))
        return xyz


def tensor_to_polar(xyz, T, inv=False):
    """
    Transform the given tensor to polar coordinates using the given
    (cartesian) coordinates.
    """
    # Obtain polar angle
    rtz = coordinates_to_polar(xyz)
    theta = rtz[:, 1]

    # Define rotation matrix and its transpose
    R = np.array([np.array([np.cos(theta),
                  np.sin(theta),
                  np.zeros_like(theta)]),
                  np.array([-np.sin(theta),
                  np.cos(theta),
                  np.zeros_like(theta)]),
                  np.array([np.zeros_like(theta),
                  np.zeros_like(theta),
                  np.ones_like(theta)])])
    R = np.transpose(R, (2, 0, 1))
    Rt = np.transpose(R, (0, 2, 1))

    # Perform dot product
    R_x_T = np.einsum('abc,acf->abf', R, T)
    T_cyl = np.einsum('abc,acf->abf', R_x_T, Rt)

    return T_cyl


def vector_to_polar(xyz, v, inv=False):
    """
    Transform the given vector to polar coordinates using the given (cartesian)
    coordinates.
    """
    # Obtain polar angle
    rtz = coordinates_to_polar(xyz)
    theta = rtz[:, 1]

    # Define rotation matrix and its transpose
    R = np.array([np.array([np.cos(theta),
                  np.sin(theta),
                  np.zeros_like(theta)]),
                  np.array([-np.sin(theta),
                  np.cos(theta),
                  np.zeros_like(theta)]),
                  np.array([np.zeros_like(theta),
                  np.zeros_like(theta),
                  np.ones_like(theta)])])
    R = np.transpose(R, (2, 0, 1))

    # Perform dot product
    R_x_v = np.einsum('aij,aj->ai', R, v)

    return R_x_v


def coordinates_to_spherical(coordinates, inv=False):
    """
    Transform the given coordinates from cartesian to spherical, or from spherical to
    cartesian (inverse) if inv=True
    """
    if not inv:
        xyz = coordinates
        xy2 = xyz[:, 0]**2 + xyz[:, 1]**2

        r = np.sqrt(xy2 + xyz[:, 2]**2)
        t = np.arctan2(np.sqrt(xy2), xyz[:, 2])
        p = np.arctan2(xyz[:, 1], xyz[:, 0])
        rtp = np.transpose(np.array([r, t, p]))
        return rtp
    else:
        rtp = coordinates
        x = rtp[:, 0] * np.sin(rtp[:, 1]) * np.cos(rtp[:, 2])
        y = rtp[:, 0] * np.sin(rtp[:, 1]) * np.sin(rtp[:, 2])
        z = rtp[:, 0] * np.cos(rtp[:, 1])
        xyz = np.transpose(np.array([x, y, z]))
        return xyz
