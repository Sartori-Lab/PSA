"""
This file contains functions that assist in calculating elastic properties. For
example, it calculates the deformation gradients.
"""

import numpy as np
from numba import jit
from scipy.spatial import distance
import scipy.linalg as spla
import numpy.linalg as npla


def compute_weights_fast(xyz_list, method="intersect", parameters=[6.0, 8.0]):
    """
    Calls one of the four methods to compute weights using numba speed-up
    procedure.
    """
    cases = {
        "intersect": intersect_weights_fast,
        "union": union_weights_fast,
        "linear": linear_weights_fast,
        "minimal": minimal_weights_fast,
    }

    return cases[method](xyz_list, parameters)


@jit(nopython=True)
def binary_weights_fast(xyz_list, parameters):
    """
    Calculate binary neighbourhoods considering atoms within a given
    radius shared among the list of coordinates. 
    """
    r = float(parameters[0])

    all_sets = []

    for xyz in xyz_list:
        current_set = set()

        for i in range(len(xyz)):
            for j in range(i + 1, len(xyz)):
                # Calculates pairwise distance between atoms
                distance = 0
                for k in range(3):
                    distance += np.power(xyz[i, k] - xyz[j, k], 2)
                distance = np.sqrt(distance)

                # Keeps all pairs within radius
                if distance < r:
                    current_set.add((i, j))
                    current_set.add((j, i))

        all_sets.append(current_set)

    return all_sets


@jit(nopython=True)
def intersect_weights_fast(xyz_list, parameters=[8.0]):
    """
    Calculates the intersection of indexes given by the binary assignment approach.
    Returns a dictionary with non-zero weights as values and indexes of atoms as 
    keys.
    """

    weights = dict()

    all_sets = binary_weights_fast(xyz_list, parameters)
    final_set = all_sets[0]
    for i in range(1, len(xyz_list)):
        final_set.intersection_update(all_sets[i])

    for pair in final_set:
        weights[pair] = 1

    return weights


@jit(nopython=True)
def union_weights_fast(xyz_list, parameters=[8.0]):
    """
    Calculates the union of indexes given by the binary assignment approach.
    Returns a dictionary with non-zero weights as values and indexes of atoms as 
    keys.
    """
    weights = dict()

    all_sets = binary_weights_fast(xyz_list, parameters)
    final_set = all_sets[0]
    for i in range(1, len(xyz_list)):
        final_set.update(all_sets[i])

    for pair in final_set:
        weights[pair] = 1

    return weights


@jit(nopython=True)
def linear_weights_fast(xyz_list, parameters=(6.0, 8.0)):
    """
    Calculate the average weights for each coordinate set using a linear decay 
    function (two radii are provided). The function returns the weights as a 
    dictionary with the pairs of position indexes as keys and the weights as 
    values. Absent pairs of coordinates have zero weight.
    """
    r1, r2 = parameters[0], parameters[1]

    weights = dict()

    for xyz in xyz_list:
        for i in range(len(xyz)):
            for j in range(i + 1, len(xyz)):
                # Calculates pairwise distance between atoms
                distance = 0
                for k in range(3):
                    distance += np.power(xyz[i, k] - xyz[j, k], 2)
                distance = np.sqrt(distance)

                # Keeps all pairs within radius 2
                if distance < r2:
                    w = np.minimum(1 - (distance - r1) / (r2 - r1), 1.0)
                    w = w / len(xyz_list)

                    if (i, j) not in weights:
                        weights[(i, j)] = w
                        weights[(j, i)] = w
                    else:
                        weights[(i, j)] += w
                        weights[(j, i)] += w

    return weights


@jit(nopython=True)
def minimal_weights_fast(xyz_list, parameters=(3)):
    """
    Weight matrix is the minimal required to solve the equations of the defor-
    mation gradients. That is, the 'n' nearest neighbors. Only uses the first
    structure given.
    """
    n = int(parameters[0])
    xyz = xyz_list[0]

    weights = dict()

    for i in range(len(xyz)):
        distance = np.zeros(len(xyz))
        for j in range(len(xyz)):
            # Calculates pairwise distance between atoms
            for k in range(3):
                distance[j] += np.power(xyz[i, k] - xyz[j, k], 2)

        distance[distance == 0] = np.inf
        for k in range(n):
            weights[(i, np.argmin(distance))] = 1
            distance[distance == min(distance)] = np.inf

    return weights


@jit(nopython=True)
def intermediate_matrixes(weights, xyz_rel, xyz_def):
    """
    Calculate the matrixes D and A, intermediate steps in the calculation of F in 
    [Gullet et al,] (Eq. X), or Eq. 17 in [Zimmerman et al, 2009].
    """
    num_res = xyz_rel.shape[0]
    D = np.zeros((num_res, 3, 3))
    A = np.zeros((num_res, 3, 3))

    dX = np.zeros((3, 3), dtype=np.float64)
    dx = np.zeros((3, 3), dtype=np.float64)

    for pos in weights:
        i, j = pos
        dX = xyz_rel[j, :] - xyz_rel[i, :]
        dx = xyz_def[j, :] - xyz_def[i, :]

        # Generate intermediate matrices
        D[i, :, :] += np.dot(dX.reshape((-1, 1)), dX.reshape((1, -1))) * weights[pos]
        A[i, :, :] += np.dot(dx.reshape((-1, 1)), dX.reshape((1, -1))) * weights[pos]

    return D, A


def deformation_gradient_fast(weights, xyz_rel, xyz_def):
    """
    Calculate the deformation gradient tensor for each (F=dx/dX) using the
    approach from [Gullet et al,] (Eq. X), or Eq. 17 in [Zimmerman et al, 2009]
    In solving, note that D.T*U=A.T -> U.T*D=A -> F=U.T.
    
    This implementation requires weights in the form of a dictionary, which
    is also generated by the fast versions of weight calculation.
    """
    D, A = intermediate_matrixes(weights, xyz_rel, xyz_def)

    num_res = D.shape[0]
    F = np.zeros((num_res, 3, 3))

    for i in range(num_res):
        F[i, :, :] = spla.solve(D[i, :, :].T, A[i, :, :].T).T

    return F


def rotations(F):
    """
    Calculate the rotation angles and axis from the deformation gradients. Note
    that, due to large deformations, we can have det F<0. This implies an im-
    proper rotation matrix, http://scipp.ucsc.edu/~haber/ph116A/rotation_11.pdf
    """
    rotation_angle = []
    rotation_axis = []

    for alpha in range(len(F)):
        # Polar decomposition and diagonalize
        Falpha = F[alpha]
        u, p = spla.polar(Falpha)

        # Obtain axis and angle, being mindful of whether u is improper
        if spla.det(u) > 0:
            axis = np.array([u[2, 1] - u[1, 2], u[2, 0] - u[0, 2], u[0, 1] - u[1, 0]])
            axis /= npla.norm(axis)
            angle = np.arccos((np.trace(u) - 1) / 2.0)
        else:
            axis = -np.array([u[2, 1] - u[1, 2], u[2, 0] - u[0, 2], u[0, 1] - u[1, 0]])
            axis /= npla.norm(axis)
            angle = np.arccos((np.trace(u) + 1) / 2.0)

        # Store
        rotation_axis.append(axis)
        rotation_angle.append(angle)

    return rotation_angle, rotation_axis


def euler_strain(F):
    """
    Calculate linear and non-linear euler strain from deformation gradients.
    """
    strain_linear = []
    strain_non_linear = []

    for alpha in range(len(F)):
        # Calculate displacement gradient
        Falpha = F[alpha]
        Finv = spla.inv(Falpha)
        dU = np.eye(3) - Finv

        # Calculate linear and non-linear strain
        eps_lin = (dU + dU.T) / 2.0
        eps_n_lin = (dU + dU.T - np.dot(dU.T, dU)) / 2.0

        # Append results
        strain_linear.append(eps_lin)
        strain_non_linear.append(eps_n_lin)

    return strain_linear, strain_non_linear


def lagrange_strain(F):
    """
    Calculate linear and non-linear lagrange strain from deformation gradients.
    """
    strain_linear = []
    strain_non_linear = []

    for alpha in range(len(F)):
        # Calculate displacement gradient
        Falpha = F[alpha]
        du = Falpha - np.eye(3)

        # Calculate linear and non-linear strain
        gam_lin = (du + du.T) / 2.0
        gam_n_lin = (du + du.T + np.dot(du.T, du)) / 2.0

        # Append results
        strain_linear.append(gam_lin)
        strain_non_linear.append(gam_n_lin)

    return strain_linear, strain_non_linear


def invariants_from_g(lagrange_strain):
    """
    Calculate the three usual invariants (I1, I2, I3). These are functions of
    the right/left Cauchy tensors. The right one is related to the lagrangian
    strain as C = 2 gam + I.
    """
    # Right Cauchy tensor
    C = 2 * np.array(lagrange_strain) + np.eye(3)

    # First invariant
    I1 = np.trace(C, axis1=1, axis2=2)

    # Second invariant
    I2_list = []
    for Ca in C:
        I2_list.append(-(np.trace(Ca ** 2) - np.trace(Ca) ** 2) / 2.0)
    I2 = np.array(I2_list)

    # Third invariant
    I3_list = []
    for Ca in C:
        I3_list.append(np.linalg.det(Ca))
    I3 = np.array(I3_list)

    return I1, I2, I3


def principal_stretches_from_g(lagrange_strain):
    """
    Calculate principal stretches and axis. These are the eigenvalues/vectors
    of the tensor U from F=RU. Since C = U^2 = 2 gam + I, we can calculate the
    stretches and axis from the lagrangian strain
    """
    # Right Cauchy tensor
    C = 2 * np.array(lagrange_strain) + np.eye(3)

    # Calculate for each atom
    stretches = []
    axis_1, axis_2, axis_3 = [], [], []
    for Ca in C:
        ls, vs = spla.eig(Ca)
        # Sort eigen-values
        stretches.append(np.sort(ls))

        # Sort eigen-vectors
        axis_1.append(vs[:, np.argsort(ls)[0]])
        axis_2.append(vs[:, np.argsort(ls)[1]])
        axis_3.append(vs[:, np.argsort(ls)[2]])

    # Convert to array
    stretches = np.array(stretches)
    axis_1 = np.array(axis_1)
    axis_2 = np.array(axis_2)
    axis_3 = np.array(axis_3)

    # Use a single hemisphere, as this carries no information
    neg_z_1 = np.argwhere(axis_1[:, 2] < 0)
    neg_z_2 = np.argwhere(axis_2[:, 2] < 0)
    neg_z_3 = np.argwhere(axis_3[:, 2] < 0)
    axis_1[neg_z_1, :] = -axis_1[neg_z_1, :]
    axis_2[neg_z_2, :] = -axis_2[neg_z_2, :]
    axis_3[neg_z_3, :] = -axis_3[neg_z_3, :]

    # Randomly distribute half in the other hemisphere, for consistency
    axis_size = np.shape(axis_3)[0]
    flip_1 = np.random.choice(range(axis_size), axis_size // 2, replace=False)
    flip_2 = np.random.choice(range(axis_size), axis_size // 2, replace=False)
    flip_3 = np.random.choice(range(axis_size), axis_size // 2, replace=False)
    axis_1[flip_1, :] = -axis_1[flip_1, :]
    axis_2[flip_2, :] = -axis_2[flip_2, :]
    axis_3[flip_3, :] = -axis_3[flip_3, :]

    return np.array(stretches), [axis_1, axis_2, axis_3]


def new_principal_stretches_from_g(lagrange_strain):
    """
    Calculate principal stretches and axis. These are the eigenvalues/vectors
    of the tensor U from F=RU. Since C = U^2 = 2 gam + I, we can calculate the
    stretches and axis from the lagrangian strain
    """
    # Right Cauchy tensor
    C = 2 * np.array(lagrange_strain) + np.eye(3)

    # Calculate for each atom
    stretches = []
    axis_1, axis_2, axis_3 = [], [], []
    for Ca in C:
        ls, vs = spla.eig(Ca)
        # Sort eigen-values
        stretches.append(np.sort(ls))

        # Sort eigen-vectors
        axis_1.append(vs[:, np.argsort(ls)[0]])
        axis_2.append(vs[:, np.argsort(ls)[1]])
        axis_3.append(vs[:, np.argsort(ls)[2]])

    axes = np.transpose(
        np.array([np.array(axis_1), np.array(axis_2), np.array(axis_3)]), (1, 0, 2)
    )

    return np.array(stretches), axes


# Energy functions


def saint_venant_energy_density(lagrange_strain, first_lame = 7.30e7, second_lame = 3.76e7):
    """
    Calculate the energy density for a hyperelastic material considering the
    Saint Venant-Kirchhoff model, psi = lambda/2 * tr(gam)^2 + mu * tr(gam^2)
    [Holzapfel, 2000], where lambda and mu are the first and the second Lamé's
    coefficients. The default values were calculated considering the Young's
    moduls Y = 100MPa and the Poisson's ratio nu = 1/3.
    """
    # Calculate stretches squared
    stretches2, _ = principal_stretches_from_g(lagrange_strain)
    stretches2 = np.abs(stretches2)
    
    # Calculate trace elements
    tr2_gam = np.sum((stretches2-1)/2, axis = 1)**2
    tr_gam2 = np.sum(((stretches2-1)/2)**2, axis = 1)
    
    psi = first_lame / 2 * tr2_gam + second_lame * tr_gam2
    
    return psi


def saint_venant_energy(energy_density, protein_volume, labels, 
                        units = "kT", verbose = False):
    """
    Calculate the energy a hyperelastic material considering the
    Saint Venant-Kirchhoff model. It integrates the energy density over the
    protein volume considering the effective volume occupied by each atom.
    
    The output is returned in kT units as default, other options are
    'kJ/mol', 'kcal/mol', and 'J'.
    """
    
    vf = spatial.effective_volume(labels, protein_volume, verbose)
    E = energy_density * vf
    
    if units == "kT":
        kT = 1.380649e-23*300
        E = E/kT
    elif units == "kJ/mol":
        mol =  6.02214076e23
        E = E*mol/1e3
    elif units == "kcal/mol":
        kcal = 4.184e3
        mol =  6.02214076e23
        E = E*mol/kcal
    elif units != "J":
        print("Units of energy not recognized, returning energy in Joules (J)")
    
    return E
    
    
def elastic_energy(lagrange_strain, labels, protein_volume, 
                   units = "kT", verbose = False,
                   first_lame = 7.30e7, second_lame = 3.76e7):
    """
    Condensated function to calculate the elastic energy, calling
    'saint_venant_energy_density' and 'saint_venant_energy'.
    """
    
    psi = saint_venant_energy_density(lagrange_strain, first_lame, second_lame)
    E = saint_venant_energy(psi, protein_volume, labels, units, verbose)
    
    return E    
    

# Set of functions to calculate weights and deformation gradient not optimized


def bintersect_weights(xyz_list, method="linear", parameters=[6.0, 8.0]):
    """
    Calculate the product of the weights for each coordinate set. Note that the
    weight matrix is not symmetric.
    """
    weight_prod = np.sign(compute_weights(xyz_list[0], method, parameters))

    for xyz in xyz_list[1:]:
        weight_prod *= np.sign(compute_weights(xyz, method, parameters))

    return weight_prod


def average_weights(xyz_list, method="linear", parameters=[6.0, 8.0]):
    """
    Calculate the average of the weights for each coordinate set. Note that the
    weight matrix is not symmetric.
    """
    weight_avg = np.zeros((np.shape(xyz_list[0])[0], np.shape(xyz_list[0])[0]))
    for xyz in xyz_list:
        weight_avg += compute_weights(xyz, method, parameters)
    weight_avg /= len(xyz_list)

    return weight_avg


def compute_weights(xyz, method="linear", parameters=[6.0, 8.0]):
    """
    This function
    """
    cases = {"linear": linear_weights, "minimal": minimal_weights}
    return cases[method](xyz, parameters)


def linear_weights(xyz, parameters=[6.0, 8.0]):
    """
    Calculate the weights of the given positions using a linear decay function
    (two radii are provided).
    """
    # Read parameters, inner and outer radii
    r1, r2 = parameters[0], parameters[1]

    # Calculate distances
    res_dis = distance.pdist(xyz)
    res_dis = distance.squareform(res_dis)

    # Make linear function
    weights = (res_dis - r1) / (r2 - r1)
    weights = 1 - weights

    # Cap between 1 and 0
    weights = np.minimum(weights, 1)
    weights = np.maximum(weights, 0)

    return weights


def minimal_weights(xyz, parameters=[3]):
    """
    Weight matrix is the minimal required to solve the equations of the defor-
    mation gradients. That is, the three nearest neighbors.
    """
    # Read parameters, required atoms
    req = int(parameters[0])

    # Calculate distances
    res_dis = distance.pdist(xyz)
    res_dis = distance.squareform(res_dis)

    # Select required nearest-neighbors + 1 (self)
    nn_req = np.argsort(res_dis)[:, : req + 1]

    # Create weight matrix
    weights = np.zeros_like(res_dis)
    for i, nn in enumerate(nn_req):
        weights[i][nn] = 1.0

    return weights


def deformation_gradient(weights, xyz_rel, xyz_def):
    """
    Calculate the deformation gradient tensor for each (F=dx/dX) using the
    approach from [Gullet et al,] (Eq. X), or Eq. 17 in [Zimmerman et al, 2009]
    In solving, note that D.T*U=A.T -> U.T*D=A -> F=U.T
    """

    num_res = weights.shape[0]
    F = np.zeros((num_res, 3, 3))

    for i in range(num_res):
        D = np.zeros((3, 3))
        A = np.zeros((3, 3))
        for j in range(num_res):
            if i != j and weights[i, j] > 0.0:
                # Calculate relaxed/deformed vectors
                dX = xyz_rel[j, :] - xyz_rel[i, :]
                dx = xyz_def[j, :] - xyz_def[i, :]
                # Generate intermediate matrices
                D += np.dot(dX[:, None], dX[None, :]) * weights[i, j]
                A += np.dot(dx[:, None], dX[None, :]) * weights[i, j]

        # Solve
        F[i, :, :] = spla.solve(D.T, A.T).T

    return F


def deformation_gradient_o2(weights, xyz_rel, xyz_def):
    """
    Calculate the deformation gradient tensor for each (F=dx/dX) using the
    approach from [Zimmerman et al, 2009]. We define the tensors omega (15),
    eta (16), xi (25), nu (28), phi(29), zeta (34)
    """

    num_res = weights.shape[0]
    F = np.zeros((num_res, 3, 3))
    H = np.zeros((num_res, 3, 3, 3))

    for i in range(num_res):
        omega = np.zeros((3, 3))
        eta = np.zeros((3, 3))
        xi = np.zeros((3, 3, 3))
        nu = np.zeros((3, 3, 3))
        phi = np.zeros((3, 3, 3, 3))
        for j in range(num_res):
            if i != j and weights[i, j] > 0.0:
                # Calculate relaxed/deformed vectors
                dX = xyz_rel[j, :] - xyz_rel[i, :]
                dx = xyz_def[j, :] - xyz_def[i, :]
                # Generate tensors
                omega += np.tensordot(dx, dX, 0) * weights[i, j]
                eta += np.tensordot(dX, dX, 0) * weights[i, j]
                xi += np.tensordot(np.tensordot(dX, dX, 0), dX, 0) * weights[i, j]
                nu += np.tensordot(np.tensordot(dx, dX, 0), dX, 0) * weights[i, j]
                phi += (
                    np.tensordot(np.tensordot(np.tensordot(dX, dX, 0), dX, 0), dX, 0)
                    * weights[i, j]
                )

        # Test for sufficient neighbors

        # Auxiliry tensors
        eta_inv = spla.inv(eta)
        zeta = phi - np.dot(np.dot(xi, eta_inv), xi)
        zeta_inv = npla.tensorinv(zeta, ind=2)

        # Deformation gradients
        bracket = 2.0 * (nu - np.dot(omega, np.dot(eta_inv, xi)))
        # H[i, :, :, :] = np.einsum(bracket, zeta_inv, 'iST,TSLK->iKL')
        # H[i,0,:,:]=#tensorsolve #np.einsum(bracket,zeta_inv, 'iST,TSLK->iKL')
        # H[i,1,:,:]=#tensorsolve #np.einsum(bracket,zeta_inv, 'iST,TSLK->iKL')
        # H[i,2,:,:]=#tensorsolve #np.einsum(bracket,zeta_inv, 'iST,TSLK->iKL')
        F[i, :, :] = np.dot(omega, eta_inv)
        F[i, :, :] -= 0.5 * np.einsum(H[i], np.dot(xi, eta_inv), " iKL,KLJ->iJ")

    return F
