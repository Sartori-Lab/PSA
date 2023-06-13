"""
This file contains functions that assist in loading and pre-processin g the
pdb files.
"""

from Bio import PDB
from urllib.request import urlretrieve
import os
import numpy as np

# Chain_ids should be a dictionary. Also, there should be a fucntion to
# download this


def single_structure(name, relabel=None, transform=None, model_id=0,
                     chain_ids=None, path='./', file_type='pdb'):
    """
    Load PDB file, read structure and generate list of
    polypeptide objects (several polypeptides consti-
    tute a chain).
    """
    # Load parser
    if file_type == 'pdb':
        parser = PDB.PDBParser()
    elif file_type == 'cif':
        parser = PDB.MMCIFParser()
    ppb = PDB.PPBuilder()

    # Download and load pdb/cif file
    download_file(name, path, file_type)
    filename = path + name + '.' + file_type
    structure = parser.get_structure(name, filename)

    # Load chains, maybe relabel
    chains = load_chains(structure[model_id], relabel, transform)

    # Load peptides
    all_pps = [ppb.build_peptides(ch) for ch in chains]

    # Choose specific chains from pps
    pps = choose_chains(all_pps, chain_ids)

    # Sort pps by chain label
    ch_ids = []
    for chain in pps:
        ch_ids.append(chain[0][0].get_parent().id)
    pps_sorted = [x for _, x in sorted(zip(ch_ids, pps))]

    return pps_sorted


def get_chain_name(chain):
    """
    Return the ID of a chain or a polypeptide.
    """
    
    if type(chain) == PDB.Chain.Chain:
        return chain.get_id()
    elif type(chain) == PDB.Polypeptide.Polypeptide:
        return chain[0].get_parent().get_id()
    elif type(chain) == list:
        return chain[0][0].get_parent().get_id()
    

def load_chains(structure, relabel=None, transform=None):
    """
    Load all chains performing, when necessary, relabeling and assembly trans-
    formations.
    """
    chains = []
    # Load all chains
    for chain in structure.get_chains():
        chains.append(chain)

    # Perform relabeling, when necessary
    if relabel:
        relabel_chains(chains, relabel)

    # Apply transformations
    if isinstance(transform, np.ndarray):
        chains = transform_chains(chains, transform)

    return chains


def relabel_chains(chains, relabel):
    """
    Relabel the chains, keeping the chains consistent at all times by using
    tempral dummy chain_ids
    """
    # Initial relabeling, with dummy id
    for chain in chains:
        if chain.id in relabel.keys():
            chain.id = relabel[chain.id] + '_'

    # Adjust labeling, final id
    for chain in chains:
        if chain.id[-1] == '_':
            chain.id = chain.id[:-1]

    return


def transform_chains(chains, transform):
    """
    Generate new chains that are symmetrically related to the previous ones
    via the transform input, which is taken directly from PDB.
    """

    N_tr = transform.shape[0] // 3
    chains_new = []

    for chain in chains:
        chains_new.append(chain)
        for i in range(1, N_tr):
            # Create new chain
            chain_i = chain.copy()
            # Transformation chain
            chain_i.transform(transform[i*3:(i+1)*3, :3].T,
                              transform[i*3:(i+1)*3, 3])
            # Rename chain
            chain_i.id = chain_i.id + '-' + str(i)
            chains_new.append(chain_i)

    return chains_new


def download_file(name, path='./', file_type='pdb'):
    """
    Download pdb files if they are not present
    """
    filename = path + name + '.' + file_type
    if not os.path.exists(filename):
        targt_url = 'https://files.rcsb.org/download/' + name + '.' + file_type
        url = urlretrieve(targt_url, filename)


def char_range(c1, c2):
    """
    Generates the characters from `c1` to `c2`, inclusive. Respectics caps.
    """
    for c in range(ord(c1), ord(c2)+1):
        yield chr(c)


def choose_chains(all_pps, chain_ids=None):
    """
    Choose only polypeptides that belong to particular chains
    """
    pps = []

    # Loop over all pps
    for chain in all_pps:
        # Check that there is a chain, e.g. 6J5J missing e, u, g, k
        if chain:
            # Read chain ID from first res in first pp
            ch_id = chain[0][0].get_parent().id
            # Store all chains if no list if given
            if not chain_ids:
                pps.append(chain)
            # Store if it matches our list
            elif ch_id in chain_ids:
                pps.append(chain)

    return pps


def coordinates(pps, common_res=None, al_dict=None):
    """
    Get coordinates of the polypeptides that were aligned. To do so, we check
    whether they are in the common set using the alignment dictionary.
    """

    coordinates = []
    labels = []

    # Loop over chains, peptides and residues
    for chain in pps:
        for pep in chain:
            for res in pep:
                if test_residue(res, common_res, al_dict):
                    coordinates.append(res['CA'].coord)
                    labels.append([res.full_id, res.resname])

    return np.array(coordinates), labels


def coordinates_all_atoms(pps, common_res=None, al_dict=None):
    """
    Get coordinates of the polypeptides that were aligned. To do so, we check
    whether they are in the common set using the alignment dictionary.
    """

    coordinates = []
    labels = []

    # Loop over chains, peptides and residues
    for chain in pps:
        for pep in chain:
            for res in pep:
                if test_residue(res, common_res, al_dict):
                    for at in res:
                        coordinates.append(at.coord)
                        labels.append([at.full_id, at.element])

    return np.array(coordinates), labels


def chain_indices(chain_labels, coordinates_labels, residue_labels=None):
    """
    Returns the indices of the residues that correspond to certain chains. If
    no residue labels are provided, then whole chain is returned.
    """
    chain_indices = []
    if not residue_labels:
        residue_labels = [None] * len(chain_labels)

    for chain_lbl, residue_lbl in zip(chain_labels, residue_labels):
        chain_indices.append([])
        for ind in range(len(coordinates_labels)):
            if coordinates_labels[ind][0][2] == chain_lbl:
                if not residue_lbl:
                    chain_indices[-1].append(ind)
                elif coordinates_labels[ind][0][3][1] in residue_lbl:
                    chain_indices[-1].append(ind)
    return chain_indices


def atoms(pps, common_res=None, al_dict=None):
    """
    Get atom object of the polypeptides that were aligned. To do so, we check
    whether they are in the common set using the alignment dictionary.
    """

    atoms = []
    labels = []

    # Loop over chains, peptides and residues
    for chain in pps:
        for pep in chain:
            for res in pep:
                if test_residue(res, common_res, al_dict):
                    atoms.append(res['CA'])
                    labels.append([res.full_id, res.resname])

    return atoms, labels


def test_residue(res, common_res=None, al_dict=None):
    """
    Perform some basic checks on the quality of the data for the residues, and
    if a common set is provided, test whether they belong to it.
    """
    # Is this really a residue? (or e.g. Mg2+)
    if type(res) != PDB.Residue.Residue:
        return False
    # Does it contain CA?
    elif not res.has_id('CA'):
        return False
    # Is CA disordered?
    elif type(res['CA']) == PDB.Atom.DisorderedAtom:
        return False
    # If there is no common set or no dict, no more tests
    if not common_res or not al_dict:
        return True
    # Is the residue aligned?
    if res.full_id not in al_dict.keys():
        return False
    # Test belonging
    else:
        return al_dict[res.full_id] in common_res


def save_bfactor(name, bfact=None, ext='_rot_clust', path = './', file_type='pdb'):
    """
    Use the labels->numbers bfact dictionary to modify the beta factors
    of a given structure and save it with an extension.
    """
    # Load parser/writer
    if file_type == 'pdb':
        parser = PDB.PDBParser()
        writer = PDB.PDBIO()
    elif file_type == 'cif':
        parser = PDB.MMCIFParser()
        writer = PDB.MMCIFIO()
    filename = path + name + '.' + file_type
    structure = parser.get_structure(name, filename)

    # Modify structure with new beta factors
    for chain in structure[0]:
        for res in chain:
            # for analyzed residues, give cluster color
            if res.full_id in bfact.keys():
                for atom in res:
                    atom.set_bfactor(bfact[res.full_id])
            # for non-analyzed, give arbitrary label, here -2
            else:
                for atom in res:
                    atom.set_bfactor(-2.)

    # write output
    writer.set_structure(structure)
    new_filename = filename[:-4] + ext + '.' + file_type
    writer.save(new_filename)
    
def save_txt(rel_pdb, def_pdb, variables, 
             labels, ext = "", path = "./"):
    """
    Creates a .txt file to contain data from variables received as argument.
    The variables should be a vector of arrays for different quantities. The
    length of each array should match the number of entries on the atom labels.
    ex: variables = [stretches[:, 0]] to save \lambda_1.
    """

    filename = path + rel_pdb + "-" + def_pdb
    if ext:
        filename += "-" + ext
    
    with open(filename + ".txt", "w") as f:
        for label, i in zip(labels, range(len(labels))):
            line = str(label[0]) + ": "
            
            for var in variables:
                line += str(var[i]) + " "
            line = line[:-1]  + "\n"
            
            f.write(line)
    return