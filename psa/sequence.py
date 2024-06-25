"""
This file contains functions that assist in loading and pre-processin g the
pdb files.
"""

from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from Bio import Align
from Bio import SeqUtils
from Bio.SeqUtils import seq1
from urllib.request import urlopen
import os
import re
import glob
import warnings
import numpy as np
from . import load


def generate_reference(ref_pps):
    """
    Generate dictionary of reference sequence objects from the reference 
    structure.
    """

    # Loop over uniprot chain ids
    reference_seqs = {}

    for chain in ref_pps:
        seq_join = generate_sequence(chain)
        chain_id = load.get_chain_name(chain)

        # Generate reference sequence
        ref_seq = SeqIO.SeqRecord(seq_join, id=chain_id, description="")

        # Fix sequences of unknown residues
        ref_seq = fix_unknown_sequence(ref_seq)
        reference_seqs[chain_id] = ref_seq
    return reference_seqs


def generate_sequence(chain):
    """
    Collapses the sequence(s) from a single or multiple polypeptides in 
    a string
    """
    # Get pps in chain
    pp_seqs = [pp.get_sequence() for pp in chain]
    seq_join = pp_seqs[0]

    # Stitch pps and generate seq
    for seq in pp_seqs[1:]:
        seq_join += seq

    return seq_join


def fix_unknown_sequence(sequence):
    """
    If a sequence is made of unknwon residues, labeled X (6J5I, u), replace by
    a repetitive sequence of DTS residues, so that alignment is possible. This
    only occurs if in build_peptides we have aa_only=0.
    """

    if sequence.seq.count("X") == len(sequence):
        print(sequence)
        warnings.warn("Aligning pps made of unknown res.", stacklevel=2)
        s = "DTS" * (len(sequence) // 3) + "DTS"[: len(sequence) % 3]
        fixed_sequence = SeqRecord(id="query", description="", seq=Seq(s))
        return sequence
    else:
        return sequence


def download_fasta(up_id):
    """
    Download fasta sequence if it is not present
    """
    filename = "data/seq/" + up_id + ".fa"
    if not os.path.exists(filename):
        target_url = "http://www.uniprot.org/uniprot/" + up_id + ".fasta"
        url = urlopen(target_url)
        infile = open(filename, "wb")
        infile.write(url.read())
        infile.close()
    return


def pairwise_alignment(rel_pps, def_pps):
    """
    Perform the default alignment pipeline for two structures with the
    same number of chains
    """
    # Create a reference sequence
    my_ref_seqs = generate_reference(rel_pps)

    # Align both structures to reference
    rel_idx = align(rel_pps, my_ref_seqs)
    def_idx = align(def_pps, my_ref_seqs)

    # Obtain a dictionary with residues that aligned correctly
    rel_dict = aligned_dict(rel_pps, rel_idx, my_ref_seqs)
    def_dict = aligned_dict(def_pps, def_idx, my_ref_seqs)

    # Intersect the aligned residues
    com_at = common(rel_dict, def_dict)

    return com_at, rel_dict, def_dict


def pairwise_alignment_multiple(structures, ref_struc=0):
    """
    Receive a list of structures and performs pairwise sequence 
    alignemnt for each of the chains using one of the structures
    as reference. Returns the set of common atoms and a list with 
    n = len(sturctures) dicts of labels
    
    Note that this method does not peform a multiple sequence 
    alignemnt
    """
    n = len(structures)

    al_dicts = []
    for i in range(n):
        pair = pairwise_alignment(structures[ref_struc], structures[i])
        al_dicts.append(pair[2])

    common_at = common_multiple(al_dicts)

    return common_at, al_dicts


def align(pps, ref_seqs):
    """
    Alignment the polypeptides in each chain of each protein to the reference
    sequences using local alignment. It returns the indices of the
    ref-seq and pps where the alignment of a segment starts and stops.
    """
    # Load sequence aligner
    aligner = Align.PairwiseAligner()
    aligner.mode = "global"
    aligner.open_gap_score = -0.5
    aligner.extend_gap_score = -0.1

    # For each chain, start/stop indices of aligned segments from ref
    start_stop = []  # ((ref_st1,ref_sp1),...,(N)), ((pep_st1,pep_sp1),...,(N))
    zeros = np.zeros((2, 1, 2), dtype=int)

    # Loop over proteins and chains
    for chain in pps:
        chain_id = load.get_chain_name(chain)

        # For chains contained in the reference
        if chain_id in ref_seqs:
            ref_seq = ref_seqs[chain_id]
            seq = generate_sequence(chain)
            alignments = aligner.align(ref_seq.seq, seq)
            start_stop.append(alignments[0].aligned)

        # For chains absent in the reference
        else:
            start_stop.append(zeros)

    # Remove temp files
    for f in glob.glob("tmp/alignment_*"):
        os.remove(f)

    return start_stop


def align_seqs(seq_pair):
    """
    Align a pair of sequences, provided as strings, and report corresponding
    masks for the aligned residues as well as the alignment score
    """
    # Load the aligner
    aligner = Align.PairwiseAligner()
    aligner.mode = "global"
    aligner.open_gap_score = -0.5
    aligner.extend_gap_score = -0.1

    # Align seqs, get start/stop and score
    al = aligner.align(seq_pair[0], seq_pair[1])
    ss = al[0].aligned
    sc = al.score

    # Create masks
    masks = [
        np.zeros(len(seq_pair[0]), dtype="bool"),
        np.zeros(len(seq_pair[1]), dtype="bool"),
    ]
    for m, s in zip(masks, ss):
        for el in s:
            m[el[0] : el[1]] = 1

    return masks, sc


def aligned_dict(pps, start_stop, ref_seqs):
    """
    A dictionary with keys the labels of the aligned residues in the pdb
    language, and values the labels in a self-made language that uses the
    reference sequence to identify each residue. This second language is
    common to all structures
    """

    aligned = {}
    ref_idx = list(ref_seqs.keys())

    # Loop over chains, peptides, residues and aligned segments
    for c_i, chain in enumerate(pps):
        chain_id = load.get_chain_name(chain)
        if chain_id in ref_seqs:
            idx = ref_idx.index(chain_id)
        res_i = 0

        for pep in chain:
            for res in pep:
                for s_i in range(len(start_stop[c_i][1])):
                    # Indexes from the reference
                    seg_ref_st = start_stop[c_i][0][s_i][0]
                    seg_ref_sp = start_stop[c_i][0][s_i][1]

                    # Indexes from the query
                    seg_qry_st = start_stop[c_i][1][s_i][0]
                    seg_qry_sp = start_stop[c_i][1][s_i][1]

                    # Verify if the residue is in the alignment range
                    if res_i >= seg_qry_st and res_i < seg_qry_sp:
                        n_num = res_i + seg_ref_st - seg_qry_st

                        # Store entry if residue passes test
                        if load.test_residue(res):
                            # Loop over atoms of the residue
                            for a_i, at in enumerate(res):
                                aligned[at.full_id] = (idx, n_num, at.id)
                res_i += 1

    return aligned


def common(rel_dict, def_dict):
    """
    Compute the set of common atoms by finding the intersect of two
    dictionaries. We use the self-made language (i.e., dict values)
    """
    # Create sets with residues labels
    rel_set = set(rel_dict.values())
    def_set = set(def_dict.values())

    # Calculate the intersect set
    common_at = rel_set.intersection(def_set)

    return common_at


def common_multiple(vec_dict):
    """
    Compute the set of common atoms by finding the intersect of two 
    or more dictionaries. We use the self-made language (i.e., dict values)
    """

    # Create sets with residues labels
    vec_set = []
    for i in range(len(vec_dict)):
        vec_set.append(set(vec_dict[i].values()))

    # Calculate the intersect set
    common_at = vec_set[0].intersection(vec_set[1])
    for i in range(1, len(vec_dict) - 1):
        temp_set = vec_set[i].intersection(vec_set[i + 1])
        common_at = common_at.intersection(temp_set)

    return common_at


def conservation(dat_list):
    """
    This function performs conservation analysis on a list of ordered data.
    Of top X data-values, it determines what fraction have the same index.
    """
    # Storage lists
    cons_frac, dat_sorted = [], []
    dat_len = len(dat_list[0])

    # Sort all data
    for dat in dat_list:
        dat_sorted.append(dat.argsort()[::-1])

    # Loop over top i residues
    for i in range(dat_len)[1:]:
        dat_sets = []
        # Create sorted sets
        for dat in dat_sorted:
            dat_sets.append(set(dat[:i]))
        # Calculate fraction intersecting
        cons_set = set.intersection(*dat_sets)
        xi = len(cons_set) / (i)
        # Store
        cons_frac.append(xi)

    return cons_frac
