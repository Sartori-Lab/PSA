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

# Internal
from . import load


def generate_reference(ref_pps, uniprot_ids):
    """
    Generate list of reference sequence objects. Either from uniprot_ids
    entries, or from the reference structure.
    """
    # Debug input
    assert len(ref_pps) == len(uniprot_ids),\
        "More/less chains than uniprot ids"

    # Loop over uniprot chain ids
    reference_seqs = []
    for ind, up_id in enumerate(uniprot_ids):
        # Download and read fasta seq
        if up_id:
            download_fasta(up_id)
            ref_seq = SeqIO.read('data/seq/' + up_id + '.fa',
                                 format='fasta')
        # Or generate seq from ref_pps
        else:
            # Get pps in chain
            pp_seqs = [pp.get_sequence() for pp in ref_pps[ind]]
            seq_join = pp_seqs[0]
            # Stitch pps and generate seq
            for seq in pp_seqs[1:]:
                seq_join += seq
            # Generate reference sequence
            ref_seq = SeqIO.SeqRecord(seq_join,
                                      id='query',
                                      description='')
            # Fix sequences of unknown residues
            ref_seq = fix_unknown_sequence(ref_seq)
        reference_seqs.append(ref_seq)
    return reference_seqs


def fix_unknown_sequence(sequence):
    """
    If a sequence is made of unknwon residues, labeled X (6J5I, u), replace by
    a repetitive sequence of DTS residues, so that alignment is possible. This
    only occurs if in build_peptides we have aa_only=0.
    """

    if sequence.seq.count('X') == len(sequence):
        print(sequence)
        warnings.warn('Aligning pps made of unknown res.', stacklevel=2)
        s = 'DTS' * (len(sequence)//3) + 'DTS'[:len(sequence) % 3]
        fixed_sequence = SeqRecord(id='query', description='', seq=Seq(s))
        return sequence
    else:
        return sequence


def download_fasta(up_id):
    """
    Download fasta sequence if it is not present
    """
    filename = 'data/seq/' + up_id + '.fa'
    if not os.path.exists(filename):
        target_url = 'http://www.uniprot.org/uniprot/' + up_id + '.fasta'
        url = urlopen(target_url)
        infile = open(filename, 'wb')
        infile.write(url.read())
        infile.close()
    return

def pairwise_alignment(rel_pps, def_pps):
    """
    Perform the default alignment pipeline for two structures with the
    same number of chains
    """
    
    # Generate a list of None's, we use sequence of residues from PDB
    uni_ids = [None] * len(rel_pps)
    my_ref_seqs = generate_reference(rel_pps, uni_ids)
    
    # Align both structures to reference
    rel_idx = align(rel_pps, my_ref_seqs)
    def_idx = align(def_pps, my_ref_seqs)
    
    # Obtain a dictionary with residues that aligned correctly
    rel_dict = aligned_dict(rel_pps, rel_idx)
    def_dict = aligned_dict(def_pps, def_idx)
    
    # Intersect the aligned residues 
    com_res = common(rel_dict, def_dict)

    return com_res, rel_dict, def_dict

def align(pps, ref_seqs):
    """
    Alignment the polypeptides in each chain of each protein to the reference
    sequences using local alignment. It returns the indices of the
    ref-seq and pps where the alignment of a segment starts and stops.
    """
    # Load sequence aligner
    aligner = Align.PairwiseAligner()
    aligner.mode = 'local'
    aligner.open_gap_score = -.5
    aligner.extend_gap_score = -.1

    # For each chain&pep, start/stop indices of aligned segments from ref/pep
    start_stop = []  # ((ref_st1,ref_sp1),...,(N)), ((pep_st1,pep_sp1),...,(N))

    # Loop over proteins, chains and peptides
    for (chain, ref_seq) in zip(pps, ref_seqs):
        start_stop.append([])
        for pep in chain:
            # Align peptide ro reference sequence
            alignments = aligner.align(ref_seq.seq, pep.get_sequence())
            start_stop[-1].append(alignments[0].aligned)

    # Remove temp files
    for f in glob.glob("tmp/alignment_*"):
        os.remove(f)
        
    return start_stop


def align_seqs(seq_pair):
    '''
    Align a pair of sequences, provided as strings, and report corresponding
    masks for the aligned residues as well as the alignment score
    '''
    # Load the aligner
    aligner = Align.PairwiseAligner()
    aligner.mode = 'local'
    aligner.open_gap_score = -.5
    aligner.extend_gap_score = -.1

    # Align seqs, get start/stop and score
    al = aligner.align(seq_pair[0], seq_pair[1])
    ss = al[0].aligned
    sc = al.score

    # Create masks
    masks = [np.zeros(len(seq_pair[0]), dtype='bool'),
             np.zeros(len(seq_pair[1]), dtype='bool')]
    for m, s in zip(masks, ss):
        for el in s:
            m[el[0]:el[1]] = 1

    return masks, sc


def aligned_dict(pps, start_stop):
    """
    A dictionary with keys the labels of the aligned residues in the pdb
    language, and values the labels in a self-made language that uses the
    reference sequence to identify each residue. This second language is
    common to all structures
    """

    aligned = {}

    # Loop over chains, peptides, residues and aligned segments
    for c_i, chain in enumerate(pps):
        for p_i, pep in enumerate(chain):
            for r_i, res in enumerate(pep):
                for s_i in range(len(start_stop[c_i][p_i][1])):
                    seg_ref_st = start_stop[c_i][p_i][0][s_i][0]
                    seg_ref_sp = start_stop[c_i][p_i][0][s_i][1]
                    seg_pep_st = start_stop[c_i][p_i][1][s_i][0]
                    seg_pep_sp = start_stop[c_i][p_i][1][s_i][1]
                    if r_i >= seg_pep_st and r_i < seg_pep_sp:
                        n_num = r_i + seg_ref_st - seg_pep_st
                        # Test duplicates
                        # assert res.full_id not in aligned.keys(),\
                        # 'Duplicated key in aligned dict'
                        # assert (c_i, n_num) not in aligned.values(),\
                        # 'Duplicated value in aligned dict'

                        # Store entry if residue passes test
                        if load.test_residue(res):
                            aligned[res.full_id] = (c_i, n_num)

    return aligned


def common(rel_dict, def_dict):
    """
    Compute the set of common residues by finding the intersect of two
    dictionaries. We use the self-made language (i.e., dict values)
    """
    # Create sets with residues labels
    rel_set = set(rel_dict.values())
    def_set = set(def_dict.values())

    # Calculate the intersect set
    common_res = rel_set.intersection(def_set)

    return common_res


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
