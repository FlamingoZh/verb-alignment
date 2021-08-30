import sys

import copy
import itertools
import json
import math
from pathlib import Path
import pickle
from pprint import pprint

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import rc
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
from PIL import Image
#import pingouin as pg
from scipy.stats import spearmanr, ttest_ind, sem
from scipy.special import comb
from sklearn.manifold import MDS, TSNE
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.mixture import GaussianMixture

from itertools import chain
import random

from utils.utils_funcs import average_multi

def permutation(z_0,z_1,max_perm=100,n_perm_keep=100):
    n_item = z_0.shape[0]

    _, rho_array_2, perm_list_2 = permutation_analysis(
        [z_0, z_1], 'last', max_perm=n_perm_keep
    )
    # print("rho_array_2:",rho_array_2)
    rho_array_2 = rho_array_2[0:n_perm_keep + 1]
    n_mismatch_2 = infer_mistmatch(perm_list_2)[0][0:n_perm_keep + 1]

    rho_array = rho_array_2
    n_mismatch = n_mismatch_2

    for val_sub in range(3,n_item+1):
        #print("val_sub:", val_sub)

        _, rho_array_sub, perm_list_sub = permutation_analysis(
            [z_0, z_1], 'sub', max_perm=max_perm, n_unknown=val_sub,
            is_exact=True
        )
        n_mismatch_sub = infer_mistmatch(perm_list_sub)[0]
        # Filter down to unique permutations.
        (_, keep_idx) = np.unique(perm_list_sub[0], return_index=True, axis=0)
        rho_array_uniq = rho_array_sub[keep_idx]
        n_mismatch_uniq = n_mismatch_sub[keep_idx]
        rho_array = np.hstack((rho_array, rho_array_uniq[1:]))
        n_mismatch = np.hstack((n_mismatch, n_mismatch_uniq[1:]))

    return n_item, n_mismatch, rho_array

def permutation_analysis(z_list, perm_type, max_perm=10000, n_unknown=2, n_known=0, is_exact=False):
    """Perform permutation analysis.

    This function assumes embeddings are passed in correctly aligned.
    """
    # Pre-compute similarity matrices.
    sim_mat_list = []
    for z_idx,z in enumerate(z_list):
        #print("z",z_idx,":",z.shape)

        sim_mat = cosine_similarity(z)
        sim_mat_list.append(sim_mat)

    if perm_type == 'last':
        rho_array, perm_list = permutation_analysis_last(
            sim_mat_list, max_perm=max_perm, n_known=n_known
        )
    elif perm_type == 'sub':
        rho_array, perm_list = permutation_analysis_sub(
            sim_mat_list, max_perm=max_perm, n_unknown=n_unknown,
            is_exact=is_exact
        )
    else:
        raise ValueError(
            'Permutation analysis {0} is not implemented'.format(perm_type)
        )
    # Compute proportion of permutations worse (i.e., less) than the correct
    # alignment.
    n_perm = len(rho_array) - 1
    perm_percentile = np.sum(np.less(rho_array[1:], rho_array[0])) / n_perm
    return perm_percentile, rho_array, perm_list

def permutation_analysis_last(sim_mat_list, max_perm=10000, n_known=0):
    """Perform last-two permutation analysis.

    Swap same rows of matrices. If more than two matrices, must decide
    which matrices will have there rows swapped (although at least one
    will always have their rows swapped).

    In order to evaluate our ability to correctly align matrices, we
    first compute the correlation of the correct ordering and then
    compute the correlations for the sampled permutations. What we
    would like to see is that the correlation for the correct ordering
    is higher than any of the permuted alignments.
    """
    n_sim_mat = len(sim_mat_list)
    alignment_combos = list(itertools.combinations(np.arange(n_sim_mat, dtype=int), 2))

    n_item = sim_mat_list[1].shape[0]
    # print("n_sim_mat:",n_sim_mat)
    # print("n_item:",n_item)

    # Sample from possible permutations. We only consider permutations that
    # have a single pair of indices swapped. For ten elements there are
    # 45 potential combinations.
    list_swap_idx = list(
        itertools.combinations(np.arange(n_item - n_known, dtype=int), 2)
    )
    list_swap_idx = np.array(list_swap_idx, dtype=int)
    #print(list_swap_idx)
    n_perm = np.minimum(max_perm, list_swap_idx.shape[0], dtype=int)
    #print(max_perm,list_swap_idx.shape[0],n_perm)
    # Select subset of swaps out of all possibilties.
    rand_idx = np.random.permutation(len(list_swap_idx))[0:n_perm]
    selected_swap_idx = list_swap_idx[rand_idx, :]

    # If more than two matrices, determine which matrices will have swapped
    # values. The first matrix is always not swapped, but there are many
    # other possibilities for remaining matrices (e.g., with three matrices:
    #  001, 010, or 011)
    if (n_sim_mat - 1) == 1:
        is_swapped = np.ones((n_perm, 1), dtype=int)
    else:
        # If more than two matrices, swap matrices stochastically with at least
        # one matrix being swapped.
        s = list(itertools.product(np.array([0, 1], dtype=int), repeat=n_sim_mat-1))
        # Drop case where no matrix has rows swapped.
        s = np.array(s[1:], dtype=int)
        n_outcome = s.shape[0]
        outcome_draw = np.random.randint(0, n_outcome, n_perm)
        is_swapped = s[outcome_draw, :]
        # Swap all matrices.
        # is_swapped = np.ones((n_perm, n_sim_mat - 1), dtype=int)

    # Flesh out matrices of indices detailing all permutations to be used.
    dmy_idx = np.arange(n_item, dtype=int)
    perm_list_all = []
    for i_sim_mat in range(n_sim_mat - 1):
        perm_list = np.tile(np.expand_dims(dmy_idx, axis=0), [n_perm + 1, 1])
        for i_perm in range(n_perm):
            if is_swapped[i_perm, i_sim_mat] == 1:
                old_value_0 = copy.copy(perm_list[i_perm + 1, selected_swap_idx[i_perm, 0]])
                old_value_1 = copy.copy(perm_list[i_perm + 1, selected_swap_idx[i_perm, 1]])
                perm_list[i_perm + 1, selected_swap_idx[i_perm, 0]] = old_value_1
                perm_list[i_perm + 1, selected_swap_idx[i_perm, 1]] = old_value_0
        perm_list_all.append(perm_list)
    # print("perm_list_all:",perm_list_all)

    # We store the correlation with correct ordering at the first idx, and the
    # correlations for the swapped ordering in the remaining indices.
    rho_array = np.zeros([n_perm + 1])
    # Correct alignment.
    rho_array[0] = alignment_score_multi(sim_mat_list, alignment_combos)
    # Permuted alignments.
    for perm_idx in range(n_perm):
        sim_mat_perm_list = []
        # Add unpermuted matrix.
        sim_mat_perm_list.append(sim_mat_list[0])
        # Add permuted matrices.
        for sim_mat_idx in range(n_sim_mat - 1):
            sim_mat_perm_list.append(
                symmetric_matrix_indexing(sim_mat_list[sim_mat_idx + 1], perm_list_all[sim_mat_idx][perm_idx + 1, :])
            )
        # Compute score
        rho_array[perm_idx + 1] = alignment_score_multi(
            sim_mat_perm_list, alignment_combos
        )
    return rho_array, perm_list_all

def permutation_analysis_sub(sim_mat_list, max_perm=10000, n_unknown=2, is_exact=False):
    """Perform random permutation analysis.

    In order to evaluate our ability to correctly align matrices, we
    first compute the correlation of the correct ordering and then
    compute the correlations for the sampled permutations. What we
    would like to see is that the correlation for the correct ordering
    is higher than any of the permuted alignments.
    """
    if is_exact:
        n_mismatch_thresh = 1
    else:
        n_mismatch_thresh = n_unknown

    #print("n_mismatch_thresh:",n_mismatch_thresh)

    n_sim_mat = len(sim_mat_list)
    n_item = sim_mat_list[0].shape[0]
    n_perm = np.minimum(max_perm, math.factorial(n_item))

    alignment_combos = list(itertools.combinations(np.arange(n_sim_mat, dtype=int), 2))
    ordered_idx = np.arange(n_item)
    # Initialize perm_list_all.
    perm_list_all = []
    for _ in range(n_sim_mat - 1):
        perm_list = np.zeros([n_perm + 1, n_item], dtype=int)
        perm_list[0, :] = np.arange(n_item)
        perm_list_all.append(perm_list)
    #print(perm_list_all)
    # Sample from possible permutations. There are 3,628,800 permutations of
    # ten elements (i.e., 10!), so it is very unlikely to get repeated samples.
    for i_perm in range(n_perm):
        # Randomly select indices that will be 'unknown'.
        sub_idx = np.random.permutation(n_item)
        sub_idx = sub_idx[0:n_unknown]
        sub_idx = np.sort(sub_idx)
        for i_sim_mat in range(n_sim_mat - 1):
            perm_idx = copy.copy(ordered_idx)
            aligned = True
            while aligned:
                # Permute indices selected as 'unknown'.
                perm_sub_idx = np.random.permutation(sub_idx)
                # Check number of mismatches.
                n_mismatch = np.sum(np.equal(sub_idx, perm_sub_idx))
                if i_sim_mat == 0:
                    # The first matrix must have at least the number of
                    # request mismatches.
                    if n_mismatch < n_mismatch_thresh:
                        aligned = False
                else:
                    # All other matrices don't matter.
                    aligned = False
            perm_idx[sub_idx] = perm_sub_idx
            perm_list_all[i_sim_mat][i_perm + 1, :] = perm_idx

    # We store the correlation with correct ordering at the first idx, and the
    # correlations for the swapped ordering in the remaining indices.
    rho_array = np.zeros([n_perm + 1])
    # Correct alignment.
    rho_array[0] = alignment_score_multi(sim_mat_list, alignment_combos)
    # Permuted alignments.
    for perm_idx in range(n_perm):
        sim_mat_perm_list = []
        # Add unpermuted matrix.
        sim_mat_perm_list.append(sim_mat_list[0])
        # Add permuted matrices.
        for sim_mat_idx in range(n_sim_mat - 1):
            sim_mat_perm_list.append(
                symmetric_matrix_indexing(sim_mat_list[sim_mat_idx + 1], perm_list_all[sim_mat_idx][perm_idx + 1, :])
            )
        # Compute score
        rho_array[perm_idx + 1] = alignment_score_multi(
            sim_mat_perm_list, alignment_combos
        )
    return rho_array, perm_list_all

def alignment_score_multi(sim_mat_list, alignment_combos):
    """"""
    score = 0
    weight = 1 / len(alignment_combos)
    for combo in alignment_combos:
        score = score + weight * alignment_score(
            sim_mat_list[combo[0]],
            sim_mat_list[combo[1]]
        )
    return score

def alignment_score(a, b, method='spearman'):
    """Return the alignment score between two similarity matrices.

    Assumes that matrix a is the smaller matrix and crops matrix b to
    be the same shape.
    """
    n_row = a.shape[0]
    b_cropped = b[0:n_row, :]
    b_cropped = b_cropped[:, 0:n_row]
    idx_upper = np.triu_indices(n_row, 1)

    if method == 'spearman':
        # Alignment score is the Spearman correlation coefficient.
        alignment_score, _ = spearmanr(a[idx_upper], b_cropped[idx_upper])
    else:
        raise ValueError(
            "The requested method '{0}'' is not implemented.".format(method)
        )
    return alignment_score

def symmetric_matrix_indexing(m, perm_idx):
    """Index matrix symmetrically.

    Can be used to symmetrically swap both rows and columns or to
    subsample.
    """
    m_perm = copy.copy(m)
    m_perm = m_perm[perm_idx, :]
    m_perm = m_perm[:, perm_idx]
    return m_perm

def infer_mistmatch(perm_list):
    """"""
    n_mismatch_list = []
    for perm_array in perm_list:
        (n_perm, n_class) = perm_array.shape
        n_mismatch = np.zeros((n_perm))
        true_idx = np.arange(n_class)
        for i_perm in range(n_perm):
            n_mismatch[i_perm] = np.sum(np.not_equal(true_idx, perm_array[i_perm]))
        n_mismatch_list.append(copy.copy(n_mismatch))
    return n_mismatch_list

def do_correlation(dataset,n_image):

    max_perm = 100
    n_perm_keep = 100

    # Load pickle data
    if dataset=="vrd_noun":
        samples = list(
            chain.from_iterable(pickle.load(open("../data/dumped_data/vrd_noun_25samples_10blocks.pkl", 'rb'))))
    elif dataset=="vrd_verb":
        samples = list(
            chain.from_iterable(pickle.load(open("../data/dumped_data/vrd_verb_25samples_10blocks.pkl", 'rb'))))
    elif dataset=="vg_noun":
        samples = list(
            chain.from_iterable(pickle.load(open("../data/dumped_data/vg_noun_freq_greater_than_10_25samples_10iter.pkl", 'rb'))))
    elif dataset=="vg_verb":
        samples = list(
            chain.from_iterable(pickle.load(open("../data/dumped_data/vg_verb_25samples_10iter.pkl", 'rb'))))
    else:
        print("error split!")
    sampled_embeddings = random.sample(samples, n_image)
    averaged_embeddings = average_multi(sampled_embeddings)
    z_0 = averaged_embeddings['z_0']
    z_1 = averaged_embeddings['z_1']

    set_size, n_mismatch_array, rho_array=permutation(z_0,z_1, max_perm=max_perm, n_perm_keep=n_perm_keep)

    return set_size, n_mismatch_array, rho_array

set_size, n_mismatch_array, rho_array=do_correlation("vrd_noun",1)