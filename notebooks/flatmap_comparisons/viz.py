from neuro.features.qa_questions import get_questions, get_merged_questions_v3_boostexamples
from neuro import analyze_helper, viz
from neuro.config import repo_dir
from sasc.config import FMRI_DIR, STORIES_DIR, RESULTS_DIR
from tqdm import tqdm
import sys
from ridge_utils.DataSequence import DataSequence
import pandas as pd
import os
import matplotlib.pyplot as plt
import cortex
import seaborn as sns
from os.path import join
from collections import defaultdict
import numpy as np
import sasc.viz
from sklearn.preprocessing import StandardScaler
import joblib
import dvu
import sys
sys.path.append('../notebooks')
flatmaps_per_question = __import__('06_flatmaps_per_question')


def compute_pvals(flatmaps_qa_list, frac_voxels_to_keep, corrs_gt_arr, flatmaps_null, mask_corrs=None):
    '''
    Params
    ------
    flatmaps_qa_list: list of D np.arrays
        each array is a flatmap of the same shape
    frac_voxels_to_keep: float
        fraction of voxels to keep
    corrs_gt_arr: np.array of size D
        array of ground truth correlations 
    flatmaps_null: np.ndarray
        (n_flatmaps, n_voxels) array of null flatmaps for a particular subject
        e.g. eng1000 weights
    mask_corrs: np.ndarray
        if passed, use this as mask rather than mask extreme
    '''
    # print(eng1000_dir)
    # flatmaps_eng1000 = joblib.load(eng1000_dir)
    pvals = []
    baseline_distr = []

    for i in tqdm(range(len(flatmaps_qa_list))):
        if frac_voxels_to_keep < 1:

            # mask based on corrs
            if mask_corrs is not None:
                mask = (mask_corrs > np.percentile(
                    mask_corrs, 100 * (1 - frac_voxels_to_keep))).astype(bool)
            else:
                # mask based on extreme values
                mask = np.abs(flatmaps_qa_list[i]) >= np.percentile(
                    np.abs(flatmaps_qa_list[i]), 100 * (1 - frac_voxels_to_keep))

        else:
            mask = np.ones_like(flatmaps_qa_list[i]).astype(bool)
        # if frac_voxels_to_keep < 1:
        #     mask_extreme = np.abs(flatmaps_qa_list[i]) >= np.percentile(
        #         np.abs(flatmaps_qa_list[i]), 100 * (1 - frac_voxels_to_keep))
        # else:
        #     mask_extreme = np.ones(flatmaps_qa_list[i].shape).astype(bool)

        flatmaps_null_masked = flatmaps_null[:, mask]
        flatmaps_qa_masked = flatmaps_qa_list[i][mask]

        # calculate correlation between each row of flatmaps_qa_masked and flatmaps_eng1000_masked
        flatmaps_null_masked_norm = StandardScaler(
        ).fit_transform(flatmaps_null_masked.T).T
        flatmaps_qa_masked_norm = (
            flatmaps_qa_masked - flatmaps_qa_masked.mean()) / flatmaps_qa_masked.std()
        corrs_perm_eng100_arr = flatmaps_null_masked_norm @ flatmaps_qa_masked_norm / \
            flatmaps_qa_masked_norm.shape[0]
        pvals.append((corrs_perm_eng100_arr > corrs_gt_arr[i]).mean())
        baseline_distr.append(corrs_perm_eng100_arr)
    return pvals, baseline_distr


def _calc_corrs(flatmaps_qa, flatmaps_gt, titles_qa, titles_gt, preproc=None):
    if preproc is not None:
        if preproc == 'quantize':
            # bin into n bins with equal number of samples
            n_bins = 10

    corrs = pd.DataFrame(
        np.zeros((len(flatmaps_qa), len(flatmaps_gt))),
        index=titles_qa,
        columns=titles_gt,
        # [f'{bd_list[i][0]}_{bd_list[i][1]}'.replace('_qa', '') for i in range(len(bd_list))],
        # index=df_pairs['qa_weight'].astype(str),
    )
    for i, qa in enumerate(flatmaps_qa):
        for j, bd in enumerate(flatmaps_gt):
            corrs.iloc[i, j] = np.corrcoef(
                flatmaps_qa[i], flatmaps_gt[j])[0, 1]

    return corrs


def _heatmap(corrs, out_dir_save):
    os.makedirs(out_dir_save, exist_ok=True)
    # normalize each column
    # corrs = corrs / corrs.abs().max()
    # normalize each row to mean zero stddev 1
    # corrs = (corrs - corrs.mean()) / corrs.std()
    # plt.figure(figsize=(20, 10))
    vmax = np.max(np.abs(corrs.values))
    # sns.clustermap(corrs, annot=False, cmap='RdBu', vmin=-vmax, vmax=vmax, dendrogram_ratio=0.01)

    sns.heatmap(corrs, annot=False, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    # plt.imshow(corrs, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    # plt.xticks(range(len(bd_list)), [f'{bd[0]}' for bd in bd_list], rotation=90)
    # plt.yticks(range(len(qa_list)), qa_list)
    # plt.colorbar()
    dvu.outline_diagonal(corrs.values.shape, roffset=0.5, coffset=0.5)
    plt.savefig(join(out_dir_save, 'corrs_heatmap.pdf'),
                bbox_inches='tight')


def corr_bars(corrs: np.ndarray, questions, out_dir_save, xlab: str = '', color='C0', label=None):
    os.makedirs(out_dir_save, exist_ok=True)
    # print(out_dir_save)
    # mask = args0['corrs_test'] >= 0
    # wt_qas = [wt_qa[mask] for wt_qa in wt_qas]
    # wt_bds = [wt_bd[mask] for wt_bd in wt_bds]

    # barplot of diagonal
    def _get_func_with_perm(x1, x2, n=10, perc=95):
        corr = np.corrcoef(x1, x2)[0, 1]
        corrs_perm = [
            np.corrcoef(np.random.permutation(x1), x2)[0, 1]
            for i in range(n)
        ]
        # print(corrs_perm)
        return corr, np.percentile(corrs_perm, 50-perc/2), np.percentile(corrs_perm, 50+perc/2)

    # corrs_mean = []
    # corrs_err = []
    # for i in tqdm(range(len(corrs.columns))):
    #     corr, corr_interval_min, corr_interval_max = _get_func_with_perm(
    #         flatmaps_qa[i], flatmaps_gt[i])
    #     corrs_mean.append(corr)
    #     corrs_err.append((corr_interval_max - corr_interval_min)/2)
    # sns.barplot(y=corrs.columns, x=np.diag(corrs), color='gray')
    plt.grid(alpha=0.2)
    # corrs_diag = np.diag(corrs)
    # idx_sort = np.argsort(corrs_diag)[::-1]
    # idx_sort = np.arange(len(questions))
    # print('idx_sort', idx_sort, corrs_diag[idx_sort])
    plt.errorbar(
        x=corrs,
        y=np.arange(len(questions)),
        # xerr=corrs_err,
        fmt='o',
        color=color,
        label=label + f' (mean {corrs.mean():.3f})',
    )

    plt.yticks(np.arange(len(questions)), questions)
    plt.axvline(0, color='gray')
    plt.axvline(corrs.mean(), color=color, linestyle='--')
    # plt.title(f'{setting} mean {np.diag(corrs).mean():.3f}')
    # annotate line with mean value
    # plt.text(np.diag(corrs).mean(), 0.1,
    #  f'{np.diag(corrs).mean():.3f}', ha='left', color=color)
    plt.xlabel(xlab)
