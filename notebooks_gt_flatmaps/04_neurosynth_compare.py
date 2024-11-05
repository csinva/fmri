from cortex import mni
import neurosynth
import viz
from neurosynth import term_dict, term_dict_rev, get_neurosynth_flatmaps
from neuro import viz
from neuro.config import repo_dir, PROCESSED_DIR
import sys
import pandas as pd
import os
import matplotlib.pyplot as plt
import cortex
from os.path import join
from collections import defaultdict
import numpy as np
import joblib
from tqdm import tqdm
import sys
from copy import deepcopy
sys.path.append('../notebooks')
os.environ["FSLDIR"] = "/home/chansingh/fsl"


def compute_corrs_df(frac_voxels_to_keep, subjects, flatmaps_qa_dicts_by_subject, apply_mask):
    corrs_df_list = defaultdict(list)
    for subject in tqdm(subjects):
        flatmaps_gt_dict = get_neurosynth_flatmaps(subject, mni=False)
        flatmaps_qa_dict = flatmaps_qa_dicts_by_subject[subject]

        if apply_mask:
            corrs_test = joblib.load(join(PROCESSED_DIR, subject.replace(
                'UT', ''), 'corrs_test_35.pkl')).values[0]
            # threshold
            if frac_voxels_to_keep < 1:
                corrs_test_mask = (corrs_test > np.percentile(
                    corrs_test, 100 * (1 - frac_voxels_to_keep))).astype(bool)
            else:
                corrs_test_mask = np.ones_like(corrs_test).astype(bool)
            flatmaps_qa_dict_masked = {k: flatmaps_qa_dict[k][corrs_test_mask]
                                       for k in flatmaps_qa_dict.keys()}
            flatmaps_gt_masked = {k: flatmaps_gt_dict[k][corrs_test_mask]
                                  for k in flatmaps_gt_dict.keys()}

        # get common flatmaps and put into d
        # common_keys = set(flatmaps_gt_masked.keys()) & set(
            # flatmaps_qa_dict_masked.keys())
        d = defaultdict(list)
        for k in qs:
            d['questions'].append(k)
            d['corr'].append(np.corrcoef(flatmaps_qa_dict_masked[k],
                                         flatmaps_gt_masked[k])[0, 1])
            d['flatmap_qa'].append(flatmaps_qa_dict_masked[k])
            d['flatmap_neurosynth'].append(flatmaps_gt_masked[k])
        d = pd.DataFrame(d).sort_values('corr', ascending=False)

        corrs = viz._calc_corrs(
            d['flatmap_qa'].values,
            d['flatmap_neurosynth'].values,
            titles_qa=d['questions'].values,
            titles_gt=d['questions'].values,
        )

        corrs_df_list[f'corrs_{frac_voxels_to_keep}'].extend(
            np.diag(corrs).tolist())
        corrs_df_list['questions'].extend(d['questions'].values.tolist())
        corrs_df_list['subject'].extend([subject] * len(d['questions'].values))

        # viz.corr_bars(
        #     corrs,
        #     out_dir_save=join(repo_dir, 'qa_results', 'neurosynth', setting),
        #     xlab='Neurosynth',
        # )

        # save flatmaps
        # for i in tqdm(range(len(d))):
        #     sasc.viz.quickshow(
        #         d.iloc[i]['flatmap_qa'],
        #         subject=subject,
        #         fname_save=join(repo_dir, 'qa_results', 'neurosynth', subject,
        #                         setting, f'{d.iloc[i]["questions"]}.png')
        #     )

        #     sasc.viz.quickshow(
        #         d.iloc[i]['flatmap_neurosynth'],
        #         subject=subject,
        #         fname_save=join(repo_dir, 'qa_results', 'neurosynth', subject,
        #                         'neurosynth', f'{d.iloc[i]["questions"]}.png')
        #     )

    corrs_df = pd.DataFrame(corrs_df_list)
    # corrs_df.to_pickle(join(repo_dir, 'qa_results',
    #    'neurosynth', setting + '_corrs_df.pkl'))
    return corrs_df


def plot_corrs_df(corrs_df, setting):

    c = corrs_df
    xlab = f'Flatmap correlation (Top-{int(100*frac_voxels_to_keep)}% best-predicted voxels)'
    plt.figure(figsize=(7, 5))
    # colors = {
    #     'UTS01': 'C0',
    #     'UTS02': 'C1',
    #     'UTS03': 'C2',
    #     'mean': 'black'
    # }

    d_mean = pd.DataFrame(c.groupby('questions')[
        f'corrs_{frac_voxels_to_keep}'].mean()).reset_index()
    d_mean['subject'] = 'mean'
    c = pd.concat([c, d_mean])
    c = c.set_index('questions')

    for subject in ['mean', 'UTS01', 'UTS02', 'UTS03']:
        corrs_df_subject = c[c['subject'] == subject]
        if subject == 'mean':
            idx_sort = corrs_df_subject[f'corrs_{frac_voxels_to_keep}'].sort_values(
                ascending=False).index
        corrs_df_subject = corrs_df_subject.loc[idx_sort]

        # plot corrs
        if subject == 'mean':
            plt.errorbar(
                corrs_df_subject[f'corrs_{frac_voxels_to_keep}'],
                range(len(corrs_df_subject)),
                color='black',
                fmt='o',
                zorder=1000,
                label=subject.capitalize(),
            )
        else:
            plt.errorbar(
                corrs_df_subject[f'corrs_{frac_voxels_to_keep}'],
                range(len(corrs_df_subject)),
                # xerr=np.sqrt(
                # r_df[f'corrs_{frac_voxels_to_keep}'] * (1-r_df[f'corrs_{frac_voxels_to_keep}'])
                # / r_df['num_test']),
                alpha=0.5,
                label=subject.upper(),
                fmt='o')
        plt.axvline(corrs_df_subject[f'corrs_{frac_voxels_to_keep}'].mean(),
                    linestyle='--',
                    # color=colors[subject],
                    zorder=-1000)

        print('mean corr',
              corrs_df_subject[f'corrs_{frac_voxels_to_keep}'].mean())

    # add horizontal bars
    plt.yticks(range(len(corrs_df_subject)), [
               term_dict_rev[k] for k in idx_sort])
    plt.xlabel(xlab, fontsize='large')
    plt.grid(axis='y', alpha=0.2)
    plt.axvline(0, color='gray')

    abs_lim = max(np.abs(plt.xlim()))
    plt.xlim(-abs_lim, abs_lim)

    # annotate with baseline and text label
    plt.legend(fontsize='large')
    plt.tight_layout()
    plt.savefig(join(repo_dir, 'qa_results',
                'neurosynth', 'corrs_' + setting + '.png'), dpi=300)


def compute_pvals_for_subject(corrs_df, subject, frac_voxels_to_keep_list):
    corrs_df_subject = corrs_df[corrs_df['subject']
                                == subject].set_index('questions')

    # corrs_df = pd.DataFrame(corrs_df_dict)
    flatmaps_qa_list_subject = [flatmaps_qa_dicts_by_subject[subject][q]
                                for q in corrs_df_subject.index]
    for frac_voxels_to_keep in tqdm(frac_voxels_to_keep_list):
        eng1000_dir = join(PROCESSED_DIR, subject.replace(
            'UT', ''), 'eng1000_weights.pkl')
        pvals = viz.compute_pvals(flatmaps_qa_list_subject, frac_voxels_to_keep,
                                  corrs_df_subject[f'corrs_{frac_voxels_to_keep}'].values, eng1000_dir=eng1000_dir)

        # get what fraction of 'corrs_perm_eng1000' column is greater than f'corrs_{frac_voxels_to_keep}'
        corrs_df_subject[f'pval_{frac_voxels_to_keep}'] = pvals

    # format scientific notation
    corrs_df_subject.sort_values(
        by=f'pval_{frac_voxels_to_keep}')
    return corrs_df_subject


if __name__ == '__main__':
    # setting = 'shapley_neurosynth'
    # setting = 'full_neurosynth'
    # setting = 'individual_gpt4'
    settings = ['full_neurosynth']  # shapley_neurosynth, individual_gpt4
    setting = settings[0]
    # subjects = ['UTS01', 'UTS02', 'UTS03']
    subjects = [f'UTS0{i}' for i in range(1, 9)]

    # comparison hyperparams
    apply_mask = True
    frac_voxels_to_keep = 0.1  # 0.10
    frac_voxels_to_keep_list = [frac_voxels_to_keep]
    # hyperparams

    # load flatmaps
    flatmaps_qa_dicts_by_subject = neurosynth.load_flatmaps_qa_dicts_by_subject(
        subjects, settings)

    # flatmaps_gt_dict_list_subject_mni = {subject: [convert_to_mni_space(flatmaps_gt_dict_list_subject[subject][qs[i]], subject=subject)
    #                                                for i in tqdm(range(len(qs)))]
    #                                      for subject in ['UTS01', 'UTS02', 'UTS03']}
    flatmaps_gt_dict_mni = neurosynth.get_neurosynth_flatmaps(mni=True)
    qs = list(set(flatmaps_qa_dicts_by_subject['UTS01'].keys()) & set(
        flatmaps_gt_dict_mni.keys()))

    corrs_df = compute_corrs_df(
        frac_voxels_to_keep, subjects, flatmaps_qa_dicts_by_subject, apply_mask)

    plot_corrs_df(corrs_df, setting)

    pvals_subject = compute_pvals_for_subject(
        corrs_df, 'UTS01', frac_voxels_to_keep_list)
    pvals_subject.style.background_gradient().format(precision=3)
