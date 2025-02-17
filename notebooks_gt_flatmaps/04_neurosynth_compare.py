from cortex import mni
import neurosynth
import viz
from neurosynth import term_dict, term_dict_rev, get_neurosynth_flatmaps
# from neuro import viz
from neuro.config import repo_dir, PROCESSED_DIR
from neuro import analyze_helper
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


def compute_corrs_df(qs, frac_voxels_to_keep, subjects, flatmaps_qa_dicts_by_subject, apply_mask):
    '''Compute correlations between QA flatmaps and GT flatmaps (with the function loads)
    '''
    corrs_df_list = defaultdict(list)
    for subject in tqdm(subjects):
        flatmaps_gt_dict = get_neurosynth_flatmaps(subject, mni=False)
        # flatmaps_gt_dict = {}
        # flatmaps_gt_dict_mni = get_neurosynth_flatmaps(subject, mni=True)
        # for k in tqdm(flatmaps_gt_dict_mni.keys()):
        #     mni_vol = cortex.Volume(
        #         flatmaps_gt_dict_mni[k], "fsaverage", "atlas_2mm")
        #     subj_vol, subj_arr = neurosynth.mni_vol_to_subj_vol_surf(
        #         mni_vol, subject=subject.replace('UT', ''))
        #     flatmaps_gt_dict[k] = subj_arr

        # flatmaps_gt_dict = get_neurosynth_flatmaps(subject, mni=True)
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
            assert k in flatmaps_qa_dict_masked, f'{k} not in flatmaps_qa_dict_masked'
            assert k in flatmaps_gt_masked, f'{k} not in flatmaps_gt_masked'
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


def plot_corrs_df(
        corrs_df, out_dir, plot_val=f'corrs_0.1',
        xlab='Flatmap correlation'
):

    c = corrs_df
    colors = {
        'UTS01': 'C0',
        'UTS02': 'C1',
        'UTS03': 'C2',
        'S01': 'C0',
        'S02': 'C1',
        'S03': 'C2',
        'mean': 'black'
    }

    d_mean = pd.DataFrame(c.groupby('questions')[
        plot_val].mean()).reset_index()
    d_mean['subject'] = 'mean'
    c = pd.concat([c, d_mean])
    c = c.set_index('questions')

    # ['mean', 'UTS01', 'UTS02', 'UTS03']:
    subjects = sorted(c.subject.unique())
    # move 'mean' to be first
    subjects = ['mean'] + [s for s in subjects if s != 'mean']

    for subject in subjects:
        corrs_df_subject = c[c['subject'] == subject]
        if subject == 'mean':
            idx_sort = corrs_df_subject[plot_val].sort_values(
                ascending=False).index
        # print(corrs_df_subject.index)
        corrs_df_subject = corrs_df_subject.loc[idx_sort]

        # plot corrs
        if subject == 'mean':
            plt.errorbar(
                corrs_df_subject[plot_val],
                range(len(corrs_df_subject)),
                color='black',
                fmt='o',
                zorder=1000,
                label=subject.capitalize(),
            )
        else:
            plt.errorbar(
                corrs_df_subject[plot_val],
                range(len(corrs_df_subject)),
                # xerr=np.sqrt(
                # r_df[plot_val] * (1-r_df[plot_val])
                # / r_df['num_test']),
                alpha=0.5,
                label=subject.upper(),
                fmt='o')
        plt.axvline(corrs_df_subject[plot_val].mean(),
                    linestyle='--',
                    color=colors[subject],
                    zorder=-1000)

        print(f'{subject} corr',
              corrs_df_subject[plot_val].mean())

    # add horizontal bars
    # if ylabels is None:
        # plt.yticks(range(len(corrs_df_subject)), [
        # term_dict_rev[k] for k in idx_sort])
        # corrs_df_subject.index)
    ylabels = [analyze_helper.abbrev_question(q) for q in idx_sort]
    plt.yticks(range(len(corrs_df_subject)), ylabels)
    # else:
    # this raises issues! things get sorted!
    # plt.yticks(range(len(corrs_df_subject)), ylabels)
    plt.xlabel(xlab)
    plt.grid(axis='y', alpha=0.2)
    plt.axvline(0, color='gray')

    abs_lim = max(np.abs(plt.xlim()))
    # plt.xlim(-abs_lim, abs_lim)

    # annotate with baseline and text label
    plt.legend()
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(join(out_dir, plot_val + '.png'), dpi=300)


def compute_pvals_for_subject(corrs_df, flatmaps_qa_dicts_by_subject, subject, frac_voxels_to_keep_list):
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


def convert_to_mni_space(flatmap_arr, subject='UTS01'):
    flatmap_vol = cortex.Volume(
        data=flatmap_arr.flatten(), subject=subject, xfmname=f'{subject}_auto')
    flatmap_to_mni_cached = cortex.db.get_mnixfm(subject, f'{subject}_auto')
    mni_vol = mni.transform_to_mni(
        flatmap_vol, flatmap_to_mni_cached, template='MNI152_T1_2mm_brain.nii.gz')
    mni_arr = mni_vol.get_fdata()  # the actual array, shape=(91, 109, 91)
    return mni_arr


# def linearly_downsample_with_interpolation(arr, factor=2):
#     from scipy.ndimage import zoom
#     return zoom(arr, 1/factor, order=1)


def compute_mni_corr_df(flatmaps_qa_dicts_by_subject, flatmaps_gt_dict_mni, qs):
    flatmaps_gt_list_mni = [flatmaps_gt_dict_mni[q] for q in qs]
    subjects = list(flatmaps_qa_dicts_by_subject.keys())
    flatmaps_qa_dict_list_subjects = {subject: [flatmaps_qa_dicts_by_subject[subject][q] for q in qs]
                                      for subject in subjects}
    # flatmaps_gt_dict_list_subject = {subject: get_neurosynth_flatmaps(subject)
    #  for subject in ['UTS01', 'UTS02', 'UTS03']}
    flatmaps_qa_dict_list_subjects_mni = {
        subject: [convert_to_mni_space(flatmaps_qa_dict_list_subjects[subject][i], subject=subject)
                  for i in tqdm(range(len(qs)))]
        for subject in subjects}

    corrs_avg_df = defaultdict(list)
    for question_idx in tqdm(range(len(qs))):

        # subjects = ['UTS01', 'UTS02', 'UTS03']
        # subjects = [f'UTS0{k}' for k in range(1, 9)]
        flatmap_avg_mni = None
        for subject in subjects:
            # flatmap_qa_arr = flatmaps_qa_dict_list_subjects[subject][question_idx]
            # flatmap_qa_mni = convert_to_mni_space(flatmap_qa_arr, subject=subject)
            flatmap_qa_mni = flatmaps_qa_dict_list_subjects_mni[subject][question_idx]
            if flatmap_avg_mni is None:
                flatmap_avg_mni = flatmap_qa_mni
            else:
                flatmap_avg_mni += flatmap_qa_mni

            # flatmap_avg_mini = flatmap_avg_mni[::2, ::2, ::2]

            # flatmap_gt_arr = flatmaps_gt_dict_list_subject[subject][qs[question_idx]]
            # flatmap_gt_mni = convert_to_mni_space(flatmap_gt_arr, subject=subject)
            # flatmap_gt_mni = flatmaps_gt_dict_list_subject_mni[subject][question_idx]
            flatmap_gt_mni = flatmaps_gt_list_mni[question_idx]

            # print('corr', np.corrcoef(flatmap_qa_arr.flatten(),
            #   flatmap_gt_arr.flatten())[0, 1])
            corrs_avg_df[f'corr_{subject}'].append(
                np.corrcoef(flatmap_qa_mni.flatten(), flatmap_gt_mni.flatten())[0, 1])

        flatmap_avg_mni /= len(subjects)
        # flatmap_avg_mni = linearly_downsample_with_interpolation(
        # flatmap_avg_mni)
        corrs_avg_df['questions'].append(qs[question_idx])
        # print('shapes', flatmap_avg_mni.shape, flatmap_gt_mni.shape)
        corr_avg = np.corrcoef(flatmap_avg_mni.flatten(),
                               flatmap_gt_mni.flatten())[0, 1]
        corrs_avg_df['corr_avg'].append(corr_avg)
        # corrs_avg_df['flatmap_qa'].append(flatmap_avg_mni)
    df = pd.DataFrame(corrs_avg_df).sort_values(
        'corr_avg', ascending=False).set_index('questions')
    # add avg row to bottom
    df.loc['avg'] = df.mean()
    return df
