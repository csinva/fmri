from cortex import mni
import neurosynth
import viz
from neurosynth import term_dict, term_dict_rev, get_neurosynth_flatmaps
# from neuro import viz
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


def plot_corrs_df(corrs_df, frac_voxels_to_keep, out_dir):

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

        print(f'{subject} corr',
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
    plt.savefig(join(out_dir, f'corrs_{frac_voxels_to_keep}.png'), dpi=300)


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


def convert_to_mni_space(flatmap_arr, subject='UTS01'):
    flatmap_vol = cortex.Volume(
        data=flatmap_arr.flatten(), subject=subject, xfmname=f'{subject}_auto')
    flatmap_to_mni_cached = cortex.db.get_mnixfm(subject, f'{subject}_auto')
    mni_vol = mni.transform_to_mni(flatmap_vol, flatmap_to_mni_cached)
    mni_arr = mni_vol.get_fdata()  # the actual array, shape=(182,218,182)
    return mni_arr


def linearly_downsample_with_interpolation(arr, factor=2):
    from scipy.ndimage import zoom
    return zoom(arr, 1/factor, order=1)


def compute_mni_corr_df(flatmaps_qa_dicts_by_subject, flatmaps_gt_dict_mni, qs):
    flatmaps_gt_list_mni = [flatmaps_gt_dict_mni[q] for q in qs]
    subjects = list(flatmaps_qa_dicts_by_subject.keys())
    flatmaps_qa_dict_list_subjects = {subject: [flatmaps_qa_dicts_by_subject[subject][q] for q in qs]
                                      for subject in subjects}
    # flatmaps_gt_dict_list_subject = {subject: get_neurosynth_flatmaps(subject)
    #  for subject in ['UTS01', 'UTS02', 'UTS03']}
    flatmaps_qa_dict_list_subjects_mni = {subject: [convert_to_mni_space(flatmaps_qa_dict_list_subjects[subject][i], subject=subject)
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
                np.corrcoef(linearly_downsample_with_interpolation(flatmap_qa_mni).flatten(), flatmap_gt_mni.flatten())[0, 1])

        flatmap_avg_mni /= len(subjects)
        flatmap_avg_mni = linearly_downsample_with_interpolation(
            flatmap_avg_mni)
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


if __name__ == '__main__':
    # setting = 'shapley_neurosynth'
    # setting = 'full_neurosynth'
    # setting = 'individual_gpt4'
    for settings in [
        ['full_neurosynth_pc'], ['full_neurosynth_wordrate_pc'], [
            'full_35_pc'], ['full_35_wordrate_pc'],
        ['full_neurosynth'], ['full_neurosynth_wordrate'], [
            'full_35'], ['full_35_wordrate'],
    ]:
        print('settings', settings)
        # settings = ['']  # shapley_neurosynth, individual_gpt4
        subjects = ['UTS01', 'UTS02', 'UTS03']
        # subjects = [f'UTS0{i}' for i in range(1, 9)]

        # comparison hyperparams
        apply_mask = True
        frac_voxels_to_keep = 0.1  # 0.10
        frac_voxels_to_keep_list = [frac_voxels_to_keep]
        # hyperparams
        out_dir = join(repo_dir, 'qa_results',
                       'neurosynth_compare', '___'.join(settings))
        os.makedirs(out_dir, exist_ok=True)

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

        plot_corrs_df(corrs_df, frac_voxels_to_keep, out_dir)

        # pvals_subject = compute_pvals_for_subject(
        # corrs_df, 'UTS01', frac_voxels_to_keep_list)
        # pvals_subject.style.background_gradient().format(precision=3)

        corrs_df_mni = compute_mni_corr_df(
            flatmaps_qa_dicts_by_subject, flatmaps_gt_dict_mni, qs)
        print('avg', corrs_df_mni.loc['avg'])
        corrs_df_mni.to_pickle(join(out_dir, 'corrs_df_mni.pkl'))
        corrs_df_mni.style.background_gradient(axis=None, cmap="coolwarm_r", vmin=-
                                               corrs_df_mni.abs().max().max(), vmax=corrs_df_mni.abs().max().max()).format(precision=3).to_html(
            join(out_dir, 'corrs_df_mni.html'))
