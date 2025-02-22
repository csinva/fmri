import warnings
from copy import deepcopy
from neuro.config import repo_dir, PROCESSED_DIR, setup_freesurfer
from neuro.analyze_helper import abbrev_question
import matplotlib.image as mpimg
import sasc.viz
import dvu
from neuro import analyze_helper
from sklearn.linear_model import LogisticRegressionCV
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import balanced_accuracy_score
from os.path import join
from collections import defaultdict
import joblib
import numpy as np
from neuro.features.questions.gpt4 import QS_35_STABLE
import seaborn as sns
import neuro.config
from neuro.features.stim_utils import load_story_wordseqs, load_story_wordseqs_huge
from neuro.data import story_names, response_utils
from neuro.features import qa_questions, feature_spaces
from tqdm import tqdm
import os
from os.path import dirname
import matplotlib.pyplot as plt
import pandas as pd
from ridge_utils.DataSequence import DataSequence
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pickle")
dvu.set_style()
data_dir = join(neuro.config.repo_dir, 'data', 'decoding')


def get_fmri_and_labs(data_dir, story_name='onapproachtopluto', train_or_test='test', subject='uts03', mask_subject=None):
    '''
    Returns
    -------
    df : pd.DataFrame
        The fMRI features, with columns corresponding to the principal components
        of the fMRI data.
    labs : pd.DataFrame
        Binary labeled annotations for each of the texts
    texts: 
        The texts corresponding to the rows of df
    '''
    df = joblib.load(
        join(data_dir, subject, train_or_test, story_name + '.pkl'))

    if mask_subject is not None:
        pca_comps = joblib.load(
            join(data_dir, f'{subject}/pca_components.pkl'))
        voxel_vals = df.values @ pca_comps
        voxel_vals_masked = voxel_vals * mask_subject
        # df.values = voxel_vals_masked @ pca_comps.T
        df = pd.DataFrame(
            voxel_vals_masked @ pca_comps.T, columns=df.columns, index=df.index
        )

    dfs = []
    for offset in [1, 2, 3, 4]:
        # for offset in [1,]:
        df_offset = df.shift(-offset)
        df_offset.columns = [col + f'_{offset}' for col in df.columns]
        dfs.append(df_offset)
    df = pd.concat(dfs, axis=1)  # .dropna()  # would want to dropna here

    # load labels
    labs = joblib.load(
        join(data_dir, 'labels', train_or_test, story_name + '_labels.pkl'))

    # drop rows with nans
    idxs_na = df.isna().sum(axis=1).values > 0
    df = df[~idxs_na]
    labs = labs[~idxs_na]
    texts = pd.Series(df.index)
    return df, labs, texts


def concatenate_running_texts(texts, frac=1/2):
    '''When decoding, you might want to concatenate 
    the text of the current and surrounding texts
    to deal with the temporal imprecision of the fMRI signal.
    '''
    texts_before = (
        texts.shift(1)
        .str.split().apply(  # only keep second half of words
            lambda l: ' '.join(l[int(-len(l) * frac):]) if l else '')
    )

    texts_after = (
        texts.shift(-1)
        .str.split().apply(  # only keep first half of words
            lambda l: ' '.join(l[:int(len(l) * frac)]) if l else '')
    )

    return texts_before + ' ' + texts + ' ' + texts_after


def load_data_by_subject(data_dir, mask_per_subject_dict=None, subjects=['uts01', 'uts02', 'uts03']):
    data_by_subject = {}
    for subject in subjects:
        data = defaultdict(list)
        for train_or_test in ['test', 'train']:
            story_names_list = os.listdir(
                join(data_dir, subject, train_or_test))
            for story_name in story_names_list:
                if mask_per_subject_dict is not None:
                    mask_subject = mask_per_subject_dict[subject]
                else:
                    mask_subject = None
                df, labs, texts = get_fmri_and_labs(
                    data_dir, story_name.replace(
                        '.pkl', ''), train_or_test, subject,
                    mask_subject=mask_subject
                )
                data['df_' + train_or_test].append(df)
                data['labs_' + train_or_test].append(labs)
                data['texts_' + train_or_test].append(texts)

        for k in data:
            data[k] = pd.concat(data[k], axis=0)
        data_by_subject[subject] = data
    return data_by_subject


def _keep_only_few_coefs(X, num_coefs_to_keep, num_features=200, num_delays=4):
    if num_coefs_to_keep is None:
        return X
    else:
        idxs_to_keep = np.arange(num_coefs_to_keep)
        idxs_to_keep = np.concatenate(
            [idxs_to_keep + i * num_features for i in range(num_delays)])
        return X[:, idxs_to_keep]


def fit_and_evaluate(data, subject, num_coefs_to_keep=None, label_num=None):
    # example fit linear decoder
    r = defaultdict(list)
    if label_num is not None:
        label_nums = [label_num]
    else:
        label_nums = range(data['labs_train'].shape[1])
    for label_num in tqdm(label_nums):
        X_train, y_train = data['df_train'].values, data['labs_train'].values[:, label_num]
        X_test, y_test = data['df_test'].values, data['labs_test'].values[:, label_num]
        print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

        # balance the binary class imbalance
        try:
            rus = RandomUnderSampler(random_state=42)
            X_train, y_train = rus.fit_resample(X_train, y_train)
            X_test, y_test = rus.fit_resample(X_test, y_test)

            # limit number of feats to keep?
            if num_coefs_to_keep is not None:
                X_train = _keep_only_few_coefs(X_train, num_coefs_to_keep)
                X_test = _keep_only_few_coefs(X_test, num_coefs_to_keep)

            if len(y_test) < 30:
                print('too few positive labels', label_num)
                continue

            print('label', label_num,
                  data['labs_train'].columns[label_num], X_train.shape, X_test.shape)
            m = LogisticRegressionCV(random_state=42)
            m.fit(X_train, y_train)
            test_acc = m.score(X_test, y_test)
            print(
                f"""\ttest acc {test_acc:.3f}""")  # \n\tnaive acc {1 -y_test.mean():.3f}""")
            r['label'].append(data['labs_train'].columns[label_num])
            # y_pred = m.predict(X_test)
            # balanced_accuracy_score(y_test, y_pred))
            r['test_acc'].append(test_acc)
            r['num_test'].append(len(y_test))
            r['coef'].append(m.coef_.copy())

            # extra test data from another subject ##########
            '''
            test_acc_ood = []
            for subject_ood in ['uts01', 'uts02', 'uts03']:
                if subject_ood == subject:
                    continue
                X_test_ood, y_test_ood = data_by_subject[subject_ood][
                    'df_test'].values, data_by_subject[subject_ood]['labs_test'].values[:, label_num]

                # balance the binary class imbalance
                rus = RandomUnderSampler(random_state=42)
                X_test_ood, y_test_ood = rus.fit_resample(
                    X_test_ood, y_test_ood)

                X_test_ood = _keep_only_few_coefs(
                    X_test_ood, num_coefs_to_keep)

                test_acc_ood_subject = m.score(X_test_ood, y_test_ood)
                test_acc_ood.append(test_acc_ood_subject)
            r['test_acc_ood'].append(np.mean(test_acc_ood))

            print(f"""\ttest acc ood {np.mean(test_acc_ood):.3f}""")
            '''
            ###############################################

        except Exception as e:
            print(e)
            print('error for', label_num)
            continue
    r_df = pd.DataFrame(r)
    return r_df


def _load_result(subject):
    if subject == 'mean':
        dfs = [pd.read_pickle(join(data_dir, 'test_acc', f'r_df_{subject}.pkl'))
               for subject in ['uts01', 'uts02', 'uts03']]
        return pd.concat(dfs, axis=0).groupby('label').mean().reset_index()
    else:
        return pd.read_pickle(join(data_dir, 'test_acc', f'r_df_{subject}.pkl'))


if __name__ == '__main__':
    # fit decoding
    '''
    data_by_subject = load_data_by_subject(data_dir)
    for subject in ['uts01', 'uts02', 'uts03'][::-1]:
        data = data_by_subject[subject]
        # r_df = fit_and_evaluate(data, subject)
        # r_df.to_pickle(join(data_dir, f'r_df_{subject}.pkl'))
    '''

    # refit ablated data
    for n_voxels_keep in [4, 20, 100, 500, 2500, 12500]:
        for suffix in ['', '_shuffle']:
            for subject in ['S01', 'S02', 'S03']:
                subject_dec = 'ut' + subject.lower()
                out_file = join(
                    data_dir, 'test_acc', f'r_df_{subject_dec}_nvoxels={n_voxels_keep}{suffix}.pkl')
                if os.path.exists(out_file):
                    print('exists', out_file)
                    continue

                # subject = 'uts03'
                pca_comps = joblib.load(
                    join(data_dir, f'{subject_dec}/pca_components.pkl'))
                # vertically stack pca_comps 4 times
                pca_comps = np.vstack([pca_comps]*4)
                r_df = _load_result(subject_dec).set_index(
                    'label').loc[QS_35_STABLE]
                rows = []

                for i, q in enumerate(tqdm(QS_35_STABLE)):
                    mask_per_subject_dict = {}
                    pc_coef = r_df.iloc[i]['coef']
                    flatmap_decode = (pc_coef @ pca_comps).squeeze()

                    # keep only largest n_voxels_keep value
                    mask_per_subject_dict[subject_dec] = np.zeros_like(
                        flatmap_decode).astype(bool)
                    mask_per_subject_dict[subject_dec][
                        np.argsort(flatmap_decode)[-n_voxels_keep:]] = True
                    if suffix == '_shuffle':
                        mask_per_subject_dict[subject_dec] = np.random.permutation(
                            mask_per_subject_dict[subject_dec])

                    data = load_data_by_subject(
                        data_dir, mask_per_subject_dict, subjects=[subject_dec])[subject_dec]

                    label_num = data['labs_train'].columns.get_loc(q)
                    row = fit_and_evaluate(data, subject, label_num=label_num)
                    rows.append(row)
                    # display(row)

                r_df = pd.concat(rows, axis=0)
                r_df.to_pickle(out_file)
                # display(r_df)
