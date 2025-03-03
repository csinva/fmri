import imodelsx.process_results
# import viz
# import dvu
from tqdm import tqdm
import seaborn as sns
import os
import pandas as pd
from collections import defaultdict
from matplotlib import pyplot as plt
from os.path import join
import numpy as np
import sys
sys.path.append('../experiments')
# dvu.set_style()
fit_encoding = __import__('02_fit_encoding')
best_results_dir = '/home/chansingh/mntv1/deep-fMRI/encoding/may7'


def load_results(save_dir):
    dfs = []
    fnames = [
        fname for fname in os.listdir(save_dir)[::-1]
        if not fname.startswith('coef')
    ]
    for fname in tqdm(fnames):
        df = pd.read_pickle(join(save_dir, fname))
        # print(fname)
        # display(df)
        dfs.append(df.reset_index())
    d = pd.concat(dfs)
    # d = d.drop(columns='coef_')
    # .round(2)
    # d.set_index(['feats', 'dset'], inplace=True)
    d['nonlin_suffix'] = d['nonlinearity'].fillna(
        '').str.replace('None', '').str.replace('tanh', '_tanh')
    d['model'] = d['model'] + d['nonlin_suffix']
    d['model_full'] = d['model'] + '_thresh=' + \
        d['perc_threshold_fmri'].astype(str)
    return d


def load_clean_results(results_dir, experiment_filename='../experiments/02_fit_encoding.py'):
    # load the results in to a pandas dataframe
    r = imodelsx.process_results.get_results_df(results_dir)
    r = imodelsx.process_results.fill_missing_args_with_default(
        r, experiment_filename)
    for k in ['save_dir', 'save_dir_unique']:
        r[k] = r[k].map(lambda x: x if x.startswith('/home')
                        else x.replace('/mntv1', '/home/chansingh/mntv1'))
    if 'use_huge' in r.columns:
        r = r.drop(columns=['use_huge'])
    r['qa_embedding_model'] = r.apply(lambda row: {
        'mistralai/Mistral-7B-Instruct-v0.2': 'mist-7B',
        'mistralai/Mixtral-8x7B-Instruct-v0.1': 'mixt-moe',
        'meta-llama/Meta-Llama-3-8B-Instruct': 'llama3-8B',
        'meta-llama/Meta-Llama-3-8B-Instruct-fewshot': 'llama3-8B-fewshot',
        'meta-llama/Meta-Llama-3-8B-Instruct-refined': 'llama3-8B-refined',
    }.get(row['qa_embedding_model'], row['qa_embedding_model']) if 'qa_emb' in row['feature_space'] else '', axis=1)
    r['subject'] = r['subject'].str.replace('UTS', 'S')
    r['qa_questions_version'] = r.apply(
        lambda row: row['qa_questions_version'] if 'qa_emb' in row['feature_space'] else '', axis=1)
    mets = [c for c in r.columns if 'corrs' in c and (
        'mean' in c or 'frac' in c)]
    cols_varied = imodelsx.process_results.get_experiment_keys(
        r, experiment_filename)
    print('experiment varied these params:', cols_varied)
    r['corrs_test_mean_sem'] = r['corrs_test'].apply(
        lambda x: np.std(x) / np.sqrt(len(x)))
    r['feature_space_simplified'] = r['feature_space'].apply(
        lambda x: 'llama' if 'llama' in x else x
    )
    mets.append('corrs_test_mean_sem')
    return r, cols_varied, mets


def abbrev_question(q):
    q = q.replace('Is time mentioned in the input?',
                  'Does the input mention time?')
    q = q.replace('express a sense of belonging or connection to a place or community',
                  'express a connection to a community')
    q = q.replace('express the narrator\'s opinion or judgment about an event or character',
                  'express an opinion about an event or character')
    q = q.replace('involve a description of', 'describe a')
    q = q.replace('involve a discussion about', 'discuss')
    q = q.replace('involve an expression of', 'express')
    q = q.replace('involve the mention of', 'mention')
    q = q.replace('that leads to a change or revelation', '')
    q = q.replace(' ?', '?')
    for prefix in ['Does the sentence ', 'Is the sentence ', 'Does the input ', 'Is the input ', 'Does the text ', 'Is there a ', 'Is an ', 'Is a ', 'Is there ',]:
        q = q.replace(prefix, '...')
    return q


def abbrev_questions(qs):
    return [abbrev_question(q) for q in qs]
