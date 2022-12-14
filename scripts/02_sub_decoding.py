import itertools
import os
from os.path import dirname
import sys
import submit_utils
repo_dir = dirname(dirname(os.path.abspath(__file__)))
# python /home/chansingh/emb-gam/experimental/fmri/02_fit_decoding.py

PARAMS_COUPLED_DICT = {
    ('save_dir', 'subsample_frac'): [
        ('/home/chansingh/mntv1/deep-fMRI/results/linear_models/nov2_compare_joint', 1),
        # ('/home/chansingh/mntv1/deep-fMRI/results/linear_models/nov2_compare_joint', 1),
        # ('/home/chansingh/mntv1/deep-fMRI/results/linear_models/nov1_compare_joint', 0.1),
        # ('/home/chansingh/mntv1/deep-fMRI/results/linear_models/oct28_compare_joint', 0.1),
        # ('/home/chansingh/mntv1/deep-fMRI/results/linear_models/oct28_relu_dset_shuffled', -1),
        # ('/home/chansingh/mntv1/deep-fMRI/results/linear_models/oct26_relu_and_normalization', -1),
        # ('/home/chansingh/mntv1/deep-fMRI/results/linear_models/subsamp_oct22', 0.1),
    ],
}

##########################################
# params shared across everything (higher up things are looped over first)
##########################################
PARAMS_SHARED_DICT = {
    # things to average over
    'seed': [1],  # , 2, 3],
    'perc_threshold_fmri': [0, 99], #[99, 0, 50, 90],  # [0, 100]

    # things to vary
    'dset': [
        'rotten_tomatoes', 'moral_stories', 'sst2', 'tweet_eval',
        # 'emotion', 'go_emotions',  'trec',
        # 'ethics-commonsense', 'ethics-deontology', 'ethics-justice', 'ethics-utilitarianism', 'ethics-virtue'
    ],
    'model': [
        'bert-10__ndel=4__pc=50000_fmri',
        'bert-10__ndel=4__pc=50000_joint',
        # 'bert-10__ndel=4_fmri',
        # 'bert-base-uncased_embs',
        # 'bert-10__ndel=4_joint',
        # 'bert-sst2-10__ndel=4_fmri',
        # 'roberta-10__ndel=4_fmri',
        # 'roberta-large_embs',
        # 'glove__ndel=4_fmri',
        # 'glove_vecs'   ,
        # 'eng1000__ndel=4_fmri',
        # 'eng1000_vecs', 'bow_vecs',
    ],
    'nonlinearity': [
        # 'relu',
        'tanh',
        # 'none',
    ],
    # 'use_normalized_feats': [0, 1],
}

ks_final, param_combos_final = submit_utils.combine_param_dicts(
    PARAMS_SHARED_DICT, PARAMS_COUPLED_DICT)

idx_model = ks_final.index('model')
idx_perc_threshold_fmri = ks_final.index('perc_threshold_fmri')

# force idx_perc_threshold to 0 (default) unless it is a joint model
param_combos_final = [
    p for p in param_combos_final
    if p[idx_model].endswith('_fmri') \
        or p[idx_model].endswith('_joint') \
            or p[idx_perc_threshold_fmri] == 0
]

submit_utils.run_dicts(
    ks_final, param_combos_final,
    script_name='02_fit_decoding.py',
    actually_run=True,
    shuffle=True,
    reverse=False,
)
