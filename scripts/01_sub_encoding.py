import itertools
import os
from os.path import dirname, join
import sys
from imodelsx import submit_utils
path_to_file = os.path.dirname(os.path.abspath(__file__))
repo_dir = dirname(dirname(os.path.abspath(__file__)))
# python /home/chansingh/fmri/01_fit_encoding.py

params_shared_dict = {
    # 'pc_components': [1000, 100, -1],  # [5000, 100, -1], # default -1 predicts each voxel independently
    'pc_components': [100],
    'encoding_model': ['ridge'],


    # things to average over
    'use_cache': [1],
    'save_dir': ['/home/chansingh/mntv1/deep-fMRI/encoding/results_mar28'],
    'nboots': [5],

    # fixed params
    # 'UTS03', 'UTS01', 'UTS02'],
    'subject': ['UTS03'],
    # 'mlp_dim_hidden': [768],
    'use_test_setup': [0],
}
params_coupled_dict = {
    ('feature_space', 'seed', 'ndelays'): [
        # ('bert-10', 1, 4),
        # ('bert-10', 1, 8),
        # ('bert-10', 1, 12),
        # ('eng1000', 1, 4),
        # ('eng1000', 1, 8),
        # ('eng1000', 1, 12),
        # ('qa_embedder-5', 1, 4),
        # ('qa_embedder-5', 1, 8),
        # ('qa_embedder-5', 1, 12),
        ('qa_embedder-10', 1, 4),
        ('qa_embedder-10', 2, 8),
        ('qa_embedder-10', 3, 12),
    ],
}
# Args list is a list of dictionaries
# If you want to do something special to remove some of these runs, can remove them before calling run_args_list
args_list = submit_utils.get_args_list(
    params_shared_dict=params_shared_dict,
    params_coupled_dict=params_coupled_dict,
)
submit_utils.run_args_list(
    args_list,
    script_name=join(repo_dir, '01_fit_encoding.py'),
    actually_run=True,
    # gpu_ids=[0, 1],
    # n_cpus=9,
    # n_cpus=2,
    gpu_ids=[1, 2, 3],
    # gpu_ids=[0, 1, 2, 3],
    # gpu_ids=[[0, 1], [2, 3]],
    repeat_failed_jobs=True,
    shuffle=True,
)
