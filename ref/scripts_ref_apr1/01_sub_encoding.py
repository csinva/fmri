from ridge_utils.config import mnt_dir
import itertools
import os
from os.path import dirname, join, expanduser
import sys
from dict_hash import sha256
import subprocess
from imodelsx import submit_utils
path_to_file = os.path.dirname(os.path.abspath(__file__))
repo_dir = dirname(dirname(os.path.abspath(__file__)))
sys.path.append(repo_dir)
# python /home/chansingh/fmri/01_fit_encoding.py

params_shared_dict = {
    # 'pc_components': [1000, 100, -1],  # [5000, 100, -1], # default -1 predicts each voxel independently
    # 'pc_components': [100, 1000, -1],
    'pc_components': [100],
    'encoding_model': ['ridge'],


    # things to average over
    'use_cache': [1],
    # 'save_dir': ['/home/chansingh/mntv1/deep-fMRI/encoding/results_mar28'],
    'save_dir': ['/home/chansingh/mntv1/deep-fMRI/encoding/results_apr1'],
    'nboots': [5],

    # fixed params
    # 'UTS03', 'UTS01', 'UTS02'],
    'subject': ['UTS03'],
    'use_test_setup': [0],

    'ndelays': [4, 8, 12],
    # 'seed': range(10),
    'seed': [1],
}


# main args are qa_embedder-10, 'v2', seed=1, ndelays=8
params_coupled_dict = {
    ('feature_space', 'qa_questions_version', 'qa_embedding_model'): [
        # baselines
        # ('bert-10', 'v1', 'mistralai/Mistral-7B-v0.1'),
        # ('eng1000', 'v1', 'mistralai/Mistral-7B-v0.1'),

        # # main
        # ('qa_embedder-10', 'v1', 'mistralai/Mistral-7B-v0.1'),

        # # vary mistral versions
        ('qa_embedder-10', 'v2', 'mistralai/Mistral-7B-v0.1'),
        ('qa_embedder-10', 'v3', 'mistralai/Mistral-7B-v0.1'),
        ('qa_embedder-10', 'v4', 'mistralai/Mistral-7B-v0.1'),

        # # mixtral
        # ('qa_embedder-10', 'v1', 'mistralai/Mixtral-8x7B-v0.1'),
        ('qa_embedder-10', 'v2', 'mistralai/Mixtral-8x7B-v0.1'),
        ('qa_embedder-10', 'v3', 'mistralai/Mixtral-8x7B-v0.1'),
        ('qa_embedder-10', 'v4', 'mistralai/Mixtral-8x7B-v0.1'),

        # llama-2
        # ('qa_embedder-10', 'v1', 'meta-llama/Llama-2-13b-hf'),
        # ('qa_embedder-10', 'v2', 'meta-llama/Llama-2-13b-hf'),
        # ('qa_embedder-10', 'v3', 'meta-llama/Llama-2-13b-hf'),
        # ('qa_embedder-10', 'v4', 'meta-llama/Llama-2-13b-hf'),

        # low priority -- ablation ngrams (should have run this with v2 but is v1)
        # ('qa_embedder-5', 'v1', 'mistralai/Mistral-7B-v0.1'),
    ],
}
# Args list is a list of dictionaries
# If you want to do something special to remove some of these runs, can remove them before calling run_args_list
args_list = submit_utils.get_args_list(
    params_shared_dict=params_shared_dict,
    params_coupled_dict=params_coupled_dict,
)
script_name = join(repo_dir, '01_fit_encoding.py')
amlt_kwargs = {
    'amlt_file': join(repo_dir, 'launch.yaml'),
    # [64G16-MI200-IB-xGMI, 64G16-MI200-xGMI
    # 'sku': '64G8-MI200-xGMI',
    # 'sku': '64G4-MI200-xGMI',
    'sku': '64G2-MI200-xGMI',
    # 'sku': '64G1-MI200-xGMI',
    'mnt_rename': ('/home/chansingh/mntv1', '/mntv1'),
}
submit_utils.run_args_list(
    args_list,
    script_name=script_name,
    actually_run=True,
    # unique_seeds=True,
    # amlt_kwargs=amlt_kwargs,
    # gpu_ids=[0, 1, 2, 3],
    # n_cpus=9,
    n_cpus=6,
    # gpu_ids=[0, 1],
    # gpu_ids=[[0, 1, 2, 3]],
    # gpu_ids=[[0, 1], [2, 3]],
    repeat_failed_jobs=True,
    shuffle=True,
    cmd_python=f'export HF_TOKEN={open(expanduser("~/.HF_TOKEN"), "r").read().strip()}; python',
)
