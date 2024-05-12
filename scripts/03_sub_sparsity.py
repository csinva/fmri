

import os
from os.path import dirname, join, expanduser
import sys
import numpy as np
from imodelsx import submit_utils
path_to_file = os.path.dirname(os.path.abspath(__file__))
repo_dir = dirname(dirname(os.path.abspath(__file__)))
sys.path.append(repo_dir)
# python /home/chansingh/fmri/01_fit_encoding.py

params_shared_dict = {
    # things to average over
    'use_cache': [1],
    'nboots': [5],
    'use_test_setup': [0],
    'use_extract_only': [0],
    'save_dir': ['/home/chansingh/mntv1/deep-fMRI/encoding/may7'],
    'pc_components': [100],
    'feature_selection_frac': [0.5],

    # feature selection...
    'subject': ['shared'],  # first run with shared
    'seed': range(5),
    'feature_selection_alpha': np.logspace(0, -3, 20),  # 1

    # run with feature selection...
    # 'subject': ['UTS01', 'UTS02', 'UTS03'], # afterwards run with subjects
    # 'feature_selection_alpha': [1],


}

params_coupled_dict = {
    ('feature_space', 'qa_questions_version', 'qa_embedding_model'): [
        # ('qa_embedder', 'v3_boostexamples', 'ensemble1')
        ('eng1000', None, None),
        # ('bert-base-uncased', None, None), # maybe never have to run this
    ],
}
# Args list is a list of dictionaries
# If you want to do something special to remove some of these runs, can remove them before calling run_args_list
args_list = submit_utils.get_args_list(
    params_shared_dict=params_shared_dict,
    params_coupled_dict=params_coupled_dict,
)
script_name = join(repo_dir, 'experiments', '02_fit_encoding.py')
# amlt_kwargs = {
#     # 'amlt_file': join(repo_dir, 'scripts', 'launch_cpu.yaml'),
#     # 'sku': 'E4ads_v5',
#     # 'mnt_rename': ('/home/chansingh/mntv1', '/mntv1'),
#     'amlt_file': join(repo_dir, 'launch.yaml'),  # change this to run a cpu job
#     'sku': '64G2-MI200-xGMI',
#     'mnt_rename': ('/home/chansingh/mntv1', '/mntv1'),
# }
amlt_kwargs = {
    'amlt_file': join(repo_dir, 'scripts', 'launch_cpu.yaml'),
    # E4ads_v5 (30 GB), E8ads_v5 (56 GB), E16ads_v5 (120GB), E32ads_v5 (240GB), E64ads_v5 (480 GB)
    'sku': 'E64ads_v5',
    # 'sku': 'E32ads_v5',
    'mnt_rename': ('/home/chansingh/mntv1', '/mntv1'),
}
submit_utils.run_args_list(
    args_list,
    script_name=script_name,
    amlt_kwargs=amlt_kwargs,
    n_cpus=6,
    # n_cpus=2,
    # gpu_ids=[0, 1],
    # gpu_ids=[0, 1, 2, 3],
    # gpu_ids=[[0, 1], [2, 3]],
    # gpu_ids=[[0, 1, 2, 3]],
    actually_run=True,
    repeat_failed_jobs=True,
    shuffle=True,
    cmd_python=f'export HF_TOKEN={open(expanduser("~/.HF_TOKEN"), "r").read().strip()}; python',
)