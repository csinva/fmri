

import os
from os.path import dirname, join, expanduser
import sys
from imodelsx import submit_utils
path_to_file = os.path.dirname(os.path.abspath(__file__))
repo_dir = dirname(dirname(os.path.abspath(__file__)))
sys.path.append(repo_dir)

params_shared_dict = {
    'use_cache': [1],
    'save_dir': ['/home/chansingh/mntv1/deep-fMRI/encoding/may15_single_question'],
    'subject': ['UTS01', 'UTS02', 'UTS03'],
    'feature_space': ['qa_embedder'],
    'qa_embedding_model': ['ensemble1'],  # [MIST7B, LLAMA8B, 'ensemble1'],
    'qa_questions_version': ['v3_boostexamples'],  # 'v1', 'v2'
    # 'ndelays': [1, 2, 4, 8],
}

params_coupled_dict = {}
# Args list is a list of dictionaries
# If you want to do something special to remove some of these runs, can remove them before calling run_args_list
args_list = submit_utils.get_args_list(
    params_shared_dict=params_shared_dict,
    params_coupled_dict=params_coupled_dict,
)
script_name = join(repo_dir, 'experiments',  '04_fit_single_question.py')
amlt_kwargs = {
    'amlt_file': join(repo_dir, 'scripts', 'launch_cpu.yaml'),
    # E4ads_v5 (30 GB), E8ads_v5 (56 GB), E16ads_v5 (120GB), E32ads_v5 (240GB), E64ads_v5 (480 GB)
    # 'sku': 'E64ads_v5',
    # 'sku': 'E32ads_v5',
    # 'sku': 'E16ads_v5',
    'mnt_rename': ('/home/chansingh/mntv1', '/mntv1'),
}
submit_utils.run_args_list(
    args_list,
    script_name=script_name,
    # unique_seeds='seed_stories',
    # amlt_kwargs=amlt_kwargs,
    n_cpus=4,
    # n_cpus=2,
    # actually_run=False,
    repeat_failed_jobs=True,
    shuffle=True,
    cmd_python=f'export HF_TOKEN={open(expanduser("~/.HF_TOKEN"), "r").read().strip()}; python',
)
