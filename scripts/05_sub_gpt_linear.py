import os
from os.path import dirname, join, expanduser
import sys
from imodelsx import submit_utils
from neuro.features.questions.gpt4 import QUESTIONS_GPT4
path_to_file = os.path.dirname(os.path.abspath(__file__))
repo_dir = dirname(dirname(os.path.abspath(__file__)))
sys.path.append(repo_dir)

params_shared_dict = {
    # things to average over
    # 'use_cache': [1],
    # 'nboots': [5],
    # 'encoding_model': ['ridge'],
    'use_test_setup': [0],
    'use_extract_only': [0],
    'pc_components': [100],

    # 'subject': [f'UTS0{k}' for k in range(1, 9)],
    # 'subject': [f'UTS0{k}' for k in range(1, 4)],
    # 'subject': [f'UTS0{k}' for k in range(4, 9)],
    'subject': ['UTS03'],

    # ['UTS01', 'UTS02', 'UTS03', 'UTS04', 'UTS05', 'UTS06', 'UTS07', 'UTS08']
    'save_dir': ['/home/chansingh/mntv1/deep-fMRI/encoding/jun16_gpt4'],
    'use_eval_brain_drive': [0],
    # 'ndelays': [4, 8],
    'ndelays': [8],

    # default is -1, SO4-SO8 have 24 or 25 stories
    # 'num_stories': [-1, 5, 10, 20],
    # 'num_stories': [5, 10, 20],
    'num_stories': [-1],
}

params_coupled_dict = {
    ('feature_space', 'qa_questions_version', 'qa_embedding_model', 'embedding_layer'):

    [
        # baselines
        # ('eng1000', None, None, None),
        ('wordrate', None, None, None),
        # ('bert-base-uncased', None, None, None),
        # ('qa_embedder', 'v3_boostexamples_merged', 'gpt4', None),
        # ('qa_embedder', repr(QUESTIONS_GPT4[0]), 'gpt4', None),
    ],
}
# Args list is a list of dictionaries
# If you want to do something special to remove some of these runs, can remove them before calling run_args_list
args_list = submit_utils.get_args_list(
    params_shared_dict=params_shared_dict,
    params_coupled_dict=params_coupled_dict,
)
script_name = join(repo_dir, 'experiments', '02_fit_encoding.py')
amlt_kwargs_cpu = {
    'amlt_file': join(repo_dir, 'scripts', 'launch_cpu.yaml'),
    # E4ads_v5 (30 GB), E8ads_v5 (56 GB), E16ads_v5 (120GB), E32ads_v5 (240GB), E64ads_v5 (480 GB)
    # 'sku': 'E64ads_v5',
    # 'sku': 'E32ads_v5',
    # 'sku': 'E16ads_v5',
    # 'sku': 'E8ads_v5',
    'mnt_rename': ('/home/chansingh/mntv1', '/mntv1'),
}
submit_utils.run_args_list(
    args_list,
    script_name=script_name,
    unique_seeds='seed_stories',
    # amlt_kwargs=amlt_kwargs_cpu,
    # n_cpus=8,
    # actually_run=False,
    # repeat_failed_jobs=True,
    shuffle=True,
    cmd_python=f'export HF_TOKEN={open(expanduser("~/.HF_TOKEN"), "r").read().strip()}; python',
)
