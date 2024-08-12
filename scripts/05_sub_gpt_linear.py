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
    'use_test_setup': [0],
    'use_extract_only': [0],
    'pc_components': [100],
    'use_eval_brain_drive': [0],
    'ndelays': [8],
    'num_stories': [-1],
    'nboots': [50],

    # things to change
    'save_dir': ['/home/chansingh/mntv1/deep-fMRI/encoding/jun16_gpt4'],
    'subject': ['UTS01', 'UTS02', 'UTS03'],
    # 'subject': ['UTS02', 'UTS03', 'UTS01'],
    # 'subject': ['UTS02'],  # , 'UTS03', 'UTS01'],
    'use_added_wordrate_feature': [0, 1],
}

params_coupled_dict = {
    ('feature_space', 'qa_questions_version', 'qa_embedding_model', 'embedding_layer'):

    [
        # baselines
        ('eng1000', None, None, None),
        ('wordrate', None, None, None),
        # ('bert-base-uncased', None, None, None),
        # ('qa_embedder', 'v3_boostexamples_merged', 'gpt4', None),
    ]
    +
    [
        ('qa_embedder', repr(QUESTIONS_GPT4[i]), 'gpt4', None)
        # for i in range(len(QUESTIONS_GPT4))
        for i in range(86)
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
    n_cpus=8,
    # actually_run=False,
    # repeat_failed_jobs=True,
    shuffle=True,
    cmd_python=f'export HF_TOKEN={open(expanduser("~/.HF_TOKEN"), "r").read().strip()}; python',
)
