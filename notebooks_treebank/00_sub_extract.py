import os
from os.path import dirname, join, expanduser
import sys
from imodelsx import submit_utils
path_to_file = os.path.dirname(os.path.abspath(__file__))
repo_dir = dirname(dirname(os.path.abspath(__file__)))
sys.path.append(repo_dir)

# main models
MIST7B = 'mistralai/Mistral-7B-Instruct-v0.2'
LLAMA8B = 'meta-llama/Meta-Llama-3-8B-Instruct'
LLAMA8B_fewshot = 'meta-llama/Meta-Llama-3-8B-Instruct-fewshot'
GEMMA7B = 'google/gemma-7b-it'

# other models (also -refined models)
LLAMA70B = 'meta-llama/Meta-Llama-3-70B-Instruct'
LLAMA70B_fewshot = 'meta-llama/Meta-Llama-3-70B-Instruct-fewshot'
MIXTMOE = 'mistralai/Mixtral-8x7B-Instruct-v0.1'

params_shared_dict = {
    'save_dir': ['/home/chansingh/fmri/results/ecog'],
    # 'seed_stories': range(2),
    # 'checkpoint': [LLAMA8B, MIST7B, GEMMA7B],
    # 'checkpoint': [MIXTMOE],
    'checkpoint': ['gpt-4o-mini'],
    'use_cache': [0],
    # 'setting': ['sec_3'],
    # 'questions': ['o1_dec26'],
    'questions': ['qs_35_stable'],
}
FACTOR = 4
# FACTOR = 1
# FACTOR = 1
params_coupled_dict = {
    ('setting', 'batch_size'): [
        ('words', int(32/FACTOR)),
        ('sec_1.5', int(16/FACTOR)),
        ('sec_3', int(8/FACTOR)),
        ('sec_6', int(8/FACTOR)),
        # ('sec_12', 4), # too long, not necessary
    ],
}

args_list = submit_utils.get_args_list(
    params_shared_dict=params_shared_dict,
    params_coupled_dict=params_coupled_dict,
)

# args_list = args_list[:1]

script_name = join(repo_dir, 'notebooks_treebank', '00_extract.py')
amlt_kwargs = {
    # change this to run a cpu job
    'amlt_file': join(repo_dir, 'scripts', 'launch.yaml'),
    # [64G16-MI200-IB-xGMI, 64G16-MI200-xGMI
    'sku': '64G8-MI200-xGMI',
    # 'sku': '64G4-MI200-xGMI',
    # 'sku': '64G2-MI200-xGMI',
    'mnt_rename': ('/home/chansingh/mntv1', '/mntv1'),
}
# print(args_list)
submit_utils.run_args_list(
    args_list,
    script_name=script_name,
    # unique_seeds='seed_stories',
    # amlt_kwargs=amlt_kwargs,
    # gpu_ids=[0, 1],
    # gpu_ids=[0, 1, 2, 3],
    # gpu_ids=[[0, 1], [2, 3]],
    # actually_run=False,
    # shuffle=True,
    # cmd_python=f'export HF_TOKEN={open(expanduser("~/.HF_TOKEN"), "r").read().strip()}; python',
    cmd_python='python',
)
