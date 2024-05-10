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

# other models (also -refined models)
LLAMA70B = 'meta-llama/Meta-Llama-3-70B-Instruct'
LLAMA70B_fewshot = 'meta-llama/Meta-Llama-3-70B-Instruct-fewshot'
MIXTMOE = 'mistralai/Mixtral-8x7B-Instruct-v0.1'

BEST_RUN = '/home/chansingh/mntv1/deep-fMRI/encoding/results_apr7/68936a10a548e2b4ce895d14047ac49e7a56c3217e50365134f78f990036c5f7'

params_shared_dict = {
    # things to average over
    # 'use_cache': [1],
    # 'nboots': [5],
    # 'encoding_model': ['ridge'],
    'use_test_setup': [0],
    'use_extract_only': [0],

    # 'subject': [f'UTS0{k}' for k in range(1, 9)],
    # 'subject': [f'UTS0{k}' for k in range(1, 4)],
    # 'subject': [f'UTS0{k}' for k in range(4, 9)],
    'subject': ['UTS03'],

    # ['UTS01', 'UTS02', 'UTS03', 'UTS04', 'UTS05', 'UTS06', 'UTS07', 'UTS08']
    'save_dir': ['/home/chansingh/mntv1/deep-fMRI/encoding/may7'],
    # 'ndelays': [4, 8],
    # 'ndelays': [8],

    # cluster
    'pc_components': [100],
}

params_coupled_dict = {
    ('feature_space', 'qa_questions_version', 'qa_embedding_model', 'embedding_layer'):

    [
        # baselines
        ('eng1000', 'v1', MIST7B, -1),
        ('bert-base-uncased', 'v1', MIST7B, -1),
        # ('finetune_roberta-base-10', 'v1', MIST7B, -1),
        # ('finetune_roberta-base_binary-10', 'v1', MIST7B, -1),
    ]
    +

    # llama versions
    [
        # (llama, 'v1', MIST7B, embedding_layer)
        # for llama in ['meta-llama/Llama-2-7b-hf', 'meta-llama/Meta-Llama-3-8B']
        # for embedding_layer in [6, 12, 18, 24, 30]
    ]
    +
    [
        # (llama, 'v1', MIST7B, embedding_layer)
        # for llama in ['meta-llama/Llama-2-70b-hf', 'meta-llama/Meta-Llama-3-70B']
        # for embedding_layer in [12, 24, 36, 48, 60]
    ]
    +

    # qa versions
    [
        # ensemble1, v4, v5, v6, v4_boostexamples
        ('qa_embedder', version, model, -1)
        for version in ['v1', 'v2', 'v3']  # 'v3_boostexamples', 'v3']
        for model in [MIST7B, LLAMA8B]  # LLAMA8B_fewshot
    ]

    +

    # qa 70B
    [
        # ('qa_embedder', version, model, -1)
        # for version in ['v1', 'v2']
        # for model in [LLAMA70B]
    ]


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
    'sku': 'E8ads_v5',
    'mnt_rename': ('/home/chansingh/mntv1', '/mntv1'),
}
submit_utils.run_args_list(
    args_list,
    script_name=script_name,
    unique_seeds='seed_stories',
    # amlt_kwargs=amlt_kwargs_cpu,
    n_cpus=9,
    # n_cpus=3,
    # actually_run=False,
    repeat_failed_jobs=True,
    shuffle=True,
    cmd_python=f'export HF_TOKEN={open(expanduser("~/.HF_TOKEN"), "r").read().strip()}; python',
)
