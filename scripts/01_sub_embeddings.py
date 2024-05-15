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
    'use_test_setup': [0],
    'use_extract_only': [1],
    'pc_components': [100],
    'subject': ['UTS05'],
    'save_dir': ['/home/chansingh/mntv1/deep-fMRI/encoding/may7'],

    'seed_stories': range(25),
}

params_coupled_dict = {
    ('feature_space', 'qa_questions_version', 'qa_embedding_model', 'embedding_layer'):

    # baselines
    [
        # ('bert-base-uncased', None, None, None),
        # ('finetune_roberta-base-10', None, None, None),
        # ('finetune_roberta-base_binary-10', None, None, None),
    ]
    +

    # llama versions
    [
        # (llama, None. None, embedding_layer)
        # for llama in ['meta-llama/Llama-2-70b-hf', 'meta-llama/Meta-Llama-3-70B']
        # for embedding_layer in [12, 24, 36, 48, 60]
    ]
    +

    # qa versions
    [
        # ('qa_embedder', version, model, None)
        # ensemble1, v4, v5, v6, v4_boostexamples
        # for version in ['v3_boostexamples']
        # for version in ['v1', 'v2', 'v3_boostexamples', 'v3']
        # for model in [LLAMA8B_fewshot]  # , LLAMA8B, LLAMA8B_fewshot]
        # for model in [MIST7B, LLAMA8B, LLAMA8B_fewshot]
    ]

    +

    # qa 70B
    [
        ('qa_embedder', version, model, None)
        for version in ['v3_boostexamples']  # 'v1', 'v2', 'v3_boostexamples'
        for model in [LLAMA70B]
    ]

    # let's just skip llama 7B/8B
    # [
        # (llama, None, None, embedding_layer)
        # for llama in ['meta-llama/Llama-2-7b-hf', 'meta-llama/Meta-Llama-3-8B']
        # for embedding_layer in [6, 12, 18, 24, 30]
    # ]

}

args_list = submit_utils.get_args_list(
    params_shared_dict=params_shared_dict,
    params_coupled_dict=params_coupled_dict,
)
script_name = join(repo_dir, 'experiments', '02_fit_encoding.py')
amlt_kwargs = {
    # change this to run a cpu job
    'amlt_file': join(repo_dir, 'scripts', 'launch.yaml'),
    # [64G16-MI200-IB-xGMI, 64G16-MI200-xGMI
    'sku': '64G8-MI200-xGMI',
    # 'sku': '64G4-MI200-xGMI',
    # 'sku': '64G2-MI200-xGMI',
    'mnt_rename': ('/home/chansingh/mntv1', '/mntv1'),
}
submit_utils.run_args_list(
    args_list,
    script_name=script_name,
    unique_seeds='seed_stories',
    amlt_kwargs=amlt_kwargs,
    # gpu_ids=[0, 1],
    # gpu_ids=[0, 1, 2, 3],
    # gpu_ids=[[0, 1], [2, 3]],
    # gpu_ids=[[0, 1, 2, 3]],
    # actually_run=False,
    # shuffle=True,
    cmd_python=f'export HF_TOKEN={open(expanduser("~/.HF_TOKEN"), "r").read().strip()}; python',
)
