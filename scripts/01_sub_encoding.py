import os
from os.path import dirname, join, expanduser
import sys
from imodelsx import submit_utils
path_to_file = os.path.dirname(os.path.abspath(__file__))
repo_dir = dirname(dirname(os.path.abspath(__file__)))
sys.path.append(repo_dir)
# python /home/chansingh/fmri/01_fit_encoding.py
MIST7B = 'mistralai/Mistral-7B-Instruct-v0.2'
MIXTMOE = 'mistralai/Mixtral-8x7B-Instruct-v0.1'
BEST_RUN = '/home/chansingh/mntv1/deep-fMRI/encoding/results_apr7/68936a10a548e2b4ce895d14047ac49e7a56c3217e50365134f78f990036c5f7'

params_shared_dict = {
    # things to average over
    'use_cache': [1],
    'nboots': [5],
    'use_test_setup': [0],
    'encoding_model': ['ridge'],
    'subject': ['UTS03'],
    'save_dir': ['/home/chansingh/mntv1/deep-fMRI/encoding/results_apr7'],
    'ndelays': [4, 8, 12],

    # cluster
    # 'seed': range(14),
    'seed': range(30),
    'pc_components': [100],
    # 'ndelays': [4],

    # local
    # 'seed': [1],
    # 'pc_components': [1000, 100, -1],
    # 'use_extract_only': [0],
}

params_coupled_dict = {
    ('feature_space', 'qa_questions_version', 'qa_embedding_model'): [
        # # baselines
        # ('bert-10', 'v1', MIST7B),
        # ('eng1000', 'v1', MIST7B),
        # ('llama2-7B_lay6-10', 'v1', MIST7B),
        # ('llama2-7B_lay12-10', 'v1', MIST7B),
        # ('llama2-7B_lay18-10', 'v1', MIST7B),
        # ('llama2-7B_lay24-10', 'v1', MIST7B),
        # ('llama2-7B_lay30-10', 'v1', MIST7B),

        # ('llama2-13B_lay6-10', 'v1', MIST7B),
        # ('llama2-13B_lay12-10', 'v1', MIST7B),
        # ('llama2-13B_lay18-10', 'v1', MIST7B),
        # ('llama2-13B_lay24-10', 'v1', MIST7B),
        # ('llama2-13B_lay30-10', 'v1', MIST7B),

        # ('llama2-70B_lay12-10', 'v1', MIST7B),
        # ('llama2-70B_lay24-10', 'v1', MIST7B),
        # ('llama2-70B_lay36-10', 'v1', MIST7B),
        # ('llama2-70B_lay48-10', 'v1', MIST7B),
        # ('llama2-70B_lay60-10', 'v1', MIST7B),


        # # # main
        # ('qa_embedder-10', 'v1', MIST7B),

        # vary question versions
        # ('qa_embedder-10', 'v2', MIST7B),
        # ('qa_embedder-10', 'v3', MIST7B),
        # ('qa_embedder-10', 'v4', MIST7B),
        # ('qa_embedder-10', 'v5', MIST7B),
        # ('qa_embedder-10', 'v6', MIST7B),
        ('qa_embedder-10', 'v3_boostbasic', MIST7B),
        ('qa_embedder-10', 'v3_boostexamples', MIST7B),

        # vary context len
        # ('qa_embedder-25', 'v1', MIST7B),

        # # mixtral
        # ('qa_embedder-10', 'v1', MIXTMOE),
        # ('qa_embedder-10', 'v2', MIXTMOE),
        # ('qa_embedder-10', 'v3', MIXTMOE),
        # ('qa_embedder-10', 'v4', MIXTMOE),

        # -last, -end versions (try 10, 50, 75)
        # ('qa_embedder-25', 'v1-last', MIST7B),
        # ('qa_embedder-25', 'v1-ending', MIST7B),
        # mixtral -last, -end.....
        # ('qa_embedder-25', 'v1-ending', MIXTMOE),

        # bert sec versions
        # ('bert-sec3', 'v1', MIST7B),
        # ('bert-sec5', 'v1', MIST7B),

        # tr versions
        # ('bert-tr2', 'v1', MIST7B),
        # ('bert-tr3', 'v1', MIST7B),

        # qa sec versions
        # ('qa_embedder-sec3', 'v1', MIST7B),
        # ('qa_embedder-sec5', 'v1', MIST7B),

        # qa tr versions
        # ('qa_embedder-tr2', 'v1', MIST7B),
        # ('qa_embedder-tr3', 'v1', MIST7B),
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
    'amlt_file': join(repo_dir, 'launch.yaml'),  # change this to run a cpu job
    # [64G16-MI200-IB-xGMI, 64G16-MI200-xGMI
    # 'sku': '64G8-MI200-xGMI',
    # 'sku': '64G4-MI200-xGMI',
    'sku': '64G2-MI200-xGMI',
    'mnt_rename': ('/home/chansingh/mntv1', '/mntv1'),
}
submit_utils.run_args_list(
    args_list,
    script_name=script_name,
    unique_seeds=True,
    amlt_kwargs=amlt_kwargs,
    # n_cpus=9,
    # n_cpus=4,
    # gpu_ids=[0, 1],
    # gpu_ids=[0, 1, 2, 3],
    # gpu_ids=[[0, 1], [2, 3]],
    # gpu_ids=[[0, 1, 2, 3]],
    # actually_run=False,
    repeat_failed_jobs=True,
    shuffle=True,
    cmd_python=f'export HF_TOKEN={open(expanduser("~/.HF_TOKEN"), "r").read().strip()}; python',
)
