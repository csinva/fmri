from neuro.features import qa_questions
from neuro.features.feat_select import get_alphas
import os
from os.path import dirname, join, expanduser
import sys
from imodelsx import submit_utils
from neuro.features.questions.gpt4 import QS_HYPOTHESES
path_to_file = os.path.dirname(os.path.abspath(__file__))
repo_dir = dirname(dirname(os.path.abspath(__file__)))
sys.path.append(repo_dir)

params_shared_dict = {
    # things to average over
    'use_extract_only': [0],
    'pc_components': [100],  # [100, -1],
    'use_eval_brain_drive': [0],
    'ndelays': [4],
    'nboots': [50],
    'feature_selection_stability_seeds': [5],


    # things to change
    'use_test_setup': [0],
    # 'save_dir': ['/home/chansingh/mntv1/deep-fMRI/encoding/aug14_neurosynth_gemv'],
    'save_dir': ['/home/chansingh/mntv1/deep-fMRI/encoding/oct17_neurosynth_gemv'],
    'subject': ['UTS01', 'UTS02', 'UTS03'],
    # 'subject': ['UTS03'],
    # 'subject': ['UTS01', 'UTS02', 'UTS03', 'UTS04', 'UTS05', 'UTS06', 'UTS07', 'UTS08'],
    'use_added_wordrate_feature': [0],  # , 1],
}

params_coupled_dict = {
    ('qa_questions_version', 'qa_embedding_model',
     'feature_selection_alpha', 'use_random_subset_features', 'seed', 'single_question_idx'):

    # run full models
    [
        # ('v3_boostexamples_merged', 'ensemble2',
        #  get_alphas('qa_embedder')[3], None, None, None),
        # ('v1neurosynth', 'ensemble2',
        #  # v1neurosynth is the set of GPT-4 hypotheses that were not computed with GPT-4
        #  None, None, None, None),
    ]
    +
    # shapley features
    [
        # ('v3_boostexamples_merged', 'ensemble2',
        #  get_alphas('qa_embedder')[3], 1, seed, None)
        # for seed in range(50)
    ]
    +
    [
        # ('v1neurosynth', 'ensemble2',
        #  None, 1, seed, None)
        # for seed in range(50)
    ]
    +
    # single question for the hypotheses
    # [
    #     ('v3_boostexamples_merged', 'ensemble2',
    #      get_alphas('qa_embedder')[3], None, None, i)
    #     for i in range(len(QS_HYPOTHESES))
    # ]
    # +
    # [
    #     ('v1neurosynth', 'ensemble2',
    #      None, None, None, i)
    #     for i in range(len(QS_HYPOTHESES))
    # ]
    # +
    # single question for everything
    [
        ('v3_boostexamples_merged', 'ensemble2',
         None, None, None, i)
        # for i in range(len(QS_HYPOTHESES))
        for i in range(qa_questions._get_merged_keep_indices_v3_boostexamples().size)
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
    # 'sku': 'E16ads_v5',
    # 'sku': 'E8ads_v5',
    'sku': '8C15',
    'mnt_rename': ('/home/chansingh/mntv1', '/mntv1'),
}

# args_list = args_list[:1]
# args_list = args_list[-1000:-500][::-1]
submit_utils.run_args_list(
    args_list,
    script_name=script_name,
    unique_seeds='seed',
    amlt_kwargs=amlt_kwargs_cpu,
    # n_cpus=8,
    # actually_run=False,
    # repeat_failed_jobs=True,
    # shuffle=True,
    cmd_python=f'export HF_TOKEN={open(expanduser("~/.HF_TOKEN"), "r").read().strip()}; python',
)
