from collections import defaultdict
import os.path
from copy import deepcopy

from tqdm import tqdm
import torch
import random
import logging
from sklearn.ensemble import RandomForestRegressor
from os.path import join, dirname
import argparse
import numpy as np
import joblib
import os
from neuro.data import response_utils
from neuro.features import feature_utils, feat_select
from neuro.encoding.ridge import bootstrap_ridge, gen_temporal_chunk_splits
import imodelsx.cache_save_utils
import neuro.data.story_names as story_names
from neuro.features.questions.gpt4 import QUESTIONS_GPT4, QS_HYPOTHESES_COMPUTED
import random
import warnings
import time
from neuro.encoding.eval import nancorr, evaluate_pc_model_on_each_voxel, add_summary_stats

# get path to current file
path_to_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path_to_file = os.path.dirname(os.path.abspath(__file__))


def add_main_args(parser):
    """Caching uses the non-default values from argparse to name the saving directory.
    Changing the default arg an argument will break cache compatibility with previous runs.
    """
    # data arguments
    parser.add_argument("--subject", type=str, default='UTS03',
                        choices=[f'UTS0{k}' for k in range(1, 9)] + ['shared'],
                        help='shared concatenates responses for S01-S03 (and only load shared stories), useful for feature selection')
    parser.add_argument('--pc_components', type=int, default=-1,
                        help='''number of principal components to use for reducing output (-1 doesnt use PCA at all).
                        Note, use_test_setup alters this to 100.''')
    parser.add_argument('--use_huge', type=int, default=1,
                        help='''Whether to use huge list of stories
                        (if use_test_setup or not UTS01-03, this will automatically be set to 0)
                        ''')
    parser.add_argument('--num_stories', type=int, default=-1,
                        help='''number of stories to use for training (-1 for all).
                        Stories are selected from huge list unless use_test_setup''')
    parser.add_argument("--distill_model_path", type=str,
                        default=None,
                        # default='/home/chansingh/mntv1/deep-fMRI/encoding/results_apr7/68936a10a548e2b4ce895d14047ac49e7a56c3217e50365134f78f990036c5f7',
                        help='Path to saved pickles for distillation. Instead of fitting responses, fit the predictions of this model.')
    parser.add_argument("--use_eval_brain_drive", type=int, default=0,
                        help='Whether to evaluate fitted model on brain drive stories')

    # encoding
    parser.add_argument("--feature_space", type=str,
                        default='qa_embedder',
                        choices=['qa_embedder', 'eng1000', 'wordrate', 'finetune_roberta-base', 'finetune_roberta-base_binary',
                                 'bert-base-uncased', 'distilbert-base-uncased',  'roberta-base',
                                 'meta-llama/Llama-2-7b-hf', 'meta-llama/Llama-2-70b-hf', 'meta-llama/Meta-Llama-3-8B', 'meta-llama/Meta-Llama-3-70B'],
                        help='''Passing a standard HF model name will compute embeddings from that model.
                        Models starting with "finetune_" load custom models
                        qa_embedder computes qa embeddings with the checkpoint in args.qa_embedding_model
                        ''')
    parser.add_argument('--embedding_layer', type=int, default=-1,
                        help='''If args.feature_space is a HF model, which layer to use for embeddings (-1 for default layer)''')
    parser.add_argument('--input_chunking_type', type=str, default='ngram',
                        choices=['ngram', 'tr', 'sec'],
                        help='''Type of chunking to use for input features.
                        ngram chunks are number of words
                        tr chunks by TRs (and does not compute features per-word, so is faster but less accurate)
                        sec chunks by seconds leading up to each word''')
    parser.add_argument('--input_chunking_size', type=int, default=10,
                        help='Number of input chunks (corresponding to input_chunking_type)')
    parser.add_argument("--feature_selection_alpha", type=float,
                        default=-1,
                        help='Alpha to use when running feature selection (if >= 0). Alpha to use for feature selection.')
    parser.add_argument("--feature_selection_frac", type=float,
                        default=0.5,
                        help='''Randomly bootsraps data to this fraction of examples.
                        Applies if feature_selection_alpha >= 0.''')
    parser.add_argument("--feature_selection_stability_seeds", type=int,
                        default=-1,
                        help='''Number of seeds to use for stability selection (only keeps a feature if it was selected in all seeds).
                        Applies if feature_selection_alpha >= 0.
                        Note: needs to run feature-selection with this many different seeds (slow, good to run in parallel before calling this)
                        ''')
    parser.add_argument("--use_added_wordrate_feature", type=int, default=0,
                        choices=[0, 1], help='Whether to add the wordrate feature')

    # qa features
    parser.add_argument("--qa_embedding_model", type=str,
                        default='mistralai/Mistral-7B-Instruct-v0.2',
                        # default='gpt4',
                        help='Model to use for QA embedding, if feature_space is qa_embedder',
                        )
    parser.add_argument("--qa_questions_version", type=str,
                        default='v1',
                        choices=['v1', 'v2', 'v3', 'v3_boostexamples',
                                 'v4_boostexamples', 'v4', 'v5', 'v3_boostexamples_merged'] +
                        ['v1neurosynth', 'qs_35'] + QS_HYPOTHESES_COMPUTED,
                        help='''Which set of QA questions to use, if feature_space is qa_embedder.
                        If passed a single question name, uses only that question with gpt4-extracted feats.
                        v1neurosynth: will use the set of GPT-4 hypotheses that were not computed with GPT-4
                        qs_35: the set of 35 stable questions
                        ''')
    parser.add_argument("--use_random_subset_features", type=int, default=0,
                        help='Whether to use a random subset of features')
    parser.add_argument("--single_question_idx", type=int, default=-1,
                        help='If passed, only use this question index for QA features')

    # linear modeling
    parser.add_argument("--encoding_model", type=str,
                        default='ridge',
                        # default='randomforest'
                        )
    parser.add_argument("--ndelays", type=int, default=4)
    parser.add_argument("--nboots", type=int, default=5)
    parser.add_argument("--chunklen", type=int, default=40,
                        help='try to get nchunks * chunklen to ~20% of training data')
    parser.add_argument("--nchunks", type=int, default=125)
    parser.add_argument("--singcutoff", type=float, default=1e-10)
    parser.add_argument("-single_alpha", action="store_true")
    # parser.add_argument("--trim", type=int, default=5) # always end up using 5
    # parser.add_argument("--l1_ratio", type=float,
    # default=0.5, help='l1 ratio for elasticnet (ignored if encoding_model is not elasticnet)')
    # parser.add_argument("--min_alpha", type=float,
    # default=-1, help='min alpha, useful for forcing sparse coefs in elasticnet. Note: if too large, we arent really doing CV at all.')
    # parser.add_argument('--pc_components_input', type=int, default=-1,
    # help='number of principal components to use to transform features (-1 doesnt use PCA at all)')
    # parser.add_argument("--mlp_dim_hidden", type=int,
    # help="hidden dim for MLP", default=512)

    # basic params
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--use_test_setup', type=int, default=1,
                        help='For fast testing - train/test on a couple stories with few nboots.')
    return parser


def add_computational_args(parser):
    """Arguments that only affect computation and not the results (shouldnt use when checking cache)"""
    parser.add_argument('--save_dir', type=str,
                        default=os.path.join(path_to_repo, 'results'))
    parser.add_argument(
        "--use_cache",
        type=int,
        default=1,
        choices=[0, 1],
        help="whether to check for cache",
    )
    parser.add_argument(
        "--use_save_features",
        type=int,
        default=1,
        choices=[0, 1],
        help="whether to save the constructed features",
    )
    parser.add_argument(
        "--use_extract_only",
        type=int,
        default=1,
        choices=[0, 1],
        help="whether to jointly extract train/test (speeds things up if running over many seeds)",
    )
    parser.add_argument(
        '--seed_stories',
        type=int,
        default=1,
        help='seed for order that stories are processed in',
    )
    return parser


def get_story_names(args):
    if args.use_test_setup == 1:
        args.nboots = 5
        args.use_extract_only = 0
        args.use_huge = 1
        story_names_train = ['sloth', 'adollshouse']
        story_names_test = ['fromboyhoodtofatherhood']
        # story_names_train = story_names.get_story_names(
        # args.subject, 'train', use_huge=args.use_huge)[:20]
        # story_names_test = story_names.get_story_names(
        # args.subject, 'test', use_huge=args.use_huge)[:20]
        args.pc_components = 100
        args.use_eval_brain_drive = 0
        # args.qa_embedding_model = 'mistralai/Mistral-7B-Instruct-v0.2'
        # args.feature_selection_frac = 0.2
    elif args.use_test_setup == 2:
        args.nboots = 3
        args.feature_space = 'eng1000'
        args.use_extract_only = 0
        args.use_huge = 1
        args.subject = 'UTS02'
        story_names_train = story_names.get_story_names(
            args.subject, 'train', use_huge=args.use_huge)
        story_names_test = ['adollshouse', 'hangtime', 'sloth']
        story_names_train = [
            s for s in story_names_train if not s in story_names_test]
        # story_names_test = ['sloth', 'adollshouse', 'fromboyhoodtofatherhood']
        story_names_test = ['GenStory27', 'GenStory28', 'GenStory29']
        args.pc_components = 100
        args.use_eval_brain_drive = 0
    elif args.use_test_setup == 3:
        # eng1000 full run
        args.nboots = 5
        args.feature_space = 'eng1000'
        args.use_extract_only = 0
        args.use_huge = 1
        args.subject = 'UTS03'
        story_names_train = story_names.get_story_names(
            args.subject, 'train', use_huge=args.use_huge)
        story_names_test = story_names.get_story_names(
            args.subject, 'test', use_huge=args.use_huge)
    else:
        story_names_train = story_names.get_story_names(
            args.subject, 'train', use_huge=args.use_huge)
        story_names_test = story_names.get_story_names(
            args.subject, 'test', use_huge=args.use_huge)

    if args.num_stories > 0:
        story_names_train = story_names_train[:args.num_stories]
        story_names_test = story_names_test[:args.num_stories]

    rng = np.random.default_rng(args.seed_stories)
    rng.shuffle(story_names_train)
    return story_names_train, story_names_test


def fit_regression(args, r, features_train_delayed, resp_train, features_test_delayed, resp_test):
    if args.pc_components > 0:
        # if args.min_alpha > 0:
        # alphas = np.logspace(np.log10(args.min_alpha), 4, 12)
        # else:
        alphas = np.logspace(1, 4, 12)
        weights_key = 'weights_pc'
        corrs_key_test = 'corrs_test_pc'
        corrs_key_tune = 'corrs_tune_pc'
    else:
        # if args.min_alpha > 0:
        # alphas = np.logspace(np.log10(args.min_alpha), 4, 12)
        # else:
        alphas = np.logspace(1, 4, 12)
        weights_key = 'weights'
        corrs_key_test = 'corrs_test'
        corrs_key_tune = 'corrs_tune'

    if args.encoding_model == 'ridge':
        if args.use_test_setup == 3:
            example_params = {
                'features_train_delayed': features_train_delayed,
                'resp_train': resp_train,
                'features_test_delayed': features_test_delayed,
                'resp_test': resp_test,
                'alphas': alphas,
                'nboots': args.nboots,
                'chunklen': args.chunklen,
                'nchunks': args.nchunks,
                'singcutoff': args.singcutoff,
                'single_alpha': args.single_alpha,
            }
            joblib.dump(example_params, 'example_params_full.joblib')
        wt, corrs_test, alphas_best, corrs_tune, valinds = bootstrap_ridge(
            features_train_delayed, resp_train, features_test_delayed, resp_test,
            alphas, args.nboots, args.chunklen,
            args.nchunks, singcutoff=args.singcutoff, single_alpha=args.single_alpha)

        # Save regression results
        model_params_to_save = {
            weights_key: wt,
            'alphas_best': alphas_best,
            # 'valinds': valinds
        }

        # corrs_tune is (alphas, voxels, and bootstrap samples)
        # now reorder so it's (voxels, alphas, bootstrap samples)
        corrs_tune = np.swapaxes(corrs_tune, 0, 1)
        # mean over bootstrap samples
        corrs_tune = corrs_tune.mean(axis=-1)

        # replace each element of alphas_best with its index in alphas
        alphas_idx = np.array([np.where(alphas == a)[0][0]
                               for a in alphas_best])

        # apply best alpha to each voxel
        corrs_tune = corrs_tune[np.arange(corrs_tune.shape[0]), alphas_idx]

        # so we average over the bootstrap samples and take the max over the alphas
        r[corrs_key_tune] = corrs_tune
        r[corrs_key_test] = corrs_test
    elif args.encoding_model == 'randomforest':
        rf = RandomForestRegressor(
            n_estimators=100, n_jobs=10)  # , max_depth=5)
        corrs_test = []
        for i in range(resp_train.shape[1]):
            rf.fit(features_train_delayed, resp_train[:, i])
            preds = rf.predict(features_test_delayed)
            # corrs_test.append(np.corrcoef(resp_test[:, i], preds)[0, 1])
            corrs_test.append(nancorr(resp_test[:, i], preds[:, i]))
            print(i, 'rf corr', corrs_test[-1])
        corrs_test = np.array(corrs_test)
        corrs_test[np.isnan(corrs_test)] = 0
        r[corrs_key_test] = corrs_test
        model_params_to_save = {
            'weights': rf.feature_importances_,
        }
    elif args.encoding_model == 'tabpfn':
        from tabpfn import TabPFNRegressor
        rf = TabPFNRegressor(device='cuda')
        corrs_test = []
        preds_pc = []
        for i in tqdm(range(resp_train.shape[1])):
            rf.fit(features_train_delayed, resp_train[:, i])
            preds = rf.predict(features_test_delayed)
            corrs_test.append(nancorr(resp_test[:, i], preds))
            # print(i, 'tabpfn corr', corrs_test[-1])
            preds_pc.append(preds)
        corrs_test = np.array(corrs_test)
        corrs_test[np.isnan(corrs_test)] = 0
        r[corrs_key_test] = corrs_test
        model_params_to_save = {'preds_pc': preds_pc}
    elif args.encoding_model == 'mlp':
        from sklearn.neural_network import MLPRegressor
        mlp = MLPRegressor(max_iter=1000)
        corrs_test = []
        mlp.fit(features_train_delayed, resp_train)
        preds = mlp.predict(features_test_delayed)
        for i in range(resp_train.shape[1]):
            corrs_test.append(nancorr(resp_test[:, i], preds[:, i]))
            # print(i, 'mlp corr', corrs_test[-1])
        corrs_test = np.array(corrs_test)
        corrs_test[np.isnan(corrs_test)] = 0
        r[corrs_key_test] = corrs_test
        model_params_to_save = {
            'preds_pc': preds,
        }

    return r, model_params_to_save


def _check_args(args):
    if args.subject not in ['UTS01', 'UTS02', 'UTS03'] and args.use_huge:
        args.use_huge = 0
        # warnings.warn(
        # f'Not using huge list of stories for subject {args.subject}')

    if args.embedding_layer >= 0:
        assert not args.feature_space in ['qa_embedder', 'eng1000', 'finetune_roberta-base',
                                          'finetune_roberta-base_binary'], f'embedding_layer only used for HF models but {args.feature_space} passed'
        assert args.qa_questions_version == 'v1', 'embedding_layer only used with v1'
        assert args.qa_embedding_model == 'mistralai/Mistral-7B-Instruct-v0.2', 'embedding_layer only used with dfeault (mistral) qa_embedding_model'

    return args


if __name__ == "__main__":
    # get args
    parser = argparse.ArgumentParser()
    parser_without_computational_args = add_main_args(parser)
    parser = add_computational_args(
        deepcopy(parser_without_computational_args))
    args = parser.parse_args()
    args = _check_args(args)

    # set up logging
    logger = logging.getLogger()
    logging.basicConfig(level=logging.INFO)

    # set up saving directory + check for cache
    already_cached, save_dir_unique = imodelsx.cache_save_utils.get_save_dir_unique(
        parser, parser_without_computational_args, args, args.save_dir
    )
    if args.use_cache and already_cached and not args.use_test_setup:
        print(f"cached version exists! Successfully skipping :)\n\n\n")
        exit(0)
    for k in sorted(vars(args)):
        print("\t" + k + " " + str(vars(args)[k]))
    logging.info(f"\n\n\tsaving to " + save_dir_unique + "\n")

    # set seed
    t0 = time.time()
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    r = defaultdict(list)
    r.update(vars(args))
    r["save_dir_unique"] = save_dir_unique

    # get data
    story_names_train, story_names_test = get_story_names(args)
    if args.use_extract_only:
        # extract braindrive
        if args.use_eval_brain_drive:
            story_names_brain_drive = story_names.get_story_names(
                use_brain_drive=True, all=True)
            stim_brain_drive_delayed = feature_utils.get_features_full(
                args, args.feature_space,  args.qa_embedding_model, story_names_brain_drive,
                use_brain_drive=True, use_added_wordrate_feature=args.use_added_wordrate_feature)

        all_stories = story_names.get_story_names(all=True)
        random.shuffle(all_stories)
        feature_utils.get_features_full(
            args, args.feature_space, args.qa_embedding_model,
            all_stories, extract_only=True, use_added_wordrate_feature=args.use_added_wordrate_feature)

    print('loading features...')
    stim_test_delayed = feature_utils.get_features_full(
        args, args.feature_space, args.qa_embedding_model, story_names_test, use_added_wordrate_feature=args.use_added_wordrate_feature)
    stim_train_delayed = feature_utils.get_features_full(
        args, args.feature_space, args.qa_embedding_model, story_names_train, use_added_wordrate_feature=args.use_added_wordrate_feature)
    print('feature shapes before selection',
          stim_train_delayed.shape, stim_test_delayed.shape)

    # select features
    if args.feature_selection_alpha >= 0:
        print('selecting features...')
        r, stim_train_delayed, stim_test_delayed = feat_select.select_features(
            args, r, stim_train_delayed, stim_test_delayed,
            story_names_train, story_names_test)
    if args.use_random_subset_features:
        r, stim_train_delayed, stim_test_delayed = feat_select.select_random_feature_subset(
            args, r, stim_train_delayed, stim_test_delayed)
    elif args.single_question_idx >= 0:
        r, stim_train_delayed, stim_test_delayed = feat_select.select_single_feature(
            args, r, stim_train_delayed, stim_test_delayed)
    print('feature shapes after selection',
          stim_train_delayed.shape, stim_test_delayed.shape)

    print('loading resps...')
    if args.pc_components <= 0:
        resp_train, resp_test = response_utils.get_resps_full(
            args, args.subject, story_names_train, story_names_test)
    else:
        resp_train, resp_test, pca, scaler_train, scaler_test = response_utils.get_resps_full(
            args, args.subject, story_names_train, story_names_test)

    # overwrite resp_train with distill model predictions
    if args.distill_model_path is not None:
        resp_train = response_utils.get_resp_distilled(
            args, story_names_train)

    # fit model
    print('fitting regression...')
    r, model_params_to_save = fit_regression(
        args, r, stim_train_delayed, resp_train, stim_test_delayed, resp_test)

    # evaluate per voxel
    if args.pc_components > 0:
        resp_test = response_utils.load_response_wrapper(
            args, story_names_test, args.subject)
        r['corrs_test'] = evaluate_pc_model_on_each_voxel(
            args, stim_test_delayed, resp_test,
            model_params_to_save, pca, scaler_test)
        # model_params_to_save['pca'] = pca
        model_params_to_save['scaler_test'] = scaler_test
        model_params_to_save['scaler_train'] = scaler_train

        # compute weighted corrs_tune_pc
        # explained_var_weight = pca.explained_variance_[:args.pc_components]
        # explained_var_weight = explained_var_weight / \
        #     explained_var_weight.sum() * len(explained_var_weight)
        # r['corrs_tune_pc_weighted_mean'] = np.mean(
        #     explained_var_weight * r['corrs_tune_pc'])

    if args.use_eval_brain_drive and args.subject in story_names.TEST_BRAINDRIVE.keys():
        story_names_brain_drive = story_names.get_story_names(
            subject=args.subject, use_brain_drive=True)
        stim_brain_drive_delayed = feature_utils.get_features_full(
            args, args.feature_space, args.qa_embedding_model, story_names_brain_drive, use_brain_drive=True, use_added_wordrate_feature=args.use_added_wordrate_feature)
        resp_brain_drive = response_utils.load_response_wrapper(
            args, story_names_brain_drive, args.subject, use_brain_drive=True)
        r['corrs_brain_drive'] = evaluate_pc_model_on_each_voxel(
            args, stim_brain_drive_delayed, resp_brain_drive,
            model_params_to_save, pca, scaler_test)

    # add extra stats
    r = add_summary_stats(r, verbose=True)

    os.makedirs(save_dir_unique, exist_ok=True)
    joblib.dump(r, join(save_dir_unique, "results.pkl"))
    if args.encoding_model == 'ridge':
        joblib.dump(model_params_to_save, join(
            save_dir_unique, "model_params.pkl"))
    print(
        f"Succesfully completed in {(time.time() - t0)/60:0.1f} minutes, saved to {save_dir_unique}")
