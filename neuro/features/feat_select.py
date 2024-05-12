from sklearn.linear_model import enet_path, MultiTaskElasticNet
from os.path import join, dirname
import neuro.config as config
import joblib
from neuro.data import response_utils
import os
import os.path
import logging
from os.path import join, dirname
import numpy as np
import joblib
import os
from neuro.data import response_utils
import random
import time


def select_features(args, r, stim_train_delayed, stim_test_delayed, story_names_train, story_names_test):
    # remove delays from stim
    stim_train = stim_train_delayed[:,
                                    :stim_train_delayed.shape[1] // args.ndelays]

    # coefs is (n_targets, n_features, n_alphas)
    cache_dir = join(config.root_dir, 'qa', 'sparse_feats_shared')
    cache_file = join(
        cache_dir,
        f"{args.feature_space.replace('/', '-')}___qa_questions_version={args.qa_questions_version}___{args.qa_embedding_model.replace('/','-')}",
        f'seed={args.seed}___feature_selection_frac={args.feature_selection_frac:.2f}___feature_selection_alpha={args.feature_selection_alpha:.2e}.joblib')
    os.makedirs(dirname(cache_file), exist_ok=True)

    if os.path.exists(cache_file):
        coef_enet = joblib.load(cache_file)
        print('Loaded from cache:', cache_file)
    else:
        print('couldn\'t find cache file:', cache_file, '\n\tfitting now...')
        # get special resps by concatenating across subjects
        resp_train_shared = response_utils.get_resps_full(
            args, 'shared', story_names_train, story_names_test)

        # randomly subsample based on seed
        if 0 < args.feature_selection_frac < 1:
            rng = np.random.RandomState(args.seed)
            idxs_subsample = rng.choice(
                np.arange(stim_train.shape[0]), int(args.feature_selection_frac * stim_train.shape[0]), replace=False)
            stim_train = stim_train[idxs_subsample]
            resp_train_shared = resp_train_shared[idxs_subsample]

        m = MultiTaskElasticNet(
            alpha=args.feature_selection_alpha,
            l1_ratio=0.9,
            max_iter=5000,
            random_state=args.seed,
        ).fit(stim_train, resp_train_shared)
        coef_enet = m.coef_
        joblib.dump(coef_enet, cache_file)
        logging.info(
            f"Succesfully completed feature selection (frac nonzero: {np.any(np.abs(coef_enet) > 0, axis=0).mean()}). Saved to {cache_file}")
    if args.subject == 'shared':
        print(
            f"Stopping because {args.subject=}")
        exit(0)

    # pick the coefs
    # coef_enet = coefs_enet[:, :, args.feature_selection_alpha_index]
    coef_nonzero = np.any(np.abs(coef_enet) > 0, axis=0)
    # r['alpha'] = alphas_enet[args.feature_selection_alpha_index]
    r['weights_enet'] = coef_enet
    r['weight_enet_mask'] = coef_nonzero
    r['weight_enet_mask_num_nonzero'] = coef_nonzero.sum()

    # mask stim_delayed based on nonzero coefs (need to repeat by args.ndelays)
    coef_nonzero_rep = np.tile(
        coef_nonzero.flatten(), args.ndelays).flatten()
    stim_train_delayed = stim_train_delayed[:, coef_nonzero_rep]
    stim_test_delayed = stim_test_delayed[:, coef_nonzero_rep]

    return r, stim_train_delayed, stim_test_delayed
