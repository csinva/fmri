from neuro.treebank.questions import QS_O1_DEC26, QS_O1_DEC26_2
from neuro.treebank.config import STORIES_POPULAR, STORIES_UNPOPULAR, ECOG_DIR
from imodelsx.qaemb.qaemb import QAEmb, get_sample_questions_and_examples
import neuro.treebank.questions as questions
from math import ceil
from numpy.linalg import norm
from copy import deepcopy
import json
import pickle as pkl
from os.path import dirname
import imodelsx.util
from pprint import pprint
import joblib
import numpy as np
from typing import List
import sys
from os.path import expanduser
import pandas as pd
from tqdm import tqdm
from os.path import join
import matplotlib.pyplot as plt
import os
import argparse
from copy import deepcopy
import logging
import random
from collections import defaultdict
from os.path import join
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
import joblib
import imodels
import inspect
import os.path
import imodelsx.cache_save_utils
path_to_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# initialize args
def add_main_args(parser):
    """Caching uses the non-default values from argparse to name the saving directory.
    Changing the default arg an argument will break cache compatibility with previous runs.
    """

    # dataset args
    parser.add_argument(
        "--checkpoint", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="name of QA model"
    )
    parser.add_argument(
        '--setting', type=str, default='words', help='how to chunk texts'
    )
    # training misc args
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument(
        "--save_dir",
        type=str,
        default=join(path_to_repo, "results"),
        help="directory for saving",
    )

    return parser


def add_computational_args(parser):
    """Arguments that only affect computation and not the results (shouldnt use when checking cache)"""
    parser.add_argument(
        "--use_cache",
        type=int,
        default=1,
        choices=[0, 1],
        help="whether to check for cache",
    )
    parser.add_argument(
        '--seed_stories',
        type=int,
        default=1,
        help='seed for order that stories are processed in',
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=512,
        help='batch size for QA model',
    )
    return parser


def get_texts(features_df, setting='words'):
    if setting == 'words':
        texts = features_df['text'].values.flatten()
    elif 'sec' in setting:
        # get num from string
        sec_window = int(setting.split('_')[-1])
        texts = []
        for i in tqdm(range(0, len(features_df))):
            row = features_df.iloc[i]
            time_end = row['end']
            time_start = time_end - sec_window
            ngram = features_df[
                (features_df['end'] >= time_start) & (
                    features_df['end'] <= time_end)
            ]['text'].values.tolist()
            assert len(ngram) > 0, f'{i=} {ngram=} {time_start=} {time_end=}'
            texts.append(' '.join(ngram))

    return texts


if __name__ == "__main__":
    stories_to_run = STORIES_POPULAR + STORIES_UNPOPULAR
    qs_to_run = QS_O1_DEC26 + QS_O1_DEC26_2

    # get args
    parser = argparse.ArgumentParser()
    parser_without_computational_args = add_main_args(parser)
    parser = add_computational_args(
        deepcopy(parser_without_computational_args))
    args = parser.parse_args()

    # set up logging
    logger = logging.getLogger()
    logging.basicConfig(level=logging.INFO)

    # set up saving directory + check for cache
    already_cached, save_dir_unique = imodelsx.cache_save_utils.get_save_dir_unique(
        parser, parser_without_computational_args, args, args.save_dir
    )

    if args.use_cache and already_cached:
        logging.info(f"cached version exists! Successfully skipping :)\n\n\n")
        exit(0)
    for k in sorted(vars(args)):
        logger.info("\t" + k + " " + str(vars(args)[k]))
    logging.info(f"\n\n\tsaving to " + save_dir_unique + "\n")

    # set seed
    np.random.seed(args.seed)
    random.seed(args.seed)

    rng = np.random.default_rng(args.seed_stories)
    rng.shuffle(stories_to_run)
    rng.shuffle(qs_to_run)

    checkpoint_clean = args.checkpoint.replace('/', '___')

    transcript_folders = os.listdir(join(ECOG_DIR, 'data', 'transcripts'))
    output_dir_clean = join(ECOG_DIR, 'features',
                            checkpoint_clean, args.setting)
    output_dir_raw = join(ECOG_DIR, 'features_raw',
                          checkpoint_clean, args.setting)
    os.makedirs(output_dir_clean, exist_ok=True)
    os.makedirs(output_dir_raw, exist_ok=True)

    qa_embedder = QAEmb(
        questions=[],
        checkpoint=args.checkpoint,
        batch_size=args.batch_size,
        # CACHE_DIR=expanduser("~/cache_qa_ecog"),
        CACHE_DIR=None,
    )

    for story in tqdm(stories_to_run, desc='stories'):
        # for story in stories_unpopular:
        story_fname = (
            story.replace(' ', '-').lower()
            .replace('lord-of-the-rings', 'lotr')
            .replace('spiderman', 'spider-man')
            .replace('the-incredibles', 'incredibles')
            .replace('antman', 'ant-man')
            .replace('mr.', 'mr')
            .replace('spider-man-homecoming', 'spider-man-3-homecoming')
        )
        assert story_fname in transcript_folders, f'{story_fname} not found'
        features_df = pd.read_csv(
            join(ECOG_DIR, 'data', 'transcripts', story_fname, 'features.csv'))
        features_df['end'] = features_df['end'].interpolate()
        features_df['text'] = features_df['text'].fillna('')

        answers_dict = {}
        texts = get_texts(features_df, setting=args.setting)
        for q in tqdm(qs_to_run, desc='question', leave=False):
            output_file_q = join(output_dir_raw, f'{story_fname}___{q}.pkl')

            if os.path.exists(output_file_q):
                answers_dict[q] = joblib.load(output_file_q).astype(bool)
                print(f'Loaded {output_file_q}')
            else:
                assert len(texts) == len(
                    features_df), f'{len(texts)=} {len(features_df)=}'
                qa_embedder.questions = [q]
                answers = qa_embedder(
                    texts, speed_up_with_unique_calls=True).flatten().astype(bool)
                joblib.dump(answers, output_file_q)
                answers_dict[q] = answers
        answers_df = pd.DataFrame(answers_dict, index=texts)
        answers_df.to_pickle(join(output_dir_clean, f'{story_fname}.pkl'))
        answers_df.to_csv(join(output_dir_clean, f'{story_fname}.csv'))

        # spot check
        q = 'Does the text reference a person’s name?'
        print('these should be names', list(
            answers_df[answers_df[q] > 0][q].index))

    # save results
    os.makedirs(save_dir_unique, exist_ok=True)
    joblib.dump(
        {'Model succeeded'}, join(save_dir_unique, "results.pkl")
    )  # caching requires that this is called results.pkl
    logging.info("Succesfully completed :)\n\n")
