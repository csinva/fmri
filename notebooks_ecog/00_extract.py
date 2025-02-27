from neuro.ecog.questions import QS_O1_DEC26, QS_O1_DEC26_2
from neuro.ecog.config import STORIES_POPULAR, STORIES_UNPOPULAR, STORIES_LOTR, ECOG_DIR
from imodelsx.qaemb.qaemb import QAEmb, get_sample_questions_and_examples
import neuro.ecog.questions as questions
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
import pandas as pd
from tqdm import tqdm
from os.path import join
import matplotlib.pyplot as plt
import os
from os.path import expanduser
import argparse
from copy import deepcopy
import logging
import random
from os.path import join
import numpy as np
import joblib
import os.path
from neuro.features.questions.gpt4 import QS_35_STABLE
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
    parser.add_argument(
        '--questions',
        type=str,
        default='o1_dec26',
        help='which questions to use',
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
        sec_window = float(setting.split('_')[-1])
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
    # stories_to_run = ['___podcasts-story___']
    stories_to_run = ['Cars 2']  # this is partially completed
    # stories_to_run = STORIES_LOTR[:1] # this is completed
    # stories_to_run = STORIES_LOTR
    # stories_to_run = STORIES_POPULAR + STORIES_UNPOPULAR
    # stories_to_run = STORIES_POPULAR

    # get args
    parser = argparse.ArgumentParser()
    parser_without_computational_args = add_main_args(parser)
    parser = add_computational_args(
        deepcopy(parser_without_computational_args))
    args = parser.parse_args()

    if args.questions == 'o1_dec26':
        qs_to_run = QS_O1_DEC26 + QS_O1_DEC26_2
        suffix_qs = ''
    elif args.questions == 'qs_35_stable':
        qs_to_run = QS_35_STABLE
        suffix_qs = '___qs_35_stable'

    # set up logging
    logger = logging.getLogger()
    # logging.basicConfig(level=logging.INFO)
    logging.basicConfig(level=logging.ERROR)

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
    if not args.checkpoint.startswith('gpt-4'):
        rng.shuffle(stories_to_run)
        rng.shuffle(qs_to_run)

    checkpoint_clean = args.checkpoint.replace('/', '___')

    transcript_folders = os.listdir(join(ECOG_DIR, 'data', 'transcripts'))
    output_dir_clean = join(ECOG_DIR, f'features{suffix_qs}',
                            checkpoint_clean, args.setting)
    output_dir_raw = join(ECOG_DIR, f'features_raw{suffix_qs}',
                          checkpoint_clean, args.setting)
    os.makedirs(output_dir_clean, exist_ok=True)
    os.makedirs(output_dir_raw, exist_ok=True)

    # breakpoint()
    if args.checkpoint.startswith('gpt-4'):
        CACHE_DIR = expanduser("~/cache_qa_gpt4_ecog")
    else:
        CACHE_DIR = None
    qa_embedder = QAEmb(
        questions=[],
        checkpoint=args.checkpoint,
        batch_size=args.batch_size,
        CACHE_DIR=CACHE_DIR,
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
        # print(texts[:40])
        for q in tqdm(qs_to_run, desc='question', leave=False):
            output_file_q = join(output_dir_raw, f'{story_fname}___{q}.pkl')

            if os.path.exists(output_file_q):
                answers_dict[q] = joblib.load(output_file_q).astype(bool)
                # print(f'Loaded {output_file_q}')
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
        q1 = 'Does the text reference a person’s name?'
        q2 = 'Does the text contain a number?'
        if q1 in answers_df:
            print('these should be names', list(
                answers_df[answers_df[q1] > 0][q1].index)[:10])
        elif q2 in answers_df:
            print('these should be numbers', list(
                answers_df[answers_df[q2] > 0][q2].index)[:10])

    # save results
    os.makedirs(save_dir_unique, exist_ok=True)
    joblib.dump(
        {'Model succeeded'}, join(save_dir_unique, "results.pkl")
    )  # caching requires that this is called results.pkl
    logging.info("Succesfully completed :)\n\n")
