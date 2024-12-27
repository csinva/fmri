from questions import QS_O1_DEC26
from treebank import STORIES_POPULAR, STORIES_UNPOPULAR, ECOG_DIR
from questions import QS_O1_DEC26
from imodelsx.qaemb.qaemb import QAEmb, get_sample_questions_and_examples
import questions
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


stories_to_run = STORIES_POPULAR
qs_to_run = QS_O1_DEC26
checkpoint = 'meta-llama/Meta-Llama-3-8B-Instruct'
checkpoint_clean = checkpoint.replace('/', '___')
setting = 'words'

if __name__ == '__main__':
    transcript_folders = os.listdir(join(ECOG_DIR, 'data', 'transcripts'))
    output_dir_clean = join(ECOG_DIR, 'features', checkpoint_clean, setting)
    output_dir_raw = join(ECOG_DIR, 'features_raw', checkpoint_clean, setting)
    os.makedirs(output_dir_clean, exist_ok=True)
    os.makedirs(output_dir_raw, exist_ok=True)

    qa_embedder = QAEmb(
        questions=[QS_O1_DEC26[0]],
        checkpoint=checkpoint,
        batch_size=512,
        # CACHE_DIR=expanduser("~/cache_qa_ecog"),
        CACHE_DIR=None,
    )

    def get_texts(features_df, setting='words', replace_nan_with_empty_string=True):
        if setting == 'words':
            texts = features_df['text'].values.flatten()
        if replace_nan_with_empty_string:
            texts = [t if isinstance(t, str) else '""' for t in texts]
        return texts

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

        answers_dict = {}
        for q in tqdm(qs_to_run, desc='question', leave=False):
            output_file_q = join(output_dir_raw, f'{story_fname}___{q}.pkl')

            if os.path.exists(output_file_q):
                answers_dict[q] = joblib.load(output_file_q).astype(bool)
                print(f'Loaded {output_file_q}')
            else:
                texts = get_texts(features_df, setting='words')
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
