from neuro.features.qa_questions import get_questions, get_merged_questions_v3_boostexamples
from neuro import analyze_helper, viz
from neuro.config import repo_dir
from sasc.config import FMRI_DIR, STORIES_DIR, RESULTS_DIR
from tqdm import tqdm
import sys
from ridge_utils.DataSequence import DataSequence
import pandas as pd
import os
import matplotlib.pyplot as plt
import cortex
import seaborn as sns
from os.path import join
from collections import defaultdict
from neuro.config import repo_dir, PROCESSED_DIR
import numpy as np
import sasc.viz
import joblib
import dvu
import sys

from neuro.features.questions.gpt4 import QS_35_STABLE
sys.path.append('../notebooks')
flatmaps_per_question = __import__('06_flatmaps_per_question')


def _load_coefs_shapley(rr, subject='S02', qa_questions_version='v3_boostexamples_merged'):
    r = rr
    r = r[r['qa_questions_version'] == qa_questions_version]
    r = r[r['subject'] == subject]
    r = r[r['use_random_subset_features'] == 1]
    # r = r[r['use_added_wordrate_feature'] == 1]
    # print(r.shape, rr_shapley.shape)
    if len(r) < 10:
        # print(
        # f'\tskipping shap with only {len(r)} runs', subject, qa_questions_version)
        return
    # print('num shapley runs', len(r))
    row = r.iloc[0]

    if qa_questions_version == 'v3_boostexamples_merged':
        questions = get_merged_questions_v3_boostexamples()
        questions = np.array(questions)[row.weight_enet_mask]
    elif qa_questions_version == 'qs_35':
        questions = QS_35_STABLE
    else:
        questions = get_questions(qa_questions_version)

    flatmaps_shapley = defaultdict(list)
    for i in tqdm(range(len(r))):
        row = r.iloc[i]
        weights, weights_pc = flatmaps_per_question.get_weights_top(row)

        # note: wordrate feature gets skipped because it's last
        question_indexes = np.arange(len(questions))[
            row['weight_random_mask'][:len(questions)]]

        for w_idx, q_idx in enumerate(question_indexes):
            flatmaps_shapley[questions[q_idx]].append(weights[w_idx])

    for q, w in flatmaps_shapley.items():
        flatmaps_shapley[q] = np.vstack(w).mean(axis=0)
    return flatmaps_shapley


def _load_coefs_full(r, subject='S02', qa_questions_version='v3_boostexamples_merged', use_added_wordrate_feature=0):
    r = r[r.qa_questions_version == qa_questions_version]
    r = r[r.feature_space == 'qa_embedder']
    r = r[r.subject == subject]
    r = r[r.use_random_subset_features == 0]
    r = r[r.use_added_wordrate_feature == use_added_wordrate_feature]
    r = r[r.single_question_idx == -1]
    if len(r) == 0:
        # print('\tskipping', subject, qa_questions_version,
        #   use_added_wordrate_feature)
        return
    assert len(r) == 1
    args0 = r[r.subject == subject].iloc[0]
    weights, weights_pc = flatmaps_per_question.get_weights_top(args0)

    if qa_questions_version == 'v3_boostexamples_merged':
        questions = get_merged_questions_v3_boostexamples()
        if use_added_wordrate_feature:
            questions = questions + ['wordrate']
        questions = np.array(questions)[
            args0.weight_enet_mask.astype(bool)]
        print('\tweight_enet_mask', args0.weight_enet_mask.sum(),
              args0.weight_enet_mask.shape)
        print('\tquestions', len(questions))  # , questions[:10])
    elif qa_questions_version == 'qs_35':
        questions = QS_35_STABLE
        if use_added_wordrate_feature:
            questions = questions + ['wordrate']
    else:
        questions = get_questions(qa_questions_version)
        if use_added_wordrate_feature:
            questions = questions + ['wordrate']

    print('\t', len(questions), 'questions', len(weights), 'weights')

    corrs_test = r['corrs_test']
    if qa_questions_version == 'v3_boostexamples_merged':
        joblib.dump(corrs_test, join(PROCESSED_DIR,
                                     subject, 'corrs_test_35.pkl'))
    elif qa_questions_version == 'v1neurosynth':
        joblib.dump(corrs_test, join(PROCESSED_DIR,
                                     subject, 'corrs_test_neurosynth.pkl'))
    elif qa_questions_version == 'qs_35':
        joblib.dump(corrs_test, join(PROCESSED_DIR,
                                     subject, 'corrs_test_qs_35.pkl'))
        print('corr', args0['corrs_test_mean'])

    return {q: w for q, w in zip(questions, weights)}


def _load_coefs_individual(r, subject='S02', qa_questions_version='v3_boostexamples_merged'):
    r = r[r.qa_questions_version == qa_questions_version]
    r = r[r.feature_space == 'qa_embedder']
    r = r[r.subject == subject]
    r = r[r.use_random_subset_features == 0]
    r = r[r.use_added_wordrate_feature == 0]
    r = r[r.single_question_idx > -1]
    args0 = r[r.subject == subject].iloc[0]

    if qa_questions_version == 'v3_boostexamples_merged':
        questions = get_merged_questions_v3_boostexamples()
        questions = np.array(questions)[args0.weight_enet_mask]
    else:
        questions = get_questions(qa_questions_version)

    # print(sorted(r.single_question_idx.unique()))
    assert r.single_question_idx.nunique() == len(
        questions), f'{r.single_question_idx.nunique()} != {len(questions)}'
    assert len(r) == len(questions), f'{len(r)} != {len(questions)}'
    args0 = r[r.subject == subject].iloc[0]
    weights = np.array([
        flatmaps_per_question.get_weights_top(r.iloc[i])[0]
        for i in tqdm(range(len(r)))
    ]).squeeze()

    return {q: w for q, w in zip(questions, weights)}


def _load_coefs_individual_gpt4(rr, subject='S02', use_added_wordrate_feature=0):
    r = rr
    r = r[r.subject == subject]
    r = r[r.use_added_wordrate_feature == use_added_wordrate_feature]
    r = r[r.feature_space == 'qa_embedder']
    r = r[r.qa_embedding_model == 'gpt4']
    r = r[r.qa_questions_version.str.endswith('?')]  # individual question
    print(subject, 'init shape', rr.shape, 'this shape',
          r.shape, 'subj shape', rr[rr.subject == subject].shape)

    weights_list = [
        flatmaps_per_question.get_weights_top(r.iloc[i])[0]
        for i in tqdm(range(len(r)))
    ]
    questions = r['qa_questions_version']

    if use_added_wordrate_feature == 1:
        weights_list = [w[0] for w in weights_list]

    weights = np.array(weights_list).squeeze()
    corrs_test_dict = {
        q: r.iloc[i]['corrs_test']
        for i, q in enumerate(questions)
    }
    joblib.dump(corrs_test_dict, join(PROCESSED_DIR,
                                      subject, 'corrs_test_individual_gpt4_qs_35.pkl'))
    # print('weights', weights.shape)

    # print(r.columns)

    return {q: w for q, w in zip(questions, weights)}


def _load_coefs_individual_wordrate(rr, subject='S02'):
    r = rr
    r = r[r.subject == subject]
    r = r[r.use_added_wordrate_feature == 1]
    r = r[r.feature_space == 'qa_embedder']

    weights = np.array([
        flatmaps_per_question.get_weights_top(r.iloc[i])[0]
        for i in tqdm(range(len(r)))
    ])[:, 0, :].squeeze()
    qs_selected = r['qa_questions_version']
    df_w_individual_wordrate = pd.DataFrame({'question': qs_selected, 'weights': [
        w for w in weights]}).set_index('question')
    # joblib.dump((r, qs_selected, df_w_individual_wordrate),
    # '../qa_results/processed/individual_weights_wordrate.pkl')
    # r, qs_selected, df_w_individual_wordrate = joblib.load(
    # '../qa_results/processed/individual_weights_wordrate.pkl')
    # return df_w_individual_wordrate
    return {q: w for q, w in zip(qs_selected, weights)}


def _load_coefs_wordrate(rr, subject='S02'):
    r = rr
    r = r[r.subject == subject]
    r = r[r.use_added_wordrate_feature == 0]
    r = r[r.feature_space == 'wordrate']

    weights = flatmaps_per_question.get_weights_top(r.iloc[0])[0]
    qs_selected = list(rr['qa_questions_version'].unique())
    df_w_wordrate_alone = pd.DataFrame({'question': qs_selected, 'weights': [
        weights for i in range(len(qs_selected))]}).set_index('question')
    # joblib.dump((r, qs_selected, df_w_wordrate_alone),
    # '../qa_results/processed/weights_wordrate_alone.pkl')
    # r, qs_selected, df_w_wordrate_alone = joblib.load(
    # '../qa_results/processed/weights_wordrate_alone.pkl')
    return df_w_wordrate_alone
