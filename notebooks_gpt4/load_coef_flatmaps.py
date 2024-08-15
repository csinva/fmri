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
sys.path.append('../notebooks')
flatmaps_per_question = __import__('06_flatmaps_per_question')


def _load_coefs_shapley(rr, subject='S02', qa_questions_version='v3_boostexamples_merged'):
    r = rr
    r = r[r['qa_questions_version'] == qa_questions_version]
    r = r[r['subject'] == subject]
    r = r[r['use_random_subset_features'] == 1]
    # print(r.shape, rr_shapley.shape)
    row = r.iloc[0]

    if qa_questions_version == 'v3_boostexamples_merged':
        questions = get_merged_questions_v3_boostexamples()
        questions = np.array(questions)[row.weight_enet_mask]
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


def _load_coefs_full(r, subject='S02', qa_questions_version='v3_boostexamples_merged'):
    r = r[r.qa_questions_version == qa_questions_version]
    r = r[r.feature_space == 'qa_embedder']
    r = r[r.subject == subject]
    r = r[r.use_random_subset_features == 0]
    r = r[r.use_added_wordrate_feature == 0]
    r = r[r.single_question_idx == -1]
    assert len(r) == 1
    args0 = r[r.subject == subject].iloc[0]
    weights, weights_pc = flatmaps_per_question.get_weights_top(args0)

    if qa_questions_version == 'v3_boostexamples_merged':
        questions = get_merged_questions_v3_boostexamples()
        questions = np.array(questions)[args0.weight_enet_mask]
    else:
        questions = get_questions(qa_questions_version)

    # qs_selected = questions[args0['weight_enet_mask']]
    df_w_full = pd.DataFrame({'question': questions, 'weights': [
        w for w in weights]}).set_index('question')
    r[['subject', 'feature_selection_alpha', 'use_added_wordrate_feature',
        'qa_questions_version', 'single_question_idx', 'corrs_test_mean']]  # .value_counts()

    corrs_test = r['corrs_test']
    if qa_questions_version == 'v3_boostexamples_merged':
        joblib.dump(corrs_test, join(PROCESSED_DIR,
                                     subject, 'corrs_test_35.pkl'))
    else:
        joblib.dump(corrs_test, join(PROCESSED_DIR,
                                     subject, 'corrs_test_neurosynth.pkl'))

    return df_w_full


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

    assert r.single_question_idx.nunique() == len(
        questions), f'{r.single_question_idx.nunique()} != {len(questions)}'
    assert len(r) == len(questions), f'{len(r)} != {len(questions)}'
    args0 = r[r.subject == subject].iloc[0]
    # weights, weights_pc = flatmaps_per_question.get_weights_top(args0)
    weights = np.array([
        flatmaps_per_question.get_weights_top(r.iloc[i])[0]
        for i in tqdm(range(len(r)))
    ]).squeeze()

    # qs_selected = questions[args0['weight_enet_mask']]
    df_w_full = pd.DataFrame({'question': questions, 'weights': [
        w for w in weights]}).set_index('question')
    r[['subject', 'feature_selection_alpha', 'use_added_wordrate_feature',
        'qa_questions_version', 'single_question_idx', 'corrs_test_mean']]  # .value_counts()

    return df_w_full


# def _load_coefs_individual(rr, subject='S02'):
#     r = rr
#     r = r[r.subject == subject]
#     r = r[r.use_added_wordrate_feature == 0]
#     r = r[r.feature_space == 'qa_embedder']
#     r = r[r.single_question_idx >= 0]

#     weights = np.array([
#         flatmaps_per_question.get_weights_top(r.iloc[i])[0]
#         for i in tqdm(range(len(r)))
#     ]).squeeze()
#     qs_selected = r['qa_questions_version']
#     df_w_individual = pd.DataFrame({'question': qs_selected, 'weights': [
#         w for w in weights]}).set_index('question')
#     # joblib.dump((r, qs_selected, df_w_individual),
#     # '../qa_results/processed/individual_weights.pkl')
#     # r, qs_selected, df_w_individual = joblib.load(
#     # '../qa_results/processed/individual_weights.pkl')
#     return df_w_individual


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
    return df_w_individual_wordrate


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
