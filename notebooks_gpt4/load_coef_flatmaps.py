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
import numpy as np
import sasc.viz
import joblib
import dvu
import sys
sys.path.append('../notebooks')
flatmaps_per_question = __import__('06_flatmaps_per_question')


results_dir = '/home/chansingh/mntv1/deep-fMRI/encoding/jun16_gpt4'
rr, cols_varied, mets = analyze_helper.load_clean_results(results_dir)


def _load_coefs_35questions(subject='S02'):
    # # load questions model weights
    # results_dir = analyze_helper.best_results_dir
    # rr, cols_varied, mets = analyze_helper.load_clean_results(results_dir)
    # metric_sort = 'corrs_tune_pc_weighted_mean'

    # # pick model to interpret
    # r = rr
    # r = r[r.qa_questions_version == 'v3_boostexamples_merged']
    # r = r[r.num_stories == -1]
    # r = r[r.weight_enet_mask_num_nonzero == 35]
    # r = r[r.feature_space == 'qa_embedder']
    # cols_varied = [c for c in cols_varied if not c in ['num_stories',
    #                                                    'feature_selection_alpha', 'feature_selection_stability_seeds']]
    # args0 = r[r.subject == subject].iloc[0]
    # weights, weights_pc = flatmaps_per_question.get_weights_top(args0)
    # qs_selected = questions[args0['weight_enet_mask']]
    # df_w_selected35 = pd.DataFrame({'question': qs_selected, 'weights': [
    #                     w for w in weights]}).set_index('question')
    # joblib.dump((args0, qs_selected, df_w_selected35),
    # '../qa_results/processed/selected_weights.pkl')

    args0, qs_selected, df_w_selected35 = joblib.load(
        '../qa_results/processed/selected_weights.pkl')
    return df_w_selected35


def _load_coefs_individual(subject='S02'):
    r = rr
    r = r[r.subject == subject]
    r = r[r.use_added_wordrate_feature == 0]
    r = r[r.feature_space == 'qa_embedder']

    weights = np.array([
        flatmaps_per_question.get_weights_top(r.iloc[i])[0]
        for i in tqdm(range(len(r)))
    ]).squeeze()
    qs_selected = r['qa_questions_version']
    df_w_individual = pd.DataFrame({'question': qs_selected, 'weights': [
        w for w in weights]}).set_index('question')
    # joblib.dump((r, qs_selected, df_w_individual),
    # '../qa_results/processed/individual_weights.pkl')
    # r, qs_selected, df_w_individual = joblib.load(
    # '../qa_results/processed/individual_weights.pkl')
    return df_w_individual


def _load_coefs_individual_wordrate(subject='S02'):
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


def _load_coefs_wordrate(subject='S02'):
    r = rr
    r = r[r.subject == 'S02']
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
