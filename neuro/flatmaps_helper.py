from copy import deepcopy
import joblib
import numpy as np
import sys
from os.path import abspath, dirname, join
try:
    from sasc.config import FMRI_DIR, STORIES_DIR, RESULTS_DIR, CACHE_DIR, cache_ngrams_dir, regions_idxs_dir
except ImportError:
    repo_path = dirname(dirname(dirname(abspath(__file__))))
    RESULTS_DIR = join(repo_path, 'results')


def load_flatmaps(normalize_flatmaps, load_timecourse=False, explanations_only=False):
    # S02
    gemv_flatmaps_default = joblib.load(join(
        RESULTS_DIR, "processed", "flatmaps_all", "UTS02", "default_pilot", 'resps_avg_dict_pilot.pkl'))
    gemv_flatmaps_qa = joblib.load(join(
        RESULTS_DIR, "processed", "flatmaps_all", "UTS02", 'qa_pilot5', 'resps_avg_dict_pilot5.pkl'))
    gemv_flatmaps_roi = joblib.load(join(
        RESULTS_DIR, "processed", "flatmaps_all", "UTS02", 'roi_pilot5', 'resps_avg_dict_pilot5.pkl'))

    gemv_flatmaps_roi_custom = joblib.load(join(
        RESULTS_DIR, "processed", "flatmaps_all", 'UTS02', 'roi_pilot6', 'resps_avg_dict_pilot6.pkl'))
    gemv_flatmaps_dict_S02 = gemv_flatmaps_default | gemv_flatmaps_qa | gemv_flatmaps_roi | gemv_flatmaps_roi_custom
    # gemv_flatmaps_dict_S02 = gemv_flatmaps_roi_custom

    # S03
    gemv_flatmaps_default = joblib.load(join(
        RESULTS_DIR, "processed", "flatmaps_all", 'UTS03', 'default', 'resps_avg_dict_pilot3.pkl'))
    gemv_flatmaps_roi_custom1 = joblib.load(join(
        RESULTS_DIR, "processed", "flatmaps_all", 'UTS03', 'roi_pilot7', 'resps_avg_dict_pilot7.pkl'))
    gemv_flatmaps_roi_custom2 = joblib.load(join(
        RESULTS_DIR, "processed", "flatmaps_all", 'UTS03', 'roi_pilot8', 'resps_avg_dict_pilot8.pkl'))
    # gemv_flatmaps_dict_S03 = gemv_flatmaps_default | gemv_flatmaps_roi_custom1 | gemv_flatmaps_roi_custom2
    gemv_flatmaps_dict_S03 = gemv_flatmaps_roi_custom1 | gemv_flatmaps_roi_custom2

    if load_timecourse:
        gemv_flatmaps_default = joblib.load(join(
            RESULTS_DIR, "processed", "flatmaps_all", "UTS02", "default_pilot", 'resps_concat_dict_pilot.pkl'))
        gemv_flatmaps_qa = joblib.load(join(
            RESULTS_DIR, "processed", "flatmaps_all", "UTS02", 'qa_pilot5', 'resps_concat_dict_pilot5.pkl'))
        gemv_flatmaps_roi = joblib.load(join(
            RESULTS_DIR, "processed", "flatmaps_all", "UTS02", 'roi_pilot5', 'resps_concat_dict_pilot5.pkl'))
        gemv_flatmaps_roi_custom = joblib.load(join(
            RESULTS_DIR, "processed", "flatmaps_all", 'UTS02', 'roi_pilot6', 'resps_concat_dict_pilot6.pkl'))

        gemv_flatmaps_dict_S02_timecourse = gemv_flatmaps_default | gemv_flatmaps_qa | gemv_flatmaps_roi | gemv_flatmaps_roi_custom
        # gemv_flatmaps_dict_S02_timecourse = gemv_flatmaps_roi_custom

        gemv_flatmaps_roi_custom1 = joblib.load(join(
            RESULTS_DIR, "processed", "flatmaps_all", 'UTS03', 'roi_pilot7', 'resps_concat_dict_pilot7.pkl'))
        gemv_flatmaps_roi_custom2 = joblib.load(join(
            RESULTS_DIR, "processed", "flatmaps_all", 'UTS03', 'roi_pilot8', 'resps_concat_dict_pilot8.pkl'))
        gemv_flatmaps_dict_S03_timecourse = gemv_flatmaps_roi_custom1 | gemv_flatmaps_roi_custom2

        return gemv_flatmaps_dict_S02, gemv_flatmaps_dict_S03, gemv_flatmaps_dict_S02_timecourse, gemv_flatmaps_dict_S03_timecourse

    # normalize flatmaps
    if normalize_flatmaps:
        for k, v in gemv_flatmaps_dict_S03.items():
            flatmap_unnormalized = gemv_flatmaps_dict_S03[k]
            gemv_flatmaps_dict_S03[k] = (
                flatmap_unnormalized - flatmap_unnormalized.mean()) / flatmap_unnormalized.std()
        for k, v in gemv_flatmaps_dict_S02.items():
            flatmap_unnormalized = gemv_flatmaps_dict_S02[k]
            gemv_flatmaps_dict_S02[k] = (
                flatmap_unnormalized - flatmap_unnormalized.mean()) / flatmap_unnormalized.std()

    return gemv_flatmaps_dict_S02, gemv_flatmaps_dict_S03
