import datasets
import numpy as np
from typing import List
import pandas as pd

from tqdm import tqdm


def get_dsets(dataset: str, seed: int = 1, subsample_frac: float = None):

    # select subsets
    def get_dset():
        return datasets.load_dataset(dataset)
    if dataset == 'tweet_eval':
        def get_dset():
            return datasets.load_dataset('tweet_eval', 'hate')
    elif dataset == 'moral_stories':
        def get_dset():
            return datasets.load_dataset(
                'demelin/moral_stories', 'cls-action+norm-minimal_pairs')
    elif dataset.startswith('ethics-'):
        # ['commonsense', 'deontology', 'justice', 'utilitarianism', 'virtue']:
        def get_dset():
            return datasets.load_dataset('metaeval/ethics', dataset.replace('ethics-', ''))

    elif dataset.startswith('probing-'):
        # ['subj_number', 'word_content', 'obj_number', 'past_present', 'sentence_length',
        # 'top_constituents', 'tree_depth', 'coordination_inversion', 'odd_man_out', 'bigram_shift']
        def get_dset():
            return datasets.load_dataset('metaeval/linguisticprobing', dataset.replace('probing-', ''))

    # default keys
    dataset_key_text = 'text'
    dataset_key_label = 'label'
    val_dset_key = 'validation'

    # changes
    if dataset == 'sst2':
        dataset_key_text = 'sentence'
    elif dataset == 'poem_sentiment':
        dataset_key_text = 'verse_text'
    elif dataset == 'trec':
        dataset_key_label = 'coarse_label'  # 'label-coarse'
        val_dset_key = 'test'
    elif dataset == 'go_emotions':
        dataset_key_label = 'labels'
    elif dataset.startswith('probing-'):
        dataset_key_text = 'sentence'

    dset = get_dset()['train']
    dset = dset.shuffle(seed=seed)
    if subsample_frac is not None and subsample_frac > 0:
        dset = dset[:int(len(dset) * subsample_frac)]

    def remove_multilabel(X: List[str], y: List[List]):
        idxs = np.array(pd.Series(y).apply(len) == 1)
        X = np.array(X)[idxs].tolist()
        y = np.array([yy[0] for yy in y])[idxs].tolist()
        return X, y

    def get_X(d, dataset_key_text, dataset):
        if not dataset == 'moral_stories':
            return d[dataset_key_text]
        if dataset == 'moral_stories':
            text = np.vstack((d['immoral_action'], d['moral_action'])).T
            texts = []
            idxs = np.array(d['label']).astype(int)
            for i, idx in enumerate(tqdm(idxs)):
                texts.append(d['norm'][i] + ' ' + text[i, idx])
        return texts

    X = get_X(dset, dataset_key_text, dataset)
    y = dset[dataset_key_label]

    dset_test = get_dset()[val_dset_key]
    # dset_test = dset_test.select(np.random.choice(len(dset_test), size=300, replace=False))
    X_test = get_X(dset_test, dataset_key_text, dataset)
    y_test = dset_test[dataset_key_label]

    if dataset == 'go_emotions':
        X, y = remove_multilabel(X, y)
        X_test, y_test = remove_multilabel(X_test, y_test)

    return X, y, X_test, y_test
