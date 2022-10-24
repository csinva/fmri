import argparse
import datasets
import numpy as np
import os
from os.path import join
import logging
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from transformers import pipeline
from ridge_utils.SemanticModel import SemanticModel
from matplotlib import pyplot as plt
from typing import List
from sklearn.linear_model import RidgeCV, LogisticRegressionCV, LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from feature_spaces import em_data_dir, data_dir, results_dir, nlp_utils_dir
import feature_spaces
from collections import defaultdict
import pandas as pd
import pickle as pkl
from sklearn import metrics
from copy import deepcopy
import sys
import data
from tqdm import tqdm


def get_word_vecs(X: List[str], model='eng1000') -> np.ndarray:
    if 'eng1000' in model:
        sm = SemanticModel.load(join(em_data_dir, 'english1000sm.hf5'))
    elif 'glove' in model:
        sm = SemanticModel.load_np(join(nlp_utils_dir, 'glove'))
    # extract features
    X = [
        [word.encode('utf-8') for word in sentence.split(' ')]
        for sentence in X
    ]
    feats = sm.project_stims(X)
    return feats


def get_ngram_vecs(X: List[str], model='bert-3') -> np.ndarray:
    if model.lower().startswith('bert-sst2'):
        checkpoint = feature_spaces._FEATURE_CHECKPOINTS['bert-sst2']
    if model.lower().startswith('bert-'):
        checkpoint = 'bert-base-uncased'
    elif model.lower().startswith('roberta'):
        checkpoint = 'roberta-large'
    pipe = pipeline("feature-extraction",
                    model=checkpoint,
                    truncation=True,
                    device=0)
    ngram_size = int(model.split('-')[-1].split('__')[0])
    return feature_spaces.get_embs_from_text(
        X, embedding_function=pipe, ngram_size=ngram_size)


def get_embs_fmri(X: List[str], model, save_dir_fmri, perc_threshold=98) -> np.ndarray:
    if model.lower().startswith('bert-') or model.lower().startswith('roberta'):
        feats = get_ngram_vecs(X, model=model)
    else:
        feats = get_word_vecs(X, model=model)
    weights_npz = np.load(join(save_dir_fmri, 'weights.npz'))
    corrs_val = np.load(join(save_dir_fmri, 'corrs.npz'))['arr_0']

    weights = weights_npz['arr_0']
    # pretty sure this is right, but might be switched...
    ndelays = int(model[model.index('ndel=') + len('ndel='):])
    weights = weights.reshape(ndelays, -1, feats.shape[-1])
    # delays for coefs are not stored next to each other!!
    # (see cell 25 file:///Users/chandan/Downloads/speechmodeltutorial-master/SpeechModelTutorial%20-%20Pre-run.html)
    # weights = weights.reshape(-1, N_DELAYS, feats.shape[-1])
    weights = weights.mean(axis=0).squeeze()  # mean over delays dimension...
    embs = feats @ weights.T

    # subselect repr
    perc = np.percentile(corrs_val, perc_threshold)
    idxs = (corrs_val > perc)
    # print('emb dim', idxs.sum(), 'val corr cutoff', perc)
    embs = embs[:, idxs]

    return embs


def get_bow_vecs(X: List[str], X_test: List[str]):
    trans = CountVectorizer().fit(X).transform
    return trans(X).todense(), trans(X_test).todense()


def get_hf_embs(X: List[str], X_test: List[str], model: str):
    pipe = pipeline("feature-extraction", model=model, device=0)

    def get_llm_embs(examples: List[str]):
        def get_emb(x):
            return {'emb': pipe(x['text'])}
        text = datasets.Dataset.from_dict({'text': examples})
        out_list = text.map(get_emb)['emb']
        # out_list is (batch_size, 1, (seq_len + 2), 768)

        # convert to np array by averaging over len (can't just convert this since seq lens vary)
        num_examples = len(out_list)
        dim_size = len(out_list[0][0][0])
        embs = np.zeros((num_examples, dim_size))
        for i in tqdm(range(num_examples)):
            embs[i] = np.mean(out_list[i], axis=1)  # avg over seq_len dim
        return embs

    logging.info('Training embs...')
    feats_train = get_llm_embs(X)
    logging.info('Testing embs...')
    feats_test = get_llm_embs(X_test)
    return feats_train, feats_test


def get_feats(model: str, X: List[str], X_test: List[str], args):
    logging.info('Extracting features for ' + model)
    mod = model.replace('fmri', '').replace('vecs', '')
    if model.endswith('fmri'):
        save_dir_fmri = join(
            results_dir, 'encoding', mod, args.subject)
        feats_train = get_embs_fmri(
            X, mod, save_dir_fmri, perc_threshold=args.perc_threshold_fmri)
        feats_test = get_embs_fmri(
            X_test, mod, save_dir_fmri, perc_threshold=args.perc_threshold_fmri)
    elif model.endswith('vecs'):
        assert mod in ['bow', 'eng1000', 'glove']
        if mod == 'bow':
            feats_train, feats_test = get_bow_vecs(X, X_test)
        elif mod in ['eng1000', 'glove']:
            feats_train = get_word_vecs(X, model=mod)
            feats_test = get_word_vecs(X_test, model=mod)
    else:  # HF checkpoint
        feats_train, feats_test = get_hf_embs(X, X_test, model=mod)
    return feats_train, feats_test


def fit_decoding(
        model, feats_train, y_train, feats_test, y_test,
        fname_save, args, frac_train_to_drop=0.1):
    """Randomly fits to only 90% of training data
    """
    np.random.seed(args.seed)
    r = defaultdict(list)

    # fit model
    logging.info('Fitting logistic...')

    if args.subsample_frac is None or args.subsample_frac < 0:
        feats_train, feats_drop, y_train, y_drop = train_test_split(
            feats_train, y_train, test_size=frac_train_to_drop, random_state=args.seed)

    # m = LogisticRegressionCV(random_state=args.seed, cv=3)
    # m = LogisticRegressionCV(random_state=args.seed, refit=True, cv=cv) # with refit, should get better performance but no variance
    # m.fit(feats_train, y_train)

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=args.seed)
    param_grid = {'C': np.logspace(-4, 4, 10)}
    logistic = LogisticRegression(random_state=args.seed)
    m = GridSearchCV(logistic, param_grid, refit=True, cv=cv)
    m.fit(feats_train, y_train)

    # save stuff
    acc = m.score(feats_test, y_test)
    logging.info(f'acc {acc:0.2f}')
    args_dict = vars(args)
    for arg in args_dict:
        r[arg].append(args_dict[arg])

    # r['dset'].append(args.dset)
    # r['feats'].append(model)
    # r['seed'].append(args.seed)
    # r['perc_threshold_fmri'].append(args.perc_threshold_fmri)
    # r['subject'].append(args.subject)
    # r['subsample_frac'].append(args.subsample_frac)
    r['acc_cv'].append(m.best_score_)
    r['acc'].append(acc)
    # r['roc_auc'].append(metrics.roc_auc_score(y_test, m.predict(feats_test)))
    r['feats_dim'].append(feats_train.shape[1])
    # r['coef_'].append(deepcopy(m))

    df = pd.DataFrame.from_dict(r).set_index('model')
    df.to_pickle(fname_save)
    fname_head_tail = os.path.split(fname_save)
    pkl.dump(m, open(join(fname_head_tail[0], 'coef_' + fname_head_tail[1]), 'wb'))
    return df


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, default='UTS03')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--perc_threshold_fmri', type=int, default=0)
    parser.add_argument("--save_dir", type=str, default='/home/chansingh/.tmp')
    parser.add_argument('--model', type=str, default='glovevecs',
                        help='Which model to extract features with. \
                                Ending in fmri uses model finetuned on fMRI. \
                                Ending in vecs uses a word-vector model. \
                                Otherwise uses HF checkpoint.')  # glovevecs, bert-10__ndel=4fmri, bert-base-uncased
    parser.add_argument('--dset', type=str, default='rotten_tomatoes',
                        choices=['trec', 'emotion', 'rotten_tomatoes', 'tweet_eval',
                                 'sst2', 'go_emotions', 'poem_sentiment', 'moral_stories'])
    # could also support more tweet dsets (currently only hate), imdb, ...
    parser.add_argument('--subsample_frac', type=float,
                        default=None, help='fraction of data to use for training. If none or negative, use all the data')
    parser.add_argument('--use_cache', type=int,
                        default=True, help='whether to use cache')
    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    # set up logging
    logger = logging.getLogger()
    logging.basicConfig(level=logging.INFO)
    # logging.basicConfig(level=logging.DEBUG)
    for k in sorted(vars(args)):
        logger.info('\t' + k + ' ' + str(vars(args)[k]))

    # check for caching
    fname_save = join(
        args.save_dir, f'{args.dset}_{args.model}_perc={args.perc_threshold_fmri}_seed={args.seed}.pkl')
    if os.path.exists(fname_save) and args.use_cache:
        logging.info('\nAlready ran ' + fname_save + '!')
        logging.info('Skipping :)!\n')
        sys.exit(0)

    # get data
    X_train, y_train, X_test, y_test = data.get_dsets(
        args.dset, seed=args.seed, subsample_frac=args.subsample_frac)

    # fit decoding
    os.makedirs(args.save_dir, exist_ok=True)
    feats_train, feats_test = get_feats(args.model, X_train, X_test, args)
    fit_decoding(args.model, feats_train, y_train,
                 feats_test, y_test, fname_save, args)
    logging.info('Succesfully completed!')
