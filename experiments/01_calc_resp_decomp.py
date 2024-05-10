from sklearn.decomposition import PCA, NMF, FastICA, DictionaryLearning, IncrementalPCA
import numpy as np
import pickle as pkl
from neuro.data import response_utils
import traceback
from os.path import join
import os
import neuro.data.story_names as story_names
import joblib
import neuro.config
import fire
from tqdm import tqdm
path_to_file = os.path.dirname(os.path.abspath(__file__))


def calc_decomp(out_dir, subject, subsample_input=None, run_mini_test=False):
    print('loading responses...')
    train_stories = story_names.get_story_names(
        subject, 'train', use_huge=True)
    if run_mini_test:
        resp_train = response_utils.load_response(
            ['sloth', 'life', 'adollshouse', 'wheretheressmoke', 'fromboyhoodtofatherhood'], subject)
    else:
        if subject in ['UTS01', 'UTS02', 'UTS03']:
            resp_train = response_utils.load_response_huge(
                train_stories, subject)  # shape (27449, 95556)
        else:
            resp_train = response_utils.load_response(
                train_stories, subject)  # shape (27449, 95556)
    print('num nans', np.sum(np.isnan(resp_train)))
    # fill nan with mean
    resp_train[np.isnan(resp_train)] = np.nanmean(resp_train)

    print('loaded shape', resp_train.shape)
    if subsample_input:
        resp_train = resp_train[::subsample_input]
        print('shape after subsampling', resp_train.shape)

    print('calculating mean/std...')
    means = np.mean(resp_train, axis=0)
    stds = np.std(resp_train, axis=0)

    os.makedirs(out_dir, exist_ok=True)
    out_file = join(out_dir, 'resps_means_stds.pkl')
    pkl.dump({'means': means, 'stds': stds}, open(out_file, 'wb'))

    print('fitting PCA...')
    out_file = join(out_dir, 'resps_pca.pkl')
    # if not os.path.exists(out_file):
    # pca = PCA().fit(resp_train)
    # pca = IncrementalPCA(n_components=500, batch_size=resp_train.shape[1]).fit(resp_train)
    # pca = IncrementalPCA(n_components=300).fit(resp_train)
    pca = PCA(n_components=200, svd_solver='randomized',
              random_state=42).fit(resp_train)
    joblib.dump(pca, out_file)

    # print('fitting ICA...')
    # out_file = join(out_dir, 'resps_ica.pkl')
    # if not os.path.exists(out_file):
    #     ica = FastICA().fit(resp_train)
    #     pkl.dump({'ica': ica}, open(out_file, 'wb'))

    # print('fitting NMF...')
    # try:
    #     out_file = join(out_dir, 'resps_nmf.pkl')
    #     if not os.path.exists(out_file):
    #         nmf = NMF(n_components=1000).fit(resp_train - resp_train.min())
    #         pkl.dump({'nmf': nmf}, open(out_file, 'wb'))
    # except:
    #     print('failed nmf!')

    # print('fitting SC...')
    # try:
    #     out_file = join(out_dir, 'resps_sc.pkl')
    #     if not os.path.exists(out_file):
    #         sc = DictionaryLearning(n_components=1000).fit(
    #             resp_train - resp_train.min())
    #         pkl.dump({'sc': sc}, open(out_file, 'wb'))
    # except:
    #     print('failed sc!')


def save_mini_pca(out_dir, pc_components=100):
    pca = joblib.load(join(out_dir, 'resps_pca.pkl'))
    pca.components_ = pca.components_[
        : pc_components]
    joblib.dump(pca, join(out_dir, f'resps_pca_{pc_components}.pkl'))

# def viz_decomp(out_dir):
#     # decomp_dir = join(path_to_file, 'decomps')
#     # os.makedirs(decomp_dir, exist_ok=True)
#     sys.path.append(join(path_to_file, '..'))
#     # viz_cortex = __import__('03_viz_cortex')
#     for k in ['pca', 'nmf', 'ica']:  # , 'sc']:
#         print('visualizing', k)
#         decomp = pkl.load(open(join(out_dir, f'resps_{k}.pkl'), 'rb'))
#         for i in tqdm(range(10)):
#             # (n_components, n_features)
#             viz_cortex.quickshow(decomp[k].components_[i])
#             plt.savefig(join(out_dir, f'{k}_component_{i}.pdf'))
#             plt.savefig(join(out_dir, f'{k}_component_{i}.png'))
#             plt.close()


def main(subject):
    run_mini_test = False
    print(subject)
    out_dir = join(neuro.config.resp_processing_dir, subject)
    os.makedirs(out_dir, exist_ok=True)
    try:
        calc_decomp(out_dir, subject, subsample_input=None,
                    run_mini_test=run_mini_test)

        save_mini_pca(out_dir, pc_components=100)
    except:
        print('failed', subject)
        # full traceback
        traceback.print_exc()
    # viz_decomp(out_dir)


if __name__ == '__main__':
    # subjects = [f'UTS0{k}' for k in range(1, 9)]
    # for subject in tqdm(subjects):
    #     main(subject)
    fire.Fire(main)
