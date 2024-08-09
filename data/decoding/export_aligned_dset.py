from ridge_utils.DataSequence import DataSequence
import pandas as pd
from os.path import dirname
import os
from tqdm import tqdm
from neuro.features import qa_questions, feature_spaces
from neuro.data import story_names, response_utils
from neuro.features.stim_utils import load_story_wordseqs, load_story_wordseqs_huge
import neuro.config
import joblib
from os.path import join

story_names_list = sorted(story_names.get_story_names(all=True))
wordseqs = load_story_wordseqs_huge(story_names_list)


class A:
    subject = 'UTS03'
    use_huge = True


args = A()

# def _get_largest_absolute_coefs(_pca, n_pcs=50, n_coefs_per_pc=50):
#     idxs_large = set()
#     for i in range(n_pcs):
#         coefs = np.abs(_pca.components_[i])
#         idxs = np.argsort(coefs)[::-1][:n_coefs_per_pc]
#         idxs_large.update(idxs)
#     idxs_large = np.array(list(idxs_large))
#     return idxs_large

for train_or_test in ['test', 'train']:
    for subject in ['UTS03', 'UTS02', 'UTS01']:
        story_names_list = story_names.get_story_names(
            subject=subject, train_or_test=train_or_test, use_huge=True)
        args.subject = subject
        for story_name in tqdm(story_names_list):
            out_file = f'{subject.lower()}/{train_or_test}/{story_name}.pkl'
            if os.path.exists(out_file):
                print('skipping', out_file)
                continue

            ngrams_list = feature_spaces.get_ngrams_list_main(
                wordseqs[story_name], num_trs_context=1)
            ngrams_list = ngrams_list[10:-5]  # apply trim
            args.pc_components = 10000
            _, resp_test, _pca, _scaler_train, _scaler_test = response_utils.get_resps_full(
                args, args.subject, [story_name], [story_name])

            # args.pc_components = -1
            # _, resp_test_full = response_utils.get_resps_full(
            # args, args.subject, [story_name], [story_name])

            # idxs_large = _get_largest_absolute_coefs(_pca)
            # resp_selected = np.hstack((resp_test, resp_test_full[:, idxs_large]))
            resp_selected = resp_test

            # print(story_name, 'shapes', resp_test.shape,
            #   resp_test_full.shape, resp_selected.shape)

            # temporal alignment
            # offset = 2
            # resp_selected = resp_selected[offset:, :]
            # ngrams_list = ngrams_list[:-offset]

            # apply convolution smoothing filter over axis 0 of resp
            # plt.plot(resp_selected[:, 0])
            # conv_filter = np.array([1/3, 1, 1/3])/(5/3)
            # resp_selected = np.apply_along_axis(
            # lambda m: np.convolve(m, conv_filter, mode='same'), axis=0, arr=resp_selected)
            # plt.plot(resp_selected[:, 0])

            # trim by 1
            # resp_selected = resp_selected[1:-1, :]
            # ngrams_list = ngrams_list[1:-1]

            assert resp_selected.shape[0] == len(
                ngrams_list), f'{resp_selected.shape[0]} != {len(ngrams_list)}'

            column_names = ['PC' + str(i) for i in range(resp_test.shape[1])]
            # + ['Vox' + str(i) for i in idxs_large]
            df = pd.DataFrame(
                resp_selected, columns=column_names, index=ngrams_list)

            # add answer to questions
            questions = [
                'Does the input contain a number?',
                'Is time mentioned in the input?',
                'Does the sentence include dialogue?',
                'Does the input mention or describe high emotional intensity?',
                'Does the sentence mention a specific location?',
                'Is the sentence emotionally positive?',
                'Does the sentence describe a relationship between people?',
            ]
            question_answers = neuro.features.feature_spaces.get_gpt4_qa_embs_cached(
                story_name=story_name, questions=questions,
                return_ngrams=False)
            question_answers = neuro.features.feature_spaces.downsample_word_vectors(
                [story_name], {story_name: question_answers}, wordseqs)[story_name][10:-5]
            df_answers = pd.DataFrame(question_answers, columns=questions)
            for c in df_answers.columns:
                df['ANS___' + c] = df_answers[c].values

            print('saving shape', df.shape)
            os.makedirs(dirname(out_file), exist_ok=True)
            df.to_pickle(out_file)
            # joblib.dump(resp_selected, f'{subject.lower()}/{story_name}_resp.pkl')
            # joblib.dump(
            # ngrams_list, f'{subject.lower()}/{story_name}_row_names_ngrams.pkl')
            # joblib.dump(
            # column_names, f'{subject.lower()}/{story_name}_column_names_fmri.pkl')
