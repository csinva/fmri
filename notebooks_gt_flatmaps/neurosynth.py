from collections import defaultdict
from copy import deepcopy
import os
import cortex
from os.path import join

import joblib
import numpy as np
from tqdm import tqdm

from neuro.config import PROCESSED_DIR

'''
term_dict_rev = [
    # ('Does the input mention anything related to arithmetic?', 'arithmetic'),
    # ('Does the input mention anything related to anger?', 'anger'),
    # ('Does the input mention anything related to disgust?', 'disgust'),
    # ('Does the input mention anything related to empathy?', 'empathy'),
    # ('Does the input mention anything related to age?', 'age'),
    # ('Does the input mention anything related to eyes?', 'eyes'),
    # ('Does the input mention anything related to knowledge?', 'knowledge'),
    # ('Does the input mention anything related to gender?', 'gender'),

    # ('Does the input mention anything related to navigation?', 'navigation'),
    # ('Does the input mention anything related to motor movements?', 'motor'),
    # ('Does the input mention anything related to taste?', 'taste'),
    # ('Does the input mention anything related to food?', 'food'),
    # ('Does the input mention anything related to alcohol?', 'alcohol'),

    # senses
    ('Does the input mention or describe a sound?', 'sound'),
    ('Does the input mention or describe a smell?', 'olfactory'),
    ('Does the input mention or describe a taste?', 'taste'),
    ('Does the input mention or describe a texture?', 'tactile'),
    ('Does the input mention or describe a visual experience?', 'visual-stimuli'),
]
'''

term_dict_rev = [
    ('Does the input contain a measurement?', 'arithmetic'),
    # ('Does the input contain a measurement?', 'size'),
    # ('Does the input contain a measurement?', 'sizes'),
    ('Does the input contain a number?', 'arithmetic'),
    ('Does the input describe a specific texture or sensation?', 'sensation'),
    # ('Does the input describe a specific texture or sensation?', 'sensations'),
    # ('Does the input describe a specific texture or sensation?', 'tactile'),
    ('Does the input involve planning or organizing?', 'planning'),
    ('Does the sentence contain a negation?', 'negative'),
    # ('Does the sentence contain a negation?', 'negative-positive'),
    # ('Does the sentence contain a negation?', 'negativity'),
    # ('Does the sentence contain a proper noun?', 'names'),
    # ('Does the sentence contain a proper noun?', 'naming'),
    ('Does the sentence contain a proper noun?', 'nouns'),
    ('Does the sentence describe a personal or social interaction that leads to a change or revelation?',
     'social-interaction'),
    # ('Does the sentence describe a personal or social interaction that leads to a change or revelation?',
    #  'social-interactions'),
    # ('Does the sentence describe a personal reflection or thought?', 'thinking'),
    # ('Does the sentence describe a personal reflection or thought?', 'thought'),
    ('Does the sentence describe a personal reflection or thought?', 'thoughts'),
    ('Does the sentence describe a physical action?', 'action'),
    # ('Does the sentence describe a physical action?', 'action-observation'),
    # ('Does the sentence describe a physical action?', 'actions'),
    # ('Does the sentence describe a physical sensation?', 'sensation'),
    # ('Does the sentence describe a physical sensation?', 'sensations'),
    # ('Does the sentence describe a physical sensation?', 'tactile'),
    ('Does the sentence describe a physical sensation?', 'touch'),
    ('Does the sentence describe a relationship between people?', 'social'),
    # ('Does the sentence describe a relationship between people?',
    #  'social-interaction'),
    # ('Does the sentence describe a relationship between people?',
    #  'social-interactions'),
    # ('Does the sentence describe a sensory experience?', 'sensation'),
    # ('Does the sentence describe a sensory experience?', 'sensations'),
    # ('Does the sentence describe a sensory experience?', 'sensory'),
    # ('Does the sentence describe a sensory experience?',
    #  'sensory-information'),
    # ('Does the sentence describe a specific sensation or feeling?', 'feeling'),
    # ('Does the sentence describe a specific sensation or feeling?', 'feelings'),
    ('Does the sentence describe a specific sensation or feeling?', 'sensation'),
    # ('Does the sentence describe a specific sensation or feeling?', 'sensations'),
    ('Does the sentence describe a visual experience or scene?',
     'visual-information'),
    # ('Does the sentence describe a visual experience or scene?',
    #  'visual-perception'),
    # ("Does the sentence express the narrator's opinion or judgment about an event or character?",
    #  'judgment'),
    # ("Does the sentence express the narrator's opinion or judgment about an event or character?",
    #  'judgment-task'),
    ("Does the sentence express the narrator's opinion or judgment about an event or character?",
     'judgments'),
    ('Does the sentence include a direct speech quotation?', 'communication'),
    # ('Does the sentence include a direct speech quotation?', 'speakers'),
    # ('Does the sentence include a direct speech quotation?', 'speaking'),
    ('Does the sentence include a personal anecdote or story?', 'personal'),
    ('Does the sentence include dialogue?', 'communication'),
    # ('Does the sentence include dialogue?', 'speakers'),
    # ('Does the sentence include dialogue?', 'speaking'),
    ('Does the sentence involve a description of physical environment or setting?',
     'location'),
    # ('Does the sentence involve a description of physical environment or setting?',
    #  'locations'),
    # ('Does the sentence involve a description of physical environment or setting?',
    #  'visual-perception'),
    # ('Does the sentence involve a discussion about personal or social values?',
    #  'belief'),
    # ('Does the sentence involve a discussion about personal or social values?',
    #  'beliefs'),
    # ('Does the sentence involve a discussion about personal or social values?',
    #  'personal'),
    ('Does the sentence involve a discussion about personal or social values?',
     'social'),
    # ('Does the sentence involve a discussion about personal or social values?',
    #  'value'),
    # ('Does the sentence involve a discussion about personal or social values?',
    #  'values'),
    # ('Does the sentence involve an expression of personal values or beliefs?',
    #  'belief'),
    # ('Does the sentence involve an expression of personal values or beliefs?',
    #  'beliefs'),
    # ('Does the sentence involve an expression of personal values or beliefs?',
    #  'value'),
    # ('Does the sentence involve an expression of personal values or beliefs?',
    #  'values'),
    ('Does the sentence involve spatial reasoning?', 'spatial'),
    # ('Does the sentence involve spatial reasoning?', 'spatial-information'),
    # ('Does the sentence involve spatial reasoning?', 'spatially'),
    # ('Does the sentence involve spatial reasoning?', 'visuo-spatial'),
    # ('Does the sentence involve spatial reasoning?', 'visuospatial'),
    # ('Does the sentence involve the mention of a specific object or item?',
    #  'item'),
    # ('Does the sentence involve the mention of a specific object or item?',
    #  'items'),
    ('Does the sentence involve the mention of a specific object or item?',
     'object'),
    # ('Does the sentence involve the mention of a specific object or item?',
    #  'object-recognition'),
    # ('Does the sentence involve the mention of a specific object or item?',
    #  'objects'),
    ('Does the sentence mention a specific location?', 'location'),
    # ('Does the sentence mention a specific location?', 'locations'),
    # ('Does the sentence mention a specific location?', 'place'),
    ('Does the text describe a mode of communication?', 'communication'),
    ('Does the text include a planning or decision-making process?', 'planning'),
    ('Is the sentence abstract rather than concrete?', 'abstract'),
    # ('Is the sentence abstract rather than concrete?', 'conceptual'),
    # ('Is the sentence abstract rather than concrete?', 'concrete'),
    # ('Is the sentence reflective, involving self-analysis or introspection?',
    #  'thinking'),
    # ('Is the sentence reflective, involving self-analysis or introspection?',
    #  'thought'),
    ('Is the sentence reflective, involving self-analysis or introspection?',
     'thoughts'),
    ('Is time mentioned in the input?', 'time-task'),
    # ('Is time mentioned in the input?', 'timing'),
    # ('Does the sentence contain a cultural reference?', 'reference'),
    # ('Is the input related to a specific industry or profession?', ''),
    # ('Does the sentence include technical or specialized terminology?', ''),
    # ('Does the input include a comparison or metaphor?', ''),
    # ('Does the sentence express a sense of belonging or connection to a place or community?', ''),
    # ('Does the text describe a journey?', ''),
]


# term_dict = {v: k for k, v in term_dict_rev.items()}
term_dict = [(v, k) for (k, v) in term_dict_rev]


def load_flatmaps_qa_dicts_by_subject(subjects, settings):
    flatmaps_qa_dicts_by_subject = {}
    for subject in tqdm(subjects):
        # for subject in tqdm([f'UTS0{i}' for i in range(1, 9)]):

        flatmaps_qa_dict_over_settings = defaultdict(list)
        for setting in settings:
            flatmaps_qa_dict = joblib.load(
                join(PROCESSED_DIR, subject.replace('UT', ''), setting + '.pkl'))
            for q in flatmaps_qa_dict.keys():
                flatmaps_qa_dict_over_settings[q].append(flatmaps_qa_dict[q])
        flatmaps_qa_dict = {
            q: np.mean(flatmaps_qa_dict_over_settings[q], axis=0)
            for q in flatmaps_qa_dict_over_settings.keys()
        }
        flatmaps_qa_dicts_by_subject[subject] = deepcopy(flatmaps_qa_dict)
    return flatmaps_qa_dicts_by_subject


def subj_vol_to_mni_surf_setup(subject='S02', pycortex_db_dir='/home/chansingh/mntv1/deep-fMRI/data/ds003020/derivative/pycortex-db/'):
    subj_mapper = cortex.get_mapper("fsaverage", "atlas_2mm")
    fs_mapper = cortex.get_mapper(
        'UT' + subject, join(pycortex_db_dir, f'UT{subject}/transforms/UT{subject}_auto/'))
    (ltrans, rtrans) = cortex.db.get_mri_surf2surf_matrix(
        subject='UT' + subject,
        surface_type="pial",
        target_subj='fsaverage',
    )
    return subj_mapper, fs_mapper, (ltrans, rtrans)


def subj_vol_to_mni_surf(
    subj_vol,
    subject='S02',
    pycortex_db_dir='/home/chansingh/mntv1/deep-fMRI/data/ds003020/derivative/pycortex-db/',
    cached_tuple=None,
):
    '''Example usage
    subj_vol = cortex.Volume(flatmap_subject, 'UT' + subject,
                         xfmname=f"UT{subject}_auto")
    mni_vol = neurosynth.subj_vol_to_mni_surf(subj_vol, subject)
    '''
    if cached_tuple is None:
        subj_mapper, fs_mapper, (ltrans, rtrans) = subj_vol_to_mni_surf_setup(
            subject, pycortex_db_dir)
    else:
        subj_mapper, fs_mapper, (ltrans, rtrans) = cached_tuple

    fs_surf = fs_mapper(subj_vol)
    surf = cortex.Vertex(
        np.hstack([ltrans@fs_surf.left, rtrans@fs_surf.right]), 'fsaverage')
    mni_vol = subj_mapper.backwards(surf)
    return mni_vol


def mni_vol_to_subj_vol_surf(
    mni_vol,
    subject='S02',
    pycortex_db_dir='/home/chansingh/mntv1/deep-fMRI/data/ds003020/derivative/pycortex-db/',
):
    '''Example usage
    # mni_vol = cortex.Volume(mni_arr, "fsaverage", "atlas_2mm")
    term = 'location'
    mni_filename = f'/home/chansingh/mntv1/deep-fMRI/qa/neurosynth_data/all_association-test_z/{term}_association-test_z.nii.gz'
    mni_vol = cortex.Volume(mni_filename, "fsaverage", "atlas_2mm")
    subj_vol, subj_arr = neurosynth.mni_vol_to_subj_vol_surf(
        mni_vol, subject=subject)
    '''
    fs_mapper = cortex.get_mapper("fsaverage", "atlas_2mm")
    subj_mapper = cortex.get_mapper(
        'UT' + subject, join(pycortex_db_dir, f'UT{subject}/transforms/UT{subject}_auto/'))

    (ltrans, rtrans) = cortex.db.get_mri_surf2surf_matrix(
        subject="fsaverage",
        surface_type="pial",
        target_subj='UT' + subject,
    )
    fs_surf = fs_mapper(mni_vol)
    subj_surf = cortex.Vertex(
        np.hstack([ltrans@fs_surf.left, rtrans@fs_surf.right]), 'UT' + subject)
    subj_vol = subj_mapper.backwards(subj_surf)
    mask = cortex.db.get_mask('UT' + subject, 'UT' + subject + '_auto')
    return subj_vol, subj_vol.data[mask]


def subj_vol_to_subj_surf(
    subj_vol,
    subject_input='S02',
    subject_target='S03',
    pycortex_db_dir='/home/chansingh/mntv1/deep-fMRI/data/ds003020/derivative/pycortex-db/',
):
    '''Example usage
    subject_input = 'S02'
    subject_target = 'S03'
    subj_vol = cortex.Volume(subject_arr, subject='UT' + subject_input,
                            xfmname=f"UT{subject_input}_auto")
    target_arr = neurosynth.subj_vol_to_subj_surf(
        subj_vol, subject_input, subject_target)
    '''

    # subj_mapper = cortex.get_mapper("fsaverage", "atlas_2mm")
    subj_mapper = cortex.get_mapper(
        'UT' + subject_target, join(pycortex_db_dir, f'UT{subject_target}/transforms/UT{subject_target}_auto/'))
    fs_mapper = cortex.get_mapper(
        'UT' + subject_input, join(pycortex_db_dir, f'UT{subject_input}/transforms/UT{subject_input}_auto/'))
    (ltrans, rtrans) = cortex.db.get_mri_surf2surf_matrix(
        subject='UT' + subject_input,
        surface_type="pial",
        target_subj='UT' + subject_target,
    )

    fs_surf = fs_mapper(subj_vol)
    surf = cortex.Vertex(
        np.hstack([ltrans@fs_surf.left, rtrans@fs_surf.right]), 'UT' + subject_target)
    subject_target_vol = subj_mapper.backwards(surf)
    mask_target = cortex.db.get_mask(
        'UT' + subject_target, 'UT' + subject_target + '_auto')
    return subject_target_vol.data[mask_target]


def get_neurosynth_flatmaps(subject='UTS01', neurosynth_dir='/home/chansingh/mntv1/deep-fMRI/qa/neurosynth_data', mni=False):
    '''
    def _get_term_dict():
        subject_s = 'S01'  # subject.replace('UT', '')
        # neurosynth_dir = '/home/chansingh/mntv1/deep-fMRI/qa/neurosynth_data/all_association-test_z'

        # get term names
        term_names = [k.replace('.nii.gz', '').replace(
            '_association-test_z', '') for k in os.listdir(join(neurosynth_dir, f'all_in_{subject_s}-BOLD'))]

        # filter dict for files that were in neurosynth
        # term_dict_ = {k: v for k, v in term_dict.items() if k in term_names}
        for k in term_dict.keys():
            assert k in term_names, f'{k} not in term names!'
        term_dict_ = term_dict
        # if k not in term_names:
        #         print(k)

        # filter dict for files that had questions run
        questions_run = [k.replace('.pkl', '') for k in os.listdir(
            '/home/chansingh/mntv1/deep-fMRI/qa/cache_gpt')]
        for k in term_dict_rev:
            assert k in questions_run, f'{k} not in questions run!'
        # term_dict_ = {k: v for k, v in term_dict_.items()
            #   if v in questions_run}
        term_dict_ = term_dict_
        return term_dict_
    '''

    def _load_flatmap_mni(term, neurosynth_dir):
        # import nibabel as nib
        output_file = join(
            neurosynth_dir, f'all_association-test_z/{term}_association-test_z.nii.gz')
        mni_array = cortex.Volume(output_file, "fsaverage", "atlas_2mm").data
        # nii_file = nib.load(output_file)
        # mni_array = nii_file.get_fdata()
        return mni_array

    def _load_flatmap(term, neurosynth_dir, subject):
        import nibabel as nib
        subject_s = subject.replace('UT', '')
        # output_file = join(neurosynth_dir, f'{term}_association-test_z.nii.gz')
        output_file = join(
            neurosynth_dir, f'all_in_{subject_s}-BOLD/{term}.nii.gz')
        vol = cortex.Volume(output_file, subject, subject + '_auto').data
        mask = cortex.db.get_mask(subject, subject + '_auto')
        return vol[mask]

    # term_dict_ = _get_term_dict()

    if mni:
        # return {q: _load_flatmap_mni(
        # term, neurosynth_dir) for (term, q) in term_dict_.items()}
        # return {q: _load_flatmap_mni(
        #     # term, neurosynth_dir) for (q, term) in term_dict_rev.items()}
        # return {(q, term): _load_flatmap(
        # term, neurosynth_dir, subject) for (q, term) in term_dict_rev.items()}
        return {(q, term): _load_flatmap(
            term, neurosynth_dir, subject) for (q, term) in term_dict_rev}

    else:
        # return {q: _load_flatmap(
        # term, neurosynth_dir, subject) for (term, q) in term_dict_.items()}
        # return {q: _load_flatmap(
        # term, neurosynth_dir, subject) for (q, term) in term_dict_rev.items()}
        # return {(q, term): _load_flatmap(
        # term, neurosynth_dir, subject) for (q, term) in term_dict_rev.items()}
        return {(q, term): _load_flatmap(
            term, neurosynth_dir, subject) for (q, term) in term_dict_rev}


if __name__ == '__main__':
    # computed_gpt4_qs = [
    #     x.replace('.pkl', '') for x in os.listdir('/home/chansingh/mntv1/deep-fMRI/qa/cache_gpt')
    #     if '?' in x
    # ]
    # print('num qs', len(computed_gpt4_qs))
    # computed_matched_gpt4_qs = [
    #     k for k in computed_gpt4_qs if k in term_dict.values()]
    # print('num matched qs', len(computed_matched_gpt4_qs))

    # print('UNMATCHED QS')
    # for q in computed_gpt4_qs:
    #     if q not in term_dict.values():
    #         print(q)

    s = [k.split('.')[0] for k in os.listdir(
        '/home/chansingh/mntv1/deep-fMRI/qa/neurosynth_data/all_in_S01-BOLD')]
    for k in term_dict.keys():
        assert k in s, k
