from collections import defaultdict
from copy import deepcopy
import os
import cortex
from os.path import join

import joblib
import numpy as np
from tqdm import tqdm

from neuro.config import PROCESSED_DIR

term_dict = {
    'actions': 'Does the input mention anything related to an action?',
    'arithmetic': 'Does the input mention anything related to arithmetic?',
    'ambiguous': 'Does the input contain a sense of ambiguity?',
    'anger': 'Does the input mention anything related to anger?',
    # 'argue': 'Does the input mention anything related to arguing?',
    'calculation': 'Does the input mention anything related to calculation?',
    'color': 'Does the input mention anything related to color?',
    'conflict': 'Does the input mention anything related to conflict?',
    # 'debate': 'Does the input mention anything related to debate?',
    'disgust': 'Does the input mention anything related to disgust?',
    'empathy': 'Does the input mention anything related to empathy?',
    # 'exact': 'Does the input mention anything related to exactness?',
    'face': 'Does the input mention anything related to faces?',
    # 'fashion': 'Does the input mention anything related to fashion?',
    # 'fast': 'Does the input mention anything related to speed?',
    'fear': 'Does the input mention anything related to fear?',
    'sad': 'Does the input mention anything related to sadness?',
    'unpleasant': 'Does the input mention anything unpleasant?',
    'hands': 'Does the input mention anything related to hands?',
    'alcohol': 'Does the input mention anything related to alcohol?',
    'age': 'Does the input mention anything related to age?',
    'children': 'Does the input mention anything related to children?',
    'diseases': 'Does the input mention anything related to diseases?',
    'eyes': 'Does the input mention anything related to eyes?',
    'knowledge': 'Does the input mention anything related to knowledge?',
    'gender': 'Does the input mention anything related to gender?',
    'navigation': 'Does the input mention anything related to navigation?',
    'motor': 'Does the input mention anything related to motor movements?',
    'sounds': 'Does the input mention anything related to sounds?',
    'taste': 'Does the input mention anything related to taste?',
    'emotions': 'Does the input mention or describe highly positive emotional valence?',
    'negative-emotions': 'Does the input mention or describe highly negative emotional valence?',

    # hand-matched
    'sensation': 'Does the sentence describe a physical sensation?',
    'planning': 'Does the input involve planning or organizing?',
    'food': 'Does the input mention anything related to food?',
    'olfactory': 'Does the input mention or describe a smell?',
    'sound': 'Does the input mention or describe a sound?',
    'emotional-valence': 'Does the input mention or describe high emotional intensity?',
    'negative': 'Does the sentence contain a negation?',
    'thought': 'Does the sentence describe a personal reflection or thought?',
    'sensory': 'Does the sentence describe a sensory experience?',
    'location': 'Does the sentence mention a specific location?',
    'communication': 'Does the text describe a mode of communication?',
    'abstract': 'Is the sentence abstract rather than concrete?',
}


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


def subj_vol_to_mni_surf(
    subj_vol,
    subject='S02',
    pycortex_db_dir='/home/chansingh/mntv1/deep-fMRI/data/ds003020/derivative/pycortex-db/',
):
    '''Example usage
    subj_vol = cortex.Volume(flatmap_subject, 'UT' + subject,
                         xfmname=f"UT{subject}_auto")
    mni_vol = neurosynth.subj_vol_to_mni_surf(subj_vol, subject)
    '''
    subj_mapper = cortex.get_mapper("fsaverage", "atlas_2mm")
    fs_mapper = cortex.get_mapper(
        'UT' + subject, join(pycortex_db_dir, f'UT{subject}/transforms/UT{subject}_auto/'))
    (ltrans, rtrans) = cortex.db.get_mri_surf2surf_matrix(
        subject='UT' + subject,
        surface_type="pial",
        target_subj='fsaverage',
    )
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


def get_neurosynth_flatmaps(subject='UTS01', neurosynth_dir='/home/chansingh/mntv1/deep-fMRI/qa/neurosynth_data', mni=False):

    def _get_term_dict():
        subject_s = 'S01'  # subject.replace('UT', '')
        # neurosynth_dir = '/home/chansingh/mntv1/deep-fMRI/qa/neurosynth_data/all_association-test_z'

        # get term names
        term_names = [k.replace('.nii.gz', '').replace(
            '_association-test_z', '') for k in os.listdir(join(neurosynth_dir, f'all_in_{subject_s}-BOLD'))]

        # filter dict for files that were in neurosynth
        term_dict_ = {k: v for k, v in term_dict.items() if k in term_names}
        # for k in term_dict.keys():
        #     if k not in term_names:
        #         print(k)

        # filter dict for files that had questions run
        questions_run = [k.replace('.pkl', '') for k in os.listdir(
            '/home/chansingh/mntv1/deep-fMRI/qa/cache_gpt')]
        term_dict_ = {k: v for k, v in term_dict_.items()
                      if v in questions_run}
        return term_dict_

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

    term_dict_ = _get_term_dict()

    if mni:
        return {q: _load_flatmap_mni(
            term, neurosynth_dir) for (term, q) in term_dict_.items()}

    else:
        return {q: _load_flatmap(
            term, neurosynth_dir, subject) for (term, q) in term_dict_.items()}


term_dict_rev = {v: k for k, v in term_dict.items()}

if __name__ == '__main__':
    computed_gpt4_qs = [
        x.replace('.pkl', '') for x in os.listdir('/home/chansingh/mntv1/deep-fMRI/qa/cache_gpt')
        if '?' in x
    ]
    print('num qs', len(computed_gpt4_qs))
    computed_matched_gpt4_qs = [
        k for k in computed_gpt4_qs if k in term_dict.values()]
    print('num matched qs', len(computed_matched_gpt4_qs))

    print('UNMATCHED QS')
    for q in computed_gpt4_qs:
        if q not in term_dict.values():
            print(q)
