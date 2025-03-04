from os.path import join, expanduser, dirname
import os.path
path_to_file = os.path.dirname(os.path.abspath(__file__))
repo_dir = dirname(path_to_file)
PROCESSED_DIR = join(repo_dir, 'qa_results', 'processed')

if 'chansingh' in expanduser('~'):
    mnt_dir = '/home/chansingh/mntv1'
else:
    mnt_dir = '/mntv1'

root_dir = join(mnt_dir, 'deep-fMRI')
cache_embs_dir = join(root_dir, 'qa', 'cache_embs')
resp_processing_dir = join(root_dir, 'qa', 'resp_processing_full')
brain_drive_resps_dir = join(root_dir, 'brain_tune', 'story_data')

# eng1000 data, download from [here](https://github.com/HuthLab/deep-fMRI-dataset)
em_data_dir = join(root_dir, 'data', 'eng1000')
nlp_utils_dir = join(root_dir, 'nlp_utils')


def setup_freesurfer():
    # set os environ SUBJECTS_DIR
    FREESURFER_VARS = {
        'FREESURFER_HOME': os.path.expanduser('~/freesurfer'),
        'FSL_DIR': os.path.expanduser('~/fsl'),
        'FSFAST_HOME': os.path.expanduser('~/freesurfer/fsfast'),
        'MNI_DIR': os.path.expanduser('~/freesurfer/mni'),
        # 'SUBJECTS_DIR': join(repo_dir, 'notebooks_gt_flatmaps'),
        'SUBJECTS_DIR': os.path.expanduser('~/freesurfer/subjects'),
        # add freesurfer bin to path
        'PATH': os.path.expanduser('~/freesurfer/bin') + ':' + os.environ['PATH'],
    }
    for k in FREESURFER_VARS.keys():
        os.environ[k] = FREESURFER_VARS[k]


VOX_COUNTS = {
    'S01': 81126,
    'S02': 94251,
    'S03': 95556,
    'Mean': 90311
}
