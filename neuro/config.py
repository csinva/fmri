from os.path import join, expanduser
# join(dirname(dirname(os.path.abspath(__file__))))

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
