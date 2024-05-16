import cortex
from collections import defaultdict
import neuro.config
from copy import deepcopy
import pandas as pd
from os.path import join
from tqdm import tqdm


# def save_voxel_roi_dfs():
dfs = []
for subject in ['UTS02', 'UTS03', 'UTS01']:
    # Get the map of which voxels are inside of our ROI
    index_volume, index_keys = cortex.utils.get_roi_masks(
        subject, f'{subject}_auto',
        # Default (None) gives all available ROIs in overlays.svg
        roi_list=None,
        gm_sampler='cortical-conservative',  # Select only voxels mostly within cortex
        # Separate left/right ROIs (this occurs anyway with index volumes)
        split_lr=True,
        threshold=0.9,  # convert probability values to boolean mask for each ROI
        return_dict=False  # return index volume, not dict of masks
    )

    # Count how many of the top_voxels are in each ROI
    roi_to_voxels_dict = {}
    for k in tqdm(index_keys):
        roi_verts = cortex.get_roi_verts(subject, k)
        roi_to_voxels_dict[k] = roi_verts[k]

    # convert dict of lists to dframe (not 1-1)
    voxel_roi_df = pd.DataFrame([
        (v, k) for k, v_list in roi_to_voxels_dict.items() for v in v_list
    ], columns=['voxel_num', 'roi'])
    voxel_roi_df['subject'] = subject
    dfs.append(voxel_roi_df)
save_dir = join(neuro.config.root_dir, 'qa', 'roi_cache')
os.makedirs(save_dir, exist_ok=True)
# groupby concenate roi values into list
rois_df = pd.concat(dfs)
rois_df = rois_df.groupby(['voxel_num', 'subject'])[
    'roi'].apply(list).reset_index()
rois_df.to_pickle(join(save_dir, f'voxel_roi_df.pkl'))
