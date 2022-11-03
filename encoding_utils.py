import numpy as np
import time
import pathlib
import os
import h5py
from multiprocessing.pool import ThreadPool
from os.path import join, dirname
import json
from typing import List

from ridge_utils.npp import zscore, mcorr
from ridge_utils.utils import make_delayed
from feature_spaces import repo_dir, data_dir, em_data_dir

def get_allstories(sessions=[1, 2, 3, 4, 5]) -> List[str]:
	sessions = list(map(str, sessions))
	with open(join(em_data_dir, "sess_to_story.json"), "r") as f:
		sess_to_story = json.load(f) 
	train_stories, test_stories = [], []
	for sess in sessions:
		stories, tstory = sess_to_story[sess][0], sess_to_story[sess][1]
		train_stories.extend(stories)
		if tstory not in test_stories:
			test_stories.append(tstory)
	assert len(set(train_stories) & set(test_stories)) == 0, "Train - Test overlap!"
	allstories = list(set(train_stories) | set(test_stories))
	return train_stories, test_stories, allstories

def add_delays(stories, downsampled_feat, trim, ndelays, zscore=True):
	"""Get (z-scored and delayed) stimulus for train and test stories.
	The stimulus matrix is delayed (typically by 2, 4, 6, 8 secs) to estimate the
	hemodynamic response function with a Finite Impulse Response model.

	Params
	------
	stories
		List of stimuli stories.
	downsampled_feat (dict)
		Downsampled feature vectors for all stories.
	trim
		Trim downsampled stimulus matrix.

	Returns
	-------
	delstim: <float32>[TRs, features * ndelays]
	"""
	stim = [downsampled_feat[s][5+trim:-trim] for s in stories]
	if zscore:
		stim = [zscore(s) for s in stim]
	stim = np.vstack(stim)
	delays = range(1, ndelays+1) # List of delays for Finite Impulse Response (FIR) model.
	delstim = make_delayed(stim, delays)
	return delstim

def get_response(stories, subject):
	"""Get the subject"s fMRI response for stories."""
	main_path = pathlib.Path(__file__).parent.parent.resolve()
	subject_dir = join(data_dir, "ds003020/derivative/preprocessed_data/%s" % subject)
	base = os.path.join(main_path, subject_dir)
	resp = []
	for story in stories:
		resp_path = os.path.join(base, "%s.hf5" % story)
		hf = h5py.File(resp_path, "r")
		resp.extend(hf["data"][:])
		hf.close()
	return np.array(resp)

def get_permuted_corrs(true, pred, blocklen):
	nblocks = int(true.shape[0] / blocklen)
	true = true[:blocklen*nblocks]
	block_index = np.random.choice(range(nblocks), nblocks)
	index = []
	for i in block_index:
		start, end = i*blocklen, (i+1)*blocklen
		index.extend(range(start, end))
	pred_perm = pred[index]
	nvox = true.shape[1]
	corrs = np.nan_to_num(mcorr(true, pred_perm))
	return corrs

def permutation_test(true, pred, blocklen, nperms):
	start_time = time.time()
	pool = ThreadPool(processes=10)
	perm_rsqs = pool.map(
		lambda perm: get_permuted_corrs(true, pred, blocklen), range(nperms))
	pool.close()
	end_time = time.time()
	print((end_time - start_time) / 60)
	perm_rsqs = np.array(perm_rsqs).astype(np.float32)
	real_rsqs = np.nan_to_num(mcorr(true, pred))
	pvals = (real_rsqs <= perm_rsqs).mean(0)
	return np.array(pvals), perm_rsqs, real_rsqs

def run_permutation_test(zPresp, pred, blocklen, nperms, mode='', thres=0.001):
	assert zPresp.shape == pred.shape, print(zPresp.shape, pred.shape)

	start_time = time.time()
	ntr, nvox = zPresp.shape
	partlen = nvox
	pvals, perm_rsqs, real_rsqs = [[] for _ in range(3)]

	for start in range(0, nvox, partlen):
		print(start, start+partlen)
		pv, pr, rs = permutation_test(zPresp[:, start:start+partlen], pred[:, start:start+partlen],
									  blocklen, nperms)
		pvals.append(pv)
		perm_rsqs.append(pr)
		real_rsqs.append(rs)
	pvals, perm_rsqs, real_rsqs = np.hstack(pvals), np.hstack(perm_rsqs), np.hstack(real_rsqs)

	assert pvals.shape[0] == nvox, (pvals.shape[0], nvox)
	assert perm_rsqs.shape[0] == nperms, (perm_rsqs.shape[0], nperms)
	assert perm_rsqs.shape[1] == nvox, (perm_rsqs.shape[1], nvox)
	assert real_rsqs.shape[0] == nvox, (real_rsqs.shape[0], nvox)

	cci.upload_raw_array(os.path.join(save_location, '%spvals'%mode), pvals)
	cci.upload_raw_array(os.path.join(save_location, '%sperm_rsqs'%mode), perm_rsqs)
	cci.upload_raw_array(os.path.join(save_location, '%sreal_rsqs'%mode), real_rsqs)
	print((time.time() - start_time)/60)
	
	pID, pN = fdr_correct(pvals, thres)
	cci.upload_raw_array(os.path.join(save_location, '%sgood_voxels'%mode), (pvals <= pN))
	cci.upload_raw_array(os.path.join(save_location, '%spN_thres'%mode), np.array([pN, thres], dtype=np.float32))
	return
