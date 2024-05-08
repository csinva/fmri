# Dataset set up
- to quickstart, just download the responses / wordsequences for 3 subjects from the [encoding scaling laws paper](https://utexas.app.box.com/v/EncodingModelScalingLaws/folder/230420528915)
  - this is all the data you need if you only want to analyze 3 subjects and don't want to make flatmaps
- to run eng1000, need to grab `em_data` directory from [here](https://github.com/HuthLab/deep-fMRI-dataset) and move its contents to `{root_dir}/em_data`
- for more, download data with `python experiments/00_load_dataset.py`
    - create a `data` dir under wherever you run it and will use [datalad](https://github.com/datalad/datalad) to download the preprocessed data as well as feature spaces needed for fitting [semantic encoding models](https://www.nature.com/articles/nature17637)
- to make flatmaps, need to set [pycortex filestore] to `{root_dir}/ds003020/derivative/pycortex-db/`


# Code install
- set `ridge_utils.config.root_dir/data` to where you put all the data
  - loading responses
    - `ridge_utils.data.response_utils` function `load_response`
    - loads responses from at `{root_dir}/ds003020/derivative/preprocessed_data/{subject}`, hwere they are stored in an h5 file for each story, e.g. `wheretheressmoke.h5`
  - loading stimulus
    - `ridge_utils.features.stim_utils` function `load_story_wordseqs`
    - loads textgrids from `{root_dir}/ds003020/derivative/TextGrids", where each story has a TextGrid file, e.g. `wheretheressmoke.TextGrid`
    - uses `{root_dir}/ds003020/derivative/respdict.json` to get the length of each story
- from the repo directory, start with `pip install -e .` to locally install the `huth` package
  - follow with `pip install -e ridge_utils_frozen to install `ridge_utils` from the 
- `python 01_fit_encoding.py --subject UTS03 --feature eng1000`
    - The other optional parameters that encoding.py takes such as sessions, ndelays, single_alpha allow the user to change the amount of data and regularization aspects of the linear regression used. 
    - This function will then save model performance metrics and model weights as numpy arrays. 

# Reference
- builds off https://github.com/HuthLab/deep-fMRI-dataset. See that wonderful repo!
- uses data from [openneuro](https://openneuro.org/datasets/ds003020).
- this repo copies a lot of code from[encoding-model-scaling-laws](https://github.com/HuthLab/encoding-model-scaling-laws/tree/main), which is the repo for the paper "Scaling laws for language encoding models in fMRI" ([antonello, vaidya, & huth, 2023](https://github.com/HuthLab/encoding-model-scaling-laws/tree/main?tab=readme-ov-file)). See the cool results there!
- it also copies a lot of code from the repo for [SASC](https://github.com/microsoft/automated-explanations/tree/main).