Each folder contains data for a single human subject (UTS03 is the most easily decodable, so start there). The `labels` folder contains annotated labels for each part of the story.

- The embeddings are different for each subject, so always train/test on each subject individually

Within each folder, each pickle file contains a dataframe that can be loaded using the code below. Each dataframe contains data for a single story that a subject listened to. Stories are split into the `train` and `test` sets, where the data quality in test stories is much higher because it is averaged over repeated presentations.

- Each row corresponds to a single example.
- The dataframe index is the string to be decoded.
- The values of the dataframe give different parts of the fMRI recorded signal (our "embedding"). The first 200 columns give the principal components of the signal (they are the most important).
- Importantly, the rows are in temporal order, so if you join the strings you get back the story that a subject was listening to. This temporal info can significantly help smooth out the decoded story.
  - Note that the embeddings in fMRI are not temporally precise, so the embedding at any timepoint contains some decodable info about the next/previous timesteps as well.
  - For few-shot decoding, things will probably work better if you use few-shot examples from the beginning of the story and then test on the end of the story.

When loading a dataframe, you should use the following function to concatenate the fMRI embeddings from multiple timepoints (this is helpful since the relevant infor for decoding is spread out over time). This function will concatenate the embeddings from the specified offsets (1, 2, 3, and 4 offsets is a good setting to use).

```python
story_name = 'onapproachtopluto'
df = joblib.load(f'uts03/test/{story_name}.pkl')
dfs = []
for offset in [1, 2, 3, 4]:
    df_offset = df.shift(-offset)
    df_offset.columns = [col + f'_{offset}' for col in df.columns]
    dfs.append(df_offset)
df = pd.concat(dfs, axis=1)  # .dropna()  # would want to dropna here

# load labels
labs = joblib.load(f'labels/test/{story_name}_labels.pkl')

# drop rows with nans
idxs_na = df.isna().sum(axis=1).values > 0
df = df[~idxs_na]
labs = labs[~idxs_na]
```