Each folder contains data for a single human subject (UTS03 is the most easily decodable, so start there). The `labels` folder contains binary labels for different questions for each ngram of the story.

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
def get_fmri_and_labs(story_name='onapproachtopluto', train_or_test='test', subject='uts03'):
    '''
    Returns
    -------
    df : pd.DataFrame
        The fMRI features, with columns corresponding to the principal components
        of the fMRI data.
    labs : pd.DataFrame
        Binary labeled annotations for each of the texts
    texts: 
        The texts corresponding to the rows of df
    '''
    df = joblib.load(f'{subject}/{train_or_test}/{story_name}.pkl')
    dfs = []
    for offset in [1, 2, 3, 4]:
        df_offset = df.shift(-offset)
        df_offset.columns = [col + f'_{offset}' for col in df.columns]
        dfs.append(df_offset)
    df = pd.concat(dfs, axis=1)  # .dropna()  # would want to dropna here

    # load labels
    labs = joblib.load(f'labels/{train_or_test}/{story_name}_labels.pkl')

    # drop rows with nans
    idxs_na = df.isna().sum(axis=1).values > 0
    df = df[~idxs_na]
    labs = labs[~idxs_na]
    texts = pd.Series(df.index)
    return df, labs, texts


def concatenate_running_texts(texts, frac=1/2):
    '''When decoding, you might want to concatenate 
    the text of the current and surrounding texts
    to deal with the temporal imprecision of the fMRI signal.
    '''
    texts_before = (
        texts.shift(1)
        .str.split().apply(  # only keep second half of words
            lambda l: ' '.join(l[int(-len(l) * frac):]) if l else '')
    )

    texts_after = (
        texts.shift(-1)
        .str.split().apply(  # only keep first half of words
            lambda l: ' '.join(l[:int(len(l) * frac)]) if l else '')
    )

    return texts_before + ' ' + texts + ' ' + texts_after

df_orig, labs, texts = get_fmri_and_labs()
texts = concatenate_running_texts(texts)
```


### Example for classification


See a full worked out example in the `example_decode.ipynb` notebook.

```python
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import cross_val_score

# load all the data for a single subject
subject = 'uts03'
data = defaultdict(list)
for train_or_test in ['test', 'train']:
    story_names_list = os.listdir(f'{subject}/{train_or_test}')
    for story_name in story_names_list:
        df, labs, texts = get_fmri_and_labs(
            story_name.replace('.pkl', ''), train_or_test, subject)
        data['df_' + train_or_test].append(df)
        data['labs_' + train_or_test].append(labs)
        data['texts_' + train_or_test].append(texts)
for k in data:
    data[k] = pd.concat(data[k], axis=0)

# example fit linear decoder
for label_num in range(data['labs_train'].shape[1]):
    X_train, y_train = data['df_train'].values, data['labs_train'].values[:, label_num]
    X_test, y_test = data['df_test'].values, data['labs_test'].values[:, label_num]

    # balance the binary class imbalance to make the problem interesting
    rus = RandomUnderSampler()
    X_train, y_train = rus.fit_resample(X_train, y_train)
    X_test, y_test = rus.fit_resample(X_test, y_test)
    
    print('label', label_num,
          data['labs_train'].columns[label_num], X_train.shape, X_test.shape)
    m = LogisticRegressionCV()
    m.fit(X_train, y_train)
    print(f"""\ttest acc {m.score(X_test, y_test):.3f}
\tnaive acc {1 -y_test.mean():.3f}""")
```

-----Sample output-------
```
label 0 Does the input contain a number? (8314, 800) (294, 800)
	test acc 0.738
	naive acc 0.500
label 1 Is time mentioned in the input? (9770, 800) (300, 800)
	test acc 0.683
	naive acc 0.500
label 2 Does the sentence include dialogue? (3890, 800) (128, 800)
	test acc 0.781
	naive acc 0.500
label 3 Does the input mention or describe high emotional intensity? (8290, 800) (212, 800)
	test acc 0.665
	naive acc 0.500
label 4 Does the sentence mention a specific location? (7172, 800) (238, 800)
	test acc 0.718
	naive acc 0.500
label 5 Is the sentence emotionally positive? (10584, 800) (332, 800)
	test acc 0.617
	naive acc 0.500
label 6 Does the sentence describe a relationship between people? (11988, 800) (292, 800)
	test acc 0.651
	naive acc 0.500
label 7 Does the input mention anything related to food? (3604, 800) (50, 800)
	test acc 0.660
	naive acc 0.500
label 8 Does the input mention or describe a sound? (4154, 800) (110, 800)
	test acc 0.745
	naive acc 0.500
```
