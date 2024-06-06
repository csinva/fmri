Each folder contains data for a single human subject (UTS03 is the most easily decodable, so start there).

- The embeddings are different for each subject, so always train/test on each subject individually

Within each folder, each pickle file contains a dataframe that can be loaded as  `df = joblib.load('filename.pkl')`. This contains data for a single story that a subject listened to.

- Each row corresponds to a single example.
- The dataframe index is the string to be decoded.
- The values of the dataframe give different parts of the fMRI recorded signal (our "embedding"). The first 200 columns give the principal components of the signal (they are the most important). The remaining columns give values for individual voxels. These can be discarded as necessary if we want to reduce the dimension of the embedding.
- Importantly, the rows are in temporal order, so if you join the strings you get back the story that a subject was listening to. This temporal info can significantly help smooth out the decoded story.
  - Note that the embeddings in fMRI are not temporally precise, so the embedding at any timepoint contains some decodable info about the next/previous timesteps as well.
  - For few-shot decoding, things will probably work better if you use few-shot examples from the beginning of the story and then test on the end of the story.