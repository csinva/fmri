import base64
from imodelsx.llm import LLM_Chat_Audio
import os.path
import numpy as np
from os.path import join
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
from neuro.ecog.config import ECOG_DIR
bids_root = join(ECOG_DIR, 'podcasts_data', 'ds005574/')

questions = [
    # 'What is the text in the recording?',
    # 'Does the recording have a rising pitch contour?',
    # 'Is there an echo present in the audio?',
    # 'Does the speaker have a happy tone?',
    # 'Does the the speaker sound relaxed?',
    'Is the speaker asking a question?',
    'Does the speaker sound happy?',
    'Is the recording spoken clearly without mumbling?',
    'Does the speaker’s voice sound breathy?',
    'Does the audio contain significant pitch variation?',
    "Does the speaker's tone indicate confidence?",
    'Does the recording have a male voice?',
    'Does the audio contain background music?',
]

# wav_folder = 'segments_1.5sec'
story_name = '___podcasts-story___'
setting = 'sec_3'  # ['words', 'sec_6', 'sec_3', 'sec_1.5']'segments_3sec'
checkpoint = "gpt-4o-audio-preview"
wav_folder = join('segments', setting)
wav_files = [join(wav_folder, f)
             for f in os.listdir(wav_folder) if f.endswith('.wav')]

# sort in numeric order
wav_files = sorted(wav_files, key=lambda x: int(
    x.split('segment_')[-1].split('.')[0]))


lm = LLM_Chat_Audio(
    checkpoint="gpt-4o-audio-preview",
    # checkpoint="gpt-4o-mini-audio-preview",
    CACHE_DIR=os.path.expanduser('~/.cache_audio'))


d = defaultdict(list)
for wav_file in tqdm(wav_files):
    with open(wav_file, "rb") as wav_file:
        wav_data = wav_file.read()
    encoded_string = base64.b64encode(wav_data).decode('utf-8')

    for question in questions:
        d[question].append(
            lm(
                prompt_str=f"{question} Answer yes or no. Don't say anything else.",
                # prompt_str=f"Transcribe the text in the recording.",
                audio_str=encoded_string,
            )
        )
        # print(d)


out_dir = join(ECOG_DIR, 'features_audio', checkpoint, setting)
os.makedirs(out_dir, exist_ok=True)


# merge with word df
df_word = pd.read_csv(join(ECOG_DIR, 'podcasts_data',
                      'df_word_with_wav_timings.csv'))

df_out = pd.DataFrame(
    d,
    index=[x.split('/')[-1].replace('.wav', '')
           for x in wav_files]
)
for c in df_out:
    df_word[c] = df_out[c].values

print('saved to', out_dir)
df_word.to_csv(join(out_dir, f'{story_name}.csv'))
df_word.to_pickle(join(out_dir, f'{story_name}.pkl'))
