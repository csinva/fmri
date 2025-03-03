import base64
from imodelsx.llm import LLM_Chat_Audio
import os.path
import numpy as np
from os.path import join
from collections import defaultdict
import pandas as pd
from tqdm import tqdm

questions = [
    # 'What is the text in the recording?',
    # 'Does the recording have a rising pitch contour?',
    'Is the recording spoken clearly without mumbling?',
    # 'Is there an echo present in the audio?',
    'Does the speaker’s voice sound breathy?',
    # 'Does the speaker have a happy tone?',
    # 'Does the the speaker sound relaxed?',
    'Does the audio contain significant pitch variation?',
    "Does the speaker's tone indicate confidence?",
    'Does the recording have a male voice?',
    'Does the audio contain background music?',
]

# wav_folder = 'segments_1.5sec'
wav_folder = 'segments_3sec'
# wav_files = [join(wav_folder, f)
#  for f in os.listdir(wav_folder) if f.endswith('.wav')]

wav_files = [
    join(wav_folder, f)
    # for i in np.arange(5, 60, 2)
    for f in sorted(os.listdir(wav_folder)) if f.endswith('.wav')
]

# sort in numeric order
wav_files = sorted(wav_files, key=lambda x: int(
    x.split('_')[-1].split('.')[0]))


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
pd.DataFrame(d, index=wav_files).T.to_csv(f'annots_podcast_audio.csv')
