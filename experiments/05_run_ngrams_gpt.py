import requests
from ratelimit import limits, sleep_and_retry
from collections import defaultdict
from tqdm import tqdm
import joblib
from neuro.features.stim_utils import load_story_wordseqs, load_story_wordseqs_huge
from neuro.data import story_names
from neuro.features import qa_questions, feature_spaces
import imodelsx.process_results
import pandas as pd
import os
import neuro.config
import sys
from neuro.features.questions.gpt4 import QUESTIONS_GPT4


@sleep_and_retry
@limits(calls=200, period=30)  # in secs
def call_api_limited(messages):
    resp = lm(messages, max_new_tokens=1, temperature=0,
              return_str=False, verbose=False)
    return resp


def call_api(messages):
    resp = lm(messages, max_new_tokens=1, temperature=0,
              return_str=False, verbose=False)
    return resp


if __name__ == '__main__':
    out_dir = os.path.join(neuro.config.root_dir, 'qa/cache_gpt')
    questions = QUESTIONS_GPT4[::-1]
    story_names_list = sorted(story_names.get_story_names(
        all=True))
    print('loaded', len(story_names_list), 'stories')
    wordseqs = load_story_wordseqs_huge(story_names_list)
    ngrams_list_total = []
    for story in story_names_list:
        ngrams_list = feature_spaces.get_ngrams_list_main(
            wordseqs[story], num_ngrams_context=10)
        ngrams_list_total.extend(ngrams_list)
    print(f'{len(ngrams_list_total)=} ngrams')

    prompt_template = 'Read the input then answer a question about the input.\nInput: {example}\nQuestion: {question} Answer yes or no.'
    prompt = prompt_template.format(
        example=ngrams_list[100], question='Does the sentence include a metaphor?')

    print('example prompt', repr(prompt_template.format(
        example=ngrams_list_total[100], question=questions[0])))
    lm = imodelsx.llm.get_llm('gpt-4-turbo-0125-spot', repeat_delay=1)

    answers = []
    for question in tqdm(questions):
        out_file = os.path.join(out_dir, f'{question}.pkl')
        answers = []
        print(out_file)
        if not os.path.exists(out_file):
            for i, ngrams in enumerate(tqdm(ngrams_list_total)):
                if len(ngrams.strip()) <= 3:
                    answers.append('No')
                else:
                    prompt = prompt_template.format(
                        example=ngrams, question=question)
                    messages = [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ]

                    already_cached = lm(messages, max_new_tokens=1, temperature=0,
                                        return_str=False, return_false_if_not_cached=True, verbose=False)
                    if already_cached:
                        # print('cached', i, question)
                        resp = call_api(messages)
                    else:
                        resp = call_api_limited(messages)
                        # print(resp)
                    answers.append(resp.choices[0].message.content)

                # print(i, resp.choices[0].message.content)

            answers = pd.Series(answers).str.lower()
            print(answers.value_counts())
            # assert set(answers.values) == {'yes', 'no'}
            os.makedirs(os.path.dirname(out_file), exist_ok=True)
            joblib.dump((answers.values == 'yes').astype(bool), out_file)
        else:
            answers = joblib.load(out_file)
            print('\tloaded', answers.shape, pd.Series(answers).value_counts())
