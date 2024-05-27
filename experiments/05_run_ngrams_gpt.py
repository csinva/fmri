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
import sys


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

    questions = [
        'Does the sentence describe a personal reflection or thought?',
        'Does the sentence contain a proper noun?',
        'Does the sentence describe a physical action?',
        'Does the sentence describe a personal or social interaction that leads to a change or revelation?',
        #  'Does the sentence involve the mention of a specific object or item?', # completed
        'Does the sentence involve a description of physical environment or setting?',
        'Does the sentence describe a relationship between people?',
        'Does the sentence mention a specific location?',
        #  'Is time mentioned in the input?', # completed
        'Is the sentence abstract rather than concrete?',
        "Does the sentence express the narrator's opinion or judgment about an event or character?",
        'Is the input related to a specific industry or profession?',
        'Does the sentence include dialogue?',
        'Does the sentence describe a visual experience or scene?',
        'Does the input involve planning or organizing?',
        'Does the sentence involve spatial reasoning?',
        'Does the sentence involve an expression of personal values or beliefs?',
        'Does the sentence contain a negation?',
        'Does the sentence describe a sensory experience?',
        'Does the sentence include technical or specialized terminology?',
        'Does the input contain a number?',
        'Does the sentence contain a cultural reference?',
        'Does the text describe a mode of communication?',
        'Does the input include a comparison or metaphor?',
        'Does the sentence express a sense of belonging or connection to a place or community?',
        'Does the sentence describe a specific sensation or feeling?',
        'Does the text include a planning or decision-making process?',
        'Does the sentence include a personal anecdote or story?',
        'Does the sentence involve a discussion about personal or social values?',
        'Does the text describe a journey?',
        'Does the input contain a measurement?',
        'Does the sentence describe a physical sensation?',
        'Does the sentence include a direct speech quotation?',
        'Is the sentence reflective, involving self-analysis or introspection?',
        'Does the input describe a specific texture or sensation?',
    ]

    prompt_template = 'Read the input then answer a question about the input.\nInput: {example}\nQuestion: {question} Answer yes or no.'
    prompt = prompt_template.format(
        example=ngrams_list[100], question='Does the sentence include a metaphor?')

    print('example prompt', repr(prompt_template.format(
        example=ngrams_list_total[100], question=questions[0])))
    lm = imodelsx.llm.get_llm('gpt-4-turbo-0125-spot')

    answers = []
    for question in tqdm(questions):
        out_file = f'/home/chansingh/mntv1/deep-fMRI/qa/cache_gpt/{question}.pkl'
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
            joblib.dump((answers.values == 'yes').astype(bool), out_file)
        else:
            answers = joblib.load(out_file)
            print('\tloaded', answers.shape, answers.value_counts())
