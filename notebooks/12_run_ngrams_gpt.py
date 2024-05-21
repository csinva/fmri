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
@limits(calls=150, period=30)  # in secs
def call_api_limited(messages):
    resp = lm(messages, max_new_tokens=1, temperature=0,
              return_str=False, verbose=False)
    return resp


def call_api(messages):
    resp = lm(messages, max_new_tokens=1, temperature=0,
              return_str=False, verbose=False)
    return resp


if __name__ == '__main__':
    story_names_list = story_names.get_story_names(
        all=True)
    print('loaded', len(story_names_list), 'stories')
    wordseqs = load_story_wordseqs_huge(story_names_list)
    ngrams_list_total = []
    for story in story_names_list:
        ngrams_list = feature_spaces.get_ngrams_list_main(
            wordseqs[story], num_ngrams_context=10)
        ngrams_list_total.extend(ngrams_list)
    print(f'{len(ngrams_list_total)=} ngrams')

    questions = ['Does the sentence describe a relationship between people?',
                 'Does the sentence contain a proper noun?',
                 'Does the sentence involve a description of an interpersonal misunderstanding or dispute?',
                 'Does the sentence involve the mention of a specific object or item?',
                 'Is time mentioned in the input?',
                 'Does the sentence involve a description of physical environment or setting?',
                 'Does the sentence include dialogue or thoughts directed towards another character?',
                 'Does the sentence describe a physical action?',
                 'Does the sentence include dialogue?',
                 'Does the sentence involve a discussion about personal or social values?',
                 'Does the sentence describe a visual experience or scene?',
                 'Does the sentence describe a personal or social interaction that leads to a change or revelation?',
                 'Does the sentence describe a specific sensation or feeling?',
                 'Is the input related to a specific industry or profession?',
                 'Does the input involve planning or organizing?',
                 'Does the sentence involve an expression of personal values or beliefs?',
                 "Is the sentence conveying the narrator's physical movement or action in detail?",
                 'Does the sentence include technical or specialized terminology?',
                 'Does the sentence contain a cultural reference?',
                 'Does the sentence contain a negation?',
                 "Does the sentence express the narrator's opinion or judgment about an event or character?",
                 'Does the input describe a specific texture or sensation?',
                 'Is the sentence in the passive voice?',
                 'Does the sentence express a sense of belonging or connection to a place or community?',
                 'Does the sentence reference a specific time or date?',
                 'Does the sentence use a unique or unusual word?',
                 'Does the sentence describe a routine activity?',
                 'Does the sentence include a personal anecdote or story?',
                 'Does the text describe a journey?',
                 'Does the input include a comparison or metaphor?',
                 'Does the sentence include a direct speech quotation?',
                 'Does the sentence involve moral reasoning?',
                 'Does the sentence express a philosophical or existential query or observation?',
                 'Does the sentence describe a sensory experience?',
                 'Is the sentence designed to persuade or convince?',
                 'Does the sentence involve spatial reasoning?',
                 'Does the input discuss a societal issue or social justice topic?',
                 'Does the sentence describe a physical sensation?',
                 'Does the sentence use irony or sarcasm?',
                 'Is the sentence a command?',
                 'Is there mention of a city, country, or geographic feature?',
                 'Does the text describe a mode of communication?',
                 'Does the sentence involve a recount of a social or community event?',
                 'Is the sentence providing an explanation or rationale?',
                 'Does the sentence describe a moment of relief or resolution of tension?',
                 'Does the sentence include a conditional clause?',
                 'Does the sentence describe a change in a physical or emotional state?',
                 'Does the sentence involve a safety or security concern?',
                 'Is the sentence structured as a list?',
                 'Is the sentence part of a mystery or puzzle?',
                 'Does the sentence include a threat or warning?',
                 'Does the sentence involve a personal anecdote about family or friends?',
                 'Does the text include a planning or decision-making process?',
                 'Is there a mention of a scientific fact or concept?',
                 'Does the sentence involve a description of adapting to a new environment or situation?',
                 'Does the sentence include an account of a miscommunication or misunderstanding?',
                 'Does the sentence include a specific sound or auditory description?',
                 'Does the sentence describe a physical sensation (e.g., touch, taste)?',
                 'Does the sentence convey a sense of urgency or haste?',
                 'Does the sentence convey a decision or choice made by the narrator?',
                 'Does the sentence describe an indoor setting?',
                 'Does the sentence describe an experience of learning or gaining new knowledge?',
                 'Does the sentence use alliteration or rhyme?',
                 'Is a form of self-improvement or personal development mentioned?',
                 'Is the sentence describing a moment of realization or epiphany?',
                 'Does the input describe a specific emotion in detail?',
                 'Does the sentence describe a significant personal interaction?',
                 'Does the input include a description of clothing?']
    # questions = qa_questions.get_questions('v3_boostexamples', full=True)
    # questions = [
    #     'Does the input include a philosophical or reflective thought?',
    #     'Does the sentence contain a proper noun?',
    #     'Does the sentence describe a physical action?',
    #     'Does the sentence describe a relationship between people?',
    #     'Does the sentence include dialogue or thoughts directed towards another character?',
    #     'Does the sentence involve a description of an interpersonal misunderstanding or dispute?',
    #     'Does the sentence involve a description of physical environment or setting?',
    #     'Does the sentence involve the mention of a specific object or item?',
    #     'Does the sentence reference a specific location or place?',
    #     'Is time mentioned in the input?',
    # ]
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
                    # print(messages)
                    already_cached = lm(messages, max_new_tokens=1, temperature=0,
                                        return_str=False, return_false_if_not_cached=True, verbose=False)
                    if already_cached:
                        resp = call_api(messages)
                    else:
                        resp = call_api_limited(messages)
                    answers.append(resp.choices[0].message.content)

                # print(i, resp.choices[0].message.content)

            answers = pd.Series(answers).str.lower()
            assert set(answers.values) == {'yes', 'no'}
            joblib.dump((answers.values == 'yes').astype(bool), out_file)
