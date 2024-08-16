import os
term_dict = {
    'actions': 'Does the input mention anything related to an action?',
    'arithmetic': 'Does the input mention anything related to arithmetic?',
    'ambiguous': 'Does the input contain a sense of ambiguity?',
    'anger': 'Does the input mention anything related to anger?',
    # 'argue': 'Does the input mention anything related to arguing?',
    'calculation': 'Does the input mention anything related to calculation?',
    'color': 'Does the input mention anything related to color?',
    'conflict': 'Does the input mention anything related to conflict?',
    # 'debate': 'Does the input mention anything related to debate?',
    'disgust': 'Does the input mention anything related to disgust?',
    'empathy': 'Does the input mention anything related to empathy?',
    # 'exact': 'Does the input mention anything related to exactness?',
    'face': 'Does the input mention anything related to faces?',
    # 'fashion': 'Does the input mention anything related to fashion?',
    # 'fast': 'Does the input mention anything related to speed?',
    'fear': 'Does the input mention anything related to fear?',
    'sad': 'Does the input mention anything related to sadness?',
    'unpleasant': 'Does the input mention anything unpleasant?',
    'hands': 'Does the input mention anything related to hands?',
    'alcohol': 'Does the input mention anything related to alcohol?',
    'age': 'Does the input mention anything related to age?',
    'children': 'Does the input mention anything related to children?',
    'diseases': 'Does the input mention anything related to diseases?',
    'eyes': 'Does the input mention anything related to eyes?',
    'knowledge': 'Does the input mention anything related to knowledge?',
    'gender': 'Does the input mention anything related to gender?',
    'navigation': 'Does the input mention anything related to navigation?',
    'motor': 'Does the input mention anything related to motor movements?',
    'sounds': 'Does the input mention anything related to sounds?',
    'taste': 'Does the input mention anything related to taste?',
    'emotions': 'Does the input mention or describe highly positive emotional valence?',
    'negative-emotions': 'Does the input mention or describe highly negative emotional valence?',

    # hand-matched
    'sensation': 'Does the sentence describe a physical sensation?',
    'planning': 'Does the input involve planning or organizing?',
    'food': 'Does the input mention anything related to food?',
    'olfactory': 'Does the input mention or describe a smell?',
    'sound': 'Does the input mention or describe a sound?',
    'emotional-valence': 'Does the input mention or describe high emotional intensity?',
    'negative': 'Does the sentence contain a negation?',
    'thought': 'Does the sentence describe a personal reflection or thought?',
    'sensory': 'Does the sentence describe a sensory experience?',
    'location': 'Does the sentence mention a specific location?',
    'communication': 'Does the text describe a mode of communication?',
    'abstract': 'Is the sentence abstract rather than concrete?',
}


term_dict_rev = {v: k for k, v in term_dict.items()}

if __name__ == '__main__':
    computed_gpt4_qs = [
        x.replace('.pkl', '') for x in os.listdir('/home/chansingh/mntv1/deep-fMRI/qa/cache_gpt')
        if '?' in x
    ]
    print('num qs', len(computed_gpt4_qs))
    computed_matched_gpt4_qs = [
        k for k in computed_gpt4_qs if k in term_dict.values()]
    print('num matched qs', len(computed_matched_gpt4_qs))

    print('UNMATCHED QS')
    for q in computed_gpt4_qs:
        if q not in term_dict.values():
            print(q)
