import numpy as np


def get_matched_lists():
    # matches
    qa_list = [
        # approx matches
        'Is time mentioned in the input?',
        'Does the input contain a measurement?',
        # 'Does the input contain a number?',
        'Does the sentence mention a specific location?',
        # 'Does the sentence describe a relationship between people?',
        # 'Does the sentence describe a relationship between people?',
        'Does the text describe a mode of communication?',
        # 'Does the sentence contain a negation?',
    ]
    gemv_list = [
        # approx matches
        ('time', 212),
        ('measurements', 171),
        # ('measurements', 171),
        # ('moments',	337),
        # ('locations', 122),
        ('locations', 368),
        # ('emotion', 179),
        # ('emotional expression', 398),
        ('communication', 299),
        # ('negativity', 248)
    ]

    qa_list += [
        'Is the sentence abstract rather than concrete?',
        'Does the sentence contain a cultural reference?',
        'Does the sentence include dialogue?',
        'Is the input related to a specific industry or profession?',
        'Does the sentence contain a negation?',
        'Does the input contain a number?',
        "Does the sentence express the narrator's opinion or judgment about an event or character?",
        'Does the sentence describe a personal or social interaction that leads to a change or revelation?',
        'Does the sentence describe a personal reflection or thought?',
        'Does the sentence involve an expression of personal values or beliefs?',
        'Does the sentence describe a physical action?',
        'Does the input involve planning or organizing?',
        'Does the sentence contain a proper noun?',
        'Does the sentence describe a relationship between people?',
        'Does the sentence describe a sensory experience?',
        'Does the sentence involve the mention of a specific object or item?',
        'Does the sentence include technical or specialized terminology?',
    ]

    gemv_list += [
        ('abstract descriptions', 'qa'),
        ('cultural references', 'qa'),
        ('dialogue', 'qa'),
        ('industry or profession', 'qa'),
        ('negations', 'qa'),
        ('numbers', 'qa'),
        ('opinions or judgments', 'qa'),
        ('personal or interactions interactions', 'qa'),
        ('personal reflections or thoughts', 'qa'),
        ('personal values or beliefs', 'qa'),
        ('physical actions', 'qa'),
        ('planning or organizing', 'qa'),
        ('proper nouns', 'qa'),
        ('relationships between people', 'qa'),
        ('sensory experiences', 'qa'),
        ('specific objects or items', 'qa'),
        ('technical or specialized terminology', 'qa'),
    ]

    qa_list += [
        # 'Does the sentence involve a description of physical environment or setting?',
        # 'Does the sentence involve a description of physical environment or setting?',
        'Does the sentence involve a description of physical environment or setting?',
        # 'Does the sentence describe a visual experience or scene?',
        # 'Does the sentence describe a visual experience or scene?',
        'Does the sentence describe a visual experience or scene?',
        'Does the sentence involve spatial reasoning?',
        'Does the input include a comparison or metaphor?',
        'Does the sentence express a sense of belonging or connection to a place or community?',
        # 'Does the sentence express a sense of belonging or connection to a place or community?',
        # 'Does the sentence express a sense of belonging or connection to a place or community?',
        # 'Does the sentence express a sense of belonging or connection to a place or community?',
        # 'Does the sentence express a sense of belonging or connection to a place or community?',
        'Does the sentence describe a specific sensation or feeling?',
        'Does the text include a planning or decision-making process?',
        'Does the sentence include a personal anecdote or story?'
    ]

    gemv_list += [
        # new (if "rename", see match_flatmaps function for what it really is)
        # ('Descriptive elements of scenes or objects', None),
        # ('Scenes and settings', None),
        ('physical setting', 'roi_rename'),
        # ('Descriptive elements of scenes or objects', None),
        # ('Scenes and settings', None),
        ('visual experience', 'roi_rename'),
        ('Spatial positioning and directions', None),
        ('comparison or metaphor', 'roi_rename'),
        ('Relationships', None),
        # ('Positive Emotional Reactions', None),
        # ('Sexual and Romantic Interactions', None),
        # ('personal or interactions interactions', 'qa'),
        # ('relationships between people', 'qa'),
        ('sensory experiences', 'qa'),
        ('planning or organizing', 'qa'),


    ]

    return qa_list, gemv_list


def match_flatmaps(gemv_flatmaps_dict):
    mappings = {
        ('numbers', 'qa'): ('Numbers', None),
        ('time', np.int64(212)): ('Times', None),
        # ('time', np.int64(212)): ('Time and numbers', None),
        ('industry or profession', 'qa'): ('Professions and Personal Backgrounds', None),
        # ('locations', 368): ('Location names', None),
        ('sensory experiences', 'qa'): ('Descriptive elements of scenes or objects', None),
        ('sensory experiences', 'qa'): ('Body parts', None),
        ('specific objects or items', 'qa'): ('Descriptive elements of scenes or objects', None),
        ('specific objects or items', 'qa'): ('Descriptive elements of scenes or objects', None),
        #  ('relationships between people', 'qa'):  ('Relationships', None),
        #  ('measurements', np.int64(171)): ('Measurements', None),
        # ('locations', 368):  ('Location names', None),

        # these are just for renaming
        ('specific objects or items', 'qa'): ('Descriptive elements of scenes or objects', None),
        ('physical setting', 'roi_rename'): ('Direction and location descriptions', None),
        ('visual experience', 'roi_rename'): ('Direction and location descriptions', None),
        ('comparison or metaphor', 'roi_rename'): ('abstract descriptions', 'qa'),
    }

    for k, v in mappings.items():
        gemv_flatmaps_dict[k] = gemv_flatmaps_dict[v]
    return gemv_flatmaps_dict
