import numpy as np


def get_matched_lists():

    questions_gemv_dict = {
        'Is time mentioned in the input?': ('time', 212),
        'Does the input contain a measurement?': ('measurements', 171),
        'Does the sentence mention a specific location?': ('locations', 368),
        'Does the text describe a mode of communication?': ('communication', 299),

        # specific qa-targets
        'Is the sentence abstract rather than concrete?': ('abstract descriptions', 'qa'),
        'Does the sentence contain a cultural reference?': ('cultural references', 'qa'),
        'Does the sentence include dialogue?': ('dialogue', 'qa'),
        'Is the input related to a specific industry or profession?': ('industry or profession', 'qa'),
        'Does the sentence contain a negation?': ('negations', 'qa'),
        'Does the sentence contain a number?': ('numbers', 'qa'),
        "Does the sentence express the narrator's opinion or judgment about an event or character?": ('opinions or judgments', 'qa'),
        'Does the sentence describe a personal or social interaction that leads to a change or revelation?': ('personal or interactions interactions', 'qa'),
        'Does the sentence describe a personal reflection or thought?': ('personal reflections or thoughts', 'qa'),
        'Does the sentence involve an expression of personal values or beliefs?': ('personal values or beliefs', 'qa'),
        'Does the sentence describe a physical action?': ('physical actions', 'qa'),
        'Does the input involve planning or organizing?': ('planning or organizing', 'qa'),
        'Does the sentence contain a proper noun?': ('proper nouns', 'qa'),
        'Does the sentence describe a relationship between people?': ('relationships between people', 'qa'),
        'Does the sentence describe a sensory experience?': ('sensory experiences', 'qa'),
        'Does the sentence involve the mention of a specific object or item?': ('specific objects or items', 'qa'),
        'Does the sentence include technical or specialized terminology?': ('technical or specialized terminology', 'qa'),

        # extra non-targeted
    }

    qa_list = list(questions_gemv_dict.keys())
    gemv_list = list(questions_gemv_dict.values())

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
        'Does the sentence include a personal anecdote or story?',
        # 'Does the sentence include a personal anecdote or story?',
        # 'Does the sentence include a personal anecdote or story?',
        'Does the sentence involve a discussion about personal or social values?',
        # 'Does the text describe a journey?',
        'Does the text describe a journey?',
        'Does the sentence describe a physical sensation?',
        # 'Does the sentence include a direct speech quotation?',
        'Does the sentence include a direct speech quotation?',
        # 'Is the sentence reflective, involving self-analysis or introspection?',
        'Is the sentence reflective, involving self-analysis or introspection?',
        # 'Is the sentence reflective, involving self-analysis or introspection?',
        'Does the input describe a specific texture or sensation?',
        # 'Does the input describe a specific texture or sensation?',
        # 'Does the input describe a specific texture or sensation?',

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
        ('community', 'roi_rename'),
        # ('Positive Emotional Reactions', None),
        # ('Sexual and Romantic Interactions', None),
        # ('personal or interactions interactions', 'qa'),
        # ('relationships between people', 'qa'),
        ('sensory experiences', 'qa'),
        ('planning or organizing', 'qa'),
        ('personal anecdote', 'roi_rename'),
        # ('personal or interactions interactions', 'qa'),
        # ('Personal growth and reflection', None),
        ('personal values or beliefs', 'qa'),
        # ('Direction and location descriptions', None),
        ('Spatial positioning and directions', None),
        ('Body parts', None),
        # ('Dialogue', None),
        ('dialogue', 'qa'),
        # ('Introspection', None),
        ('personal reflections or thoughts', 'qa'),
        # ('Personal growth and reflection', None),
        ('specific sensation', 'roi_rename'),
        # # ('Clothing and Physical Appearance', None),
        # ('Body parts', None),
    ]

    return qa_list, gemv_list


def match_flatmaps(gemv_flatmaps_dict):
    mappings = {
        # these remap matches before roi rename
        ('numbers', 'qa'): ('Numbers', None),
        ('time', np.int64(212)): ('Times', None),
        # ('time', np.int64(212)): ('Time and numbers', None),
        ('industry or profession', 'qa'): ('Professions and Personal Backgrounds', None),
        # ('locations', 368): ('Location names', None),
        # ('sensory experiences', 'qa'): ('Descriptive elements of scenes or objects', None),
        ('sensory experiences', 'qa'): ('Body parts', None),
        ('specific objects or items', 'qa'): ('Descriptive elements of scenes or objects', None),
        #  ('relationships between people', 'qa'):  ('Relationships', None),
        #  ('measurements', np.int64(171)): ('Measurements', None),
        # ('locations', 368):  ('Location names', None),

        # these are just for renaming
        ('physical setting', 'roi_rename'): ('Direction and location descriptions', None),
        ('visual experience', 'roi_rename'): ('Direction and location descriptions', None),
        ('comparison or metaphor', 'roi_rename'): ('abstract descriptions', 'qa'),
        ('community', 'roi_rename'): ('Relationships', None),
        ('personal anecdote', 'roi_rename'): ('Dialogue', None),
        ('specific sensation', 'roi_rename'): ('sensory experiences', 'qa'),
    }

    for k, v in mappings.items():
        gemv_flatmaps_dict[k] = gemv_flatmaps_dict[v]
    return gemv_flatmaps_dict
