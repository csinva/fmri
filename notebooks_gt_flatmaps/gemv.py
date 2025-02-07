import numpy as np


def get_matched_lists():

    questions_gemv_dict = {
        'Is time mentioned in the input?': ('Times', None),
        # ('time', 212),
        'Does the input contain a measurement?': ('measurements', 171),
        # ('Measurements', None),
        'Does the sentence mention a specific location?': ('locations', 368),
        # ('Location names', None),
        'Does the text describe a mode of communication?': ('communication', 299),

        # specific qa-targets
        'Is the sentence abstract rather than concrete?': ('abstract descriptions', 'qa'),
        'Does the sentence contain a cultural reference?': ('cultural references', 'qa'),
        'Does the sentence include dialogue?': ('dialogue', 'qa'),
        'Is the input related to a specific industry or profession?': ('Professions and Personal Backgrounds', None),
        # ('industry or profession', 'qa'),
        'Does the sentence contain a negation?': ('negations', 'qa'),
        'Does the input contain a number?': ('Numbers', None),
        # ('numbers', 'qa'),
        "Does the sentence express the narrator's opinion or judgment about an event or character?": ('opinions or judgments', 'qa'),
        'Does the sentence describe a personal or social interaction that leads to a change or revelation?': ('personal or interactions interactions', 'qa'),
        'Does the sentence describe a personal reflection or thought?': ('personal reflections or thoughts', 'qa'),
        'Does the sentence involve an expression of personal values or beliefs?': ('personal values or beliefs', 'qa'),
        'Does the sentence describe a physical action?': ('physical actions', 'qa'),
        'Does the input involve planning or organizing?': ('planning or organizing', 'qa'),
        'Does the sentence contain a proper noun?': ('proper nouns', 'qa'),
        'Does the sentence describe a relationship between people?': ('relationships between people', 'qa'),
        # ('Relationships', None)
        'Does the sentence describe a sensory experience?': ('Body parts', None),
        # ('sensory experiences', 'qa'),
        # ('Descriptive elements of scenes or objects', None),
        'Does the sentence involve the mention of a specific object or item?': ('Descriptive elements of scenes or objects', None),
        # ('specific objects or items', 'qa'),
        'Does the sentence include technical or specialized terminology?': ('technical or specialized terminology', 'qa'),

        # extra non-targeted
        'Does the sentence involve a description of physical environment or setting?': ('Direction and location descriptions', None),
        # ('Descriptive elements of scenes or objects', None),
        # ('Scenes and settings', None),
        # ('physical setting', 'roi_rename'),
        'Does the sentence describe a visual experience or scene?': ('Direction and location descriptions', None),
        # ('Descriptive elements of scenes or objects', None),
        # ('Scenes and settings', None),
        'Does the sentence involve spatial reasoning?': ('Spatial positioning and directions', None),
        'Does the input include a comparison or metaphor?': ('abstract descriptions', 'qa'),
        'Does the sentence express a sense of belonging or connection to a place or community?': ('Relationships', None),
        # ('Positive Emotional Reactions', None),
        # ('Sexual and Romantic Interactions', None),
        # ('personal or interactions interactions', 'qa'),
        # ('relationships between people', 'qa'),
        'Does the sentence describe a specific sensation or feeling?': ('Body parts', None),
        # ('sensory experiences', 'qa'),
        'Does the text include a planning or decision-making process?': ('planning or organizing', 'qa'),
        'Does the sentence include a personal anecdote or story?': ('Dialogue', None),
        # ('personal or interactions interactions', 'qa'),
        'Does the sentence involve a discussion about personal or social values?': ('personal values or beliefs', 'qa'),
        'Does the text describe a journey?': ('Spatial positioning and directions', None),
        # ('Direction and location descriptions', None),
        'Does the sentence describe a physical sensation?': ('Body parts', None),
        'Does the sentence include a direct speech quotation?': ('dialogue', 'qa'),
        # ('Dialogue', None),
        'Is the sentence reflective, involving self-analysis or introspection?': ('personal reflections or thoughts', 'qa'),
        # ('Introspection', None),
        # ('Personal growth and reflection', None),
        'Does the input describe a specific texture or sensation?':  ('sensory experiences', 'qa'),
        # # ('Clothing and Physical Appearance', None),
        # ('Body parts', None),

    }

    qa_list = list(questions_gemv_dict.keys())
    gemv_list = list(questions_gemv_dict.values())

    return qa_list, gemv_list
