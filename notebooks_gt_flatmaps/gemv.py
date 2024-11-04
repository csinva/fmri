
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
        ('technical or specialized terminology', 'qa')
    ]
    return qa_list, gemv_list
