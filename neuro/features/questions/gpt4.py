

QS_35_STABLE = [
    'Does the sentence describe a personal reflection or thought?',
    'Does the sentence contain a proper noun?',
    'Does the sentence describe a physical action?',
    'Does the sentence describe a personal or social interaction that leads to a change or revelation?',
    'Does the sentence involve the mention of a specific object or item?',
    'Does the sentence involve a description of physical environment or setting?',
    'Does the sentence describe a relationship between people?',
    'Does the sentence mention a specific location?',
    'Is time mentioned in the input?',
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

# note: only 54 stable in this setting as 2 qs get dropped from the previous 36
QS_36_56_STABLE = [
    'Is the input about a discovery or realization?',
    'Does the sentence include an account of a miscommunication or misunderstanding?',
    'Does the sentence include a specific sound or auditory description?',
    'Does the sentence use a unique or unusual word?',
    'Does the sentence describe a change in a physical or emotional state?',
    'Does the sentence describe a moment of relief or resolution of tension?',
    'Does the sentence include a conditional clause?',
    'Does the sentence reference a specific time or date?',
    "Is the sentence conveying the narrator's physical movement or action in detail?",
    'Is there mention of a city, country, or geographic feature?',
    'Does the sentence involve an unexpected incident or accident?',
    'Does the sentence involve a recount of a social or community event?',
    'Does the sentence express a philosophical or existential query or observation?',
    'Does the story involve a personal project or creation?',
    'Is the sentence emotionally positive?',
    'Does the sentence describe an activity related to daily life or routine?',
    'Does the text include a reference to a past era or time period?',
    'Does the input discuss a societal issue or social justice topic?',
    'Does the sentence convey a decision or choice made by the narrator?',
    'Does the sentence convey a sense of urgency or haste?',
    'Is the sentence providing an explanation or rationale?',
]

QS_HYPOTHESES = [
    'Does the input mention anything related to food?',
]

QUESTIONS_GPT4 = QS_35_STABLE + QS_36_56_STABLE + QS_HYPOTHESES

if __name__ == '__main__':
    assert len(QS_35_STABLE) == 35
    assert len(QS_36_56_STABLE) == 56-35, len(QS_36_56_STABLE)