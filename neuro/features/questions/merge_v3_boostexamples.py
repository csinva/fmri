PROMPT_MERGE = '''### Below is a numbered list of questions. Return an output in json format that groups together questions that are very similar to each other. Be comprehensive but keep the groups precise. Some groups may only have 2 questions.

{qs_v3_boostexamples_numbered}

### Example to start: 

{
    "numbers": [
        "Does the input contain a number?",
        "Does the sentence include a number or statistic?",
        "Is a mathematical concept or number mentioned?",
        "Does the sentence include numerical information?",
    ],
}
'''


DICT_MERGE_V3_BOOSTEXAMPLES = {
    "Does the input contain a number?": [
        "Does the input contain a number?",
        "Does the sentence include a number or statistic?",
        "Is a mathematical concept or number mentioned?",
        "Does the sentence include numerical information?",
    ],
    "Does the sentence mention a specific location?": [
        "Does the sentence mention a specific location?",
        "Is a specific location described?",
        "Is a specific location referred to in the input?",
        "Does the sentence reference a specific location or place?",
        "Does the sentence mention a specific location or place?",
    ],
    "Does the input discuss a health issue or concern?": [
        "Does the input discuss a health issue or concern?",
        "Does the input discuss a health-related issue or wellness?",
        "Does the sentence involve a health or wellness practice?",
        "Does the input discuss a medical treatment or procedure?",
        "Is a health condition or disease mentioned?",
        "Is a health or fitness routine described?",
        "Is a health or wellness practice described?",
        "Is there a reference to health and safety precautions?",
        "Is there a reference to physical health or fitness?",
        "Does the sentence relate to personal health or bodily functions?",
        "Is the sentence related to health and fitness advice?",
        "Does the sentence describe a health-related topic or concern?",
        "Does the sentence describe a physical or mental health issue?",
        "Does the sentence involve a personal story of a medical or health issue?",
    ],
    "Does the input contain a question?": [
        "Does the input contain a question?",
        "Does the input discuss a philosophical or existential question?",
        "Does the input involve a philosophical question or theory?",
        "Does the sentence contain a question?",
        "Does the sentence contain a question that encourages reflection?",
        "Does the sentence include a rhetorical question?",
        "Is the sentence a question?",
        "Does the input include a philosophical or reflective thought?",
    ],
    "Does the input mention a form of artistic expression?": [
        "Does the input mention a form of artistic expression?",
        "Does the input mention any form of art or artistic activity?",
        "Does the story mention a form of art or exhibition?",
        "Does the input discuss a craft or DIY project?",
        "Does the story mention a craft or DIY project?",
        'Is the sentence describing a craft or DIY project?',
        "Is an art or craft mentioned?",
        "Does the sentence describe an act of creativity or imagination?",
        "Does the sentence describe an artistic or creative activity?",
        "Does the sentence involve an artistic or creative activity?",
        "Does the sentence involve an artistic or cultural event?",
        "Is there a reference to a famous painting, sculpture, or artist?",
        "Does the sentence describe a creative or artistic activity?",
        "Does the sentence describe an art or craft technique?",
        "Does the sentence involve an act of creativity or artistic expression?",
    ],
    "Does the input contain humor or sarcasm?": [
        "Does the input contain humor or sarcasm?",
        "Is a moment of comedy or humor described?",
        "Does the sentence convey the narrator's sense of humor or wit?",
        "Is the sentence humorous because of a play on words?",
        "Is the sentence humorous?",
        "Does the sentence contain a humorous or ironic comment?",
        "Is the input about a practical joke or prank?",
    ],
    "Does the sentence describe an animal or involve animals?": [
        "Does the sentence describe an animal or involve animals?",
        "Does the sentence describe an interaction with an animal?",
        "Does the sentence include an animal?",
        "Does the input involve animals or wildlife?",
        "Does the input mention a specific animal species?",
        "Does the story involve a pet or domestic animal?",
        "Is a specific type of animal described?",
        "Is the input mentioning an animal?",
        "Is the sentence mentioning a pet or domestic animal?",
    ],
    "Does the input contain negativity?": [
        "Does the input contain negativity?",
        "Is the sentence emotionally negative?"
    ],
    "Does the input contain advice or recommendations?": [
        "Does the input contain advice or recommendations?",
        "Does the text mention a piece of advice?",
        "Is the sentence giving advice or a suggestion?",
    ],
    "Is a piece of technology described?": [
        "Is a piece of technology described?",
        "Does the input discuss a technology trend or future prediction?",
        "Does the story involve a technological problem or solution?",
        "Is a new technological advancement discussed?",
        "Is a specific piece of technology or gadget mentioned?",
        "Is technology or innovation discussed in the input?",
        "Is there a description of a technological device or app?",
        "Is there a discussion of a technological problem or bug?",
        "Is there a mention of a gadget or piece of technology?",
        "Does the sentence describe a technological concept?",
        "Does the sentence involve the use of technology or digital communication?",
    ],
}
