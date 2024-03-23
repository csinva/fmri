import sglang as sgl
import numpy as np
from typing import List
import json
import argparse
import os
from tqdm import tqdm
import guidance
import imodelsx.llm

PROMPT_SEMANTIC = '''Generate a bulleted list of 500 diverse, non-overlapping questions that can be used to classify an input based on its semantic properties. Phrase the questions in diverse ways.

Here are some example questions:
- Is the input related to food preparation?
- Does the input mention laughter?
- Does the input contain a number?
- Is time mentioned in the input?
- Is a specific location referred to in the input?
- Does the input contain a measurement?
- Does the input contain negativity?
- Is hair or clothing mentioned in the input?
- Is the input related to physical injury or trauma?

Return only a bulleted list of questions and nothing else'''

ANS_SEMANTIC = '''- Is the input related to food preparation?
- Does the input mention laughter?
- Does the input contain a number?
- Is time mentioned in the input?
- Is a specific location referred to in the input?
- Does the input contain a measurement?
- Does the input contain negativity?
- Is hair or clothing mentioned in the input?
- Is the input related to physical injury or trauma?
- Does the input involve a historical event?
- Is technology or innovation discussed in the input?
- Does the input mention any form of art or artistic activity?
- Is there a reference to a specific individual or public figure?
- Is the input about a scientific discovery or concept?
- Does the input involve animals or wildlife?
- Is there any mention of weather or climate conditions?
- Is a holiday or celebration referenced in the input?
- Does the input involve a vehicle or mode of transportation?
- Is there mention of a book, film, or television show?
- Does the input discuss a health-related issue or wellness?
- Is a sport or physical activity mentioned in the input?
- Does the input involve a legal matter or law enforcement?
- Is there a reference to space, astronomy, or the universe?
- Does the input discuss educational content or learning?
- Is there mention of a political event, policy, or figure?
- Does the input contain a question?
- Is there a reference to music, either listening or playing?
- Does the input involve a financial topic or economic concept?
- Is there mention of a religious or spiritual theme?
- Does the input involve gardening or plant care?
- Is cooking or a recipe discussed in the input?
- Is the input related to a specific industry or profession?
- Does the input mention a construction or building?
- Is there a discussion of a psychological concept or mental health issue?
- Does the input contain advice or recommendations?
- Is there a reference to a cultural tradition or practice?
- Does the input discuss a societal issue or social justice topic?
- Is there mention of a family or personal relationship?
- Does the input involve a game or recreational activity?
- Is a specific piece of technology or gadget mentioned?
- Does the input reference an event or phenomenon in nature?
- Is there mention of a city, country, or geographic feature?
- Does the input discuss travel or exploration?
- Is there a discussion about sleep, dreams, or rest?
- Does the input mention a specific animal species?
- Is a mathematical concept or number mentioned?
- Does the input involve a puzzle or problem to solve?
- Is there a reference to health and safety precautions?
- Does the input discuss an act of kindness or charity?
- Is a specific language or linguistic feature mentioned?
- Does the input involve the creation or invention of something?
- Is there mention of a war, battle, or military event?
- Does the input discuss a personal achievement or milestone?
- Is a health condition or disease mentioned?
- Does the input involve a craft or handiwork project?
- Is there a reference to a myth, legend, or folklore?
- Does the input discuss a business or entrepreneurial venture?
- Is there mention of a concert, play, or live performance?
- Does the input involve a scientific experiment or research study?
- Is a specific school, university, or educational institution mentioned?
- Does the input discuss a technology trend or future prediction?
- Is there a reference to a diet, nutrition, or food choice?
- Does the input involve an outdoor activity or adventure?
- Is there mention of a personal challenge or obstacle overcome?
- Does the input discuss an environmental issue or conservation effort?
- Is a specific chemical or substance mentioned?
- Does the input involve a philosophical question or theory?
- Is there a reference to a famous work of literature or author?
- Does the input discuss a fashion trend or style advice?
- Is there mention of a traditional or cultural dish?
- Does the input involve a fitness routine or exercise?
- Is a specific historical figure or leader mentioned?
- Does the input discuss a craft or DIY project?
- Is there a reference to a social media trend or online phenomenon?
- Does the input involve a coding or programming concept?
- Is there mention of a personal hobby or interest?
- Does the input discuss a medical treatment or procedure?
- Is a specific scientific field or discipline mentioned?
- Does the input involve a debate or controversial topic?
- Is there a reference to a famous painting, sculpture, or artist?
- Does the input discuss a personal goal or aspiration?
- Is there mention of a public service or community project?
- Does the input involve a financial planning or investment strategy?
- Is a specific tool or equipment mentioned?
- Does the input discuss a natural disaster or emergency situation?
- Is there a reference to a philosophical or ethical dilemma?
- Does the input involve a cultural festival or event?
- Is there mention of a unique or unusual hobby?
- Does the input discuss a breakthrough in medical research?
- Is a specific form of literature or writing style mentioned?'''

PROMPT_STORY = '''Generate a bulleted list of 100 diverse, non-overlapping questions that can be used to classify sentences from a first-person story. Phrase the questions in diverse ways.

Here are some example questions:
- Is someone running in the input text?
- Does the input contain an expression of anger?
- Does the input mention a friendship?

Return only a bulleted list of questions and nothing else'''

ANS_STORY = '''- Is someone running in the input text?
- Does the input contain an expression of anger?
- Does the input mention a friendship?
- Is there a description of a meal?
- Does the text describe a journey?
- Is there any mention of a dream or aspiration?
- Does the text include a reflection on past events?
- Is a family member mentioned?
- Does the input describe an animal?
- Is a specific location described?
- Does the story mention a childhood memory?
- Is there an expression of love or affection?
- Does the text talk about a hobby or pastime?
- Is there a description of weather conditions?
- Does the input include dialogue between characters?
- Is a conflict or problem introduced?
- Does the story describe a work or school setting?
- Is there a mention of reading or writing?
- Does the text describe a form of transportation?
- Is a holiday or celebration mentioned?
- Does the input express feelings of sadness?
- Is there a depiction of a physical activity?
- Does the story mention a health issue or injury?
- Is a piece of technology described?
- Does the text involve a financial transaction?
- Is a cultural practice or tradition mentioned?
- Does the input include a description of clothing?
- Is there a mention of music or dancing?
- Does the story describe a building or architecture?
- Is there an expression of surprise?
- Does the text mention a plant or gardening?
- Is a historical event referenced?
- Does the input describe a body of water?
- Is an art or craft mentioned?
- Does the story involve a game or sport?
- Is there a discussion of a social issue?
- Does the text mention a celebrity or public figure?
- Is there a description of a landscape or natural scene?
- Does the input express gratitude?
- Is a religious or spiritual practice mentioned?
- Does the story describe a shopping experience?
- Is a meal being prepared or cooked?
- Does the text involve a journey by foot?
- Is there a mention of a gift or present?
- Does the input describe a nighttime setting?
- Is a fear or phobia expressed?
- Does the story include a lesson or moral?
- Is there a mention of a fictional character or story?
- Does the text describe a hobby in detail?
- Is a personal goal or objective mentioned?
- Does the input include a comparison or metaphor?
- Is there an expression of frustration or annoyance?
- Does the story mention an invention or discovery?
- Is a specific type of animal described?
- Does the text include a prediction or expectation?
- Is an outdoor activity mentioned?
- Does the input describe a specific emotion in detail?
- Is a friendship being formed or ending?
- Does the story talk about a personal challenge?
- Is a cultural event or festival mentioned?
- Does the text describe an act of kindness?
- Is there a mention of a natural disaster?
- Does the input include a philosophical or reflective thought?
- Is a specific season described?
- Does the story describe a ceremony or ritual?
- Is there a mention of a hobby or interest?
- Does the text describe a mode of communication?
- Is an apology or expression of regret included?
- Does the story mention a form of entertainment?
- Is a health or wellness practice described?
- Does the input mention a change in a relationship?
- Is there a description of a unique or unusual event?
- Does the story include a portrayal of a city or urban setting?
- Is a professional or career-related decision discussed?
- Does the text mention a piece of advice?
- Is there an expression of hope or optimism?
- Does the input describe a specific texture or sensation?
- Is a personal transformation or change described?
- Does the story involve planning or preparing for an event?
- Is there a mention of a puzzle or mystery?
- Does the text include a reference to a past era or time period?
- Is a form of leisure or relaxation described?
- Does the input mention a routine or daily activity?
- Is there a description of a fantasy or imaginary scenario?
- Does the story involve a technological problem or solution?
- Is an environmental concern or issue mentioned?
- Does the text describe a personal achievement?
- Is a legal or ethical dilemma discussed?
- Does the input include a reference to a book or movie?
- Is a specific color or pattern described?
- Does the story mention a form of art or exhibition?
- Is a culinary experience or taste described?
- Does the text involve a financial goal or saving?
- Is a historical figure or leader mentioned?
- Does the input describe a scenic view or panorama?
- Is a superstition or myth discussed?
- Does the story include an act of bravery or courage?
- Is a personal belief or value expressed?
- Does the text mention a form of manual labor or craft?
- Is a social gathering or event described?'''

ANS_STORY_FOLLOWUP = '''- Does the story mention an act of charity or volunteering?
- Is there a discussion about politics or government?
- Does the input describe a scientific experiment or discovery?
- Is a language or linguistic detail mentioned?
- Does the story involve a pet or domestic animal?
- Is there a depiction of a dream or nightmare?
- Does the text mention a form of digital communication?
- Is an educational lesson or class described?
- Does the input express feelings of jealousy or envy?
- Is there a mention of a competition or contest?
- Does the story describe a personal crisis or emergency?
- Is a supernatural or paranormal event discussed?
- Does the text include a reference to a sport or athletic activity?
- Is there a description of a childhood toy or game?
- Does the input mention a form of artistic expression?
- Is a health or fitness routine described?
- Does the story talk about a vacation or travel experience?
- Is there a mention of an environmental or ecological practice?
- Does the text describe a personal space or room?
- Is a significant life event, like a wedding or graduation, mentioned?
- Does the input include an expression of disappointment?
- Is there a discussion of a new skill or hobby being learned?
- Does the story mention a form of public transportation?
- Is a natural phenomenon, like an eclipse or meteor shower, described?
- Does the text involve a safety or security measure?
- Is there a depiction of a relaxing or peaceful moment?
- Does the input mention a piece of furniture or home decor?
- Is a seasonal holiday or event described?
- Does the story talk about a misunderstanding or confusion?
- Is there a mention of a geographical feature, like a mountain or river?
- Does the text include a planning or decision-making process?
- Is a form of self-improvement or personal development mentioned?
- Does the input describe a moment of realization or epiphany?
- Is there a depiction of a routine or habit?
- Does the story mention a craft or DIY project?
- Is a mythical or legendary story referenced?
- Does the text involve a financial challenge or setback?
- Is there a discussion of a family tradition or custom?
- Does the input describe an encounter with a stranger?
- Is a scientific concept or theory mentioned?
- Does the story talk about an act of defiance or rebellion?
- Is there a mention of a space or astronomical phenomenon?
- Does the text describe a form of therapy or healing?
- Is a cultural or societal norm discussed?
- Does the input include a mention of a personal idol or hero?
- Is there a description of a technological device or app?
- Does the story involve a form of meditation or mindfulness?
- Is a historic landmark or monument described?
- Does the text mention a form of alternative transportation?
- Is a personal philosophy or worldview expressed?
- Does the input describe a moment of teamwork or collaboration?
- Is there a mention of a childhood aspiration or dream job?
- Does the story talk about a personal sacrifice or compromise?
- Is an unexpected discovery or surprise mentioned?
- Does the text include a reference to a famous painting or sculpture?
- Is a moment of solitude or reflection described?
- Does the input mention a social or community issue?
- Is there a discussion about the future or predictions?
- Does the story involve a close call or narrow escape?
- Is a personal preference or taste described?
- Does the text mention a fitness or sports achievement?
- Is there a depiction of an urban or rural difference?
- Does the input describe a moment of cultural shock or adaptation?
- Is a new technological advancement discussed?
- Does the story talk about a loss or letting go?
- Is there a mention of a significant weather event?
- Does the text involve a change in perspective or attitude?
- Is a moment of comedy or humor described?
- Does the input mention a personal insecurity or doubt?
- Is there a discussion about ethical consumption or sustainability?
- Does the story involve a personal project or creation?
- Is a famous quote or saying referenced?
- Does the text describe an adventure or risky activity?
- Is there a mention of a family heirloom or artifact?
- Does the input include a moment of self-discovery?
- Is a significant historical period or event described?
- Does the story talk about a friendship or relationship ending?
- Is there a depiction of a public speaking or performance event?
- Does the text involve a personal challenge or goal being achieved?
- Is a personal or family recipe mentioned?
- Does the input describe a form of local or community service?
- Is there a mention of a new beginning or start?
- Does the story involve a retreat or getaway?
- Is a moment of unexpected kindness or generosity described?
- Does the text mention a personal limit or boundary?
- Is there a discussion of a technological problem or bug?
- Does the input describe an old habit or something the narrator wishes to change?
- Is a form of non-verbal communication depicted?'''

PROMPT_WORDS = '''Generate a bulleted list of 200 diverse, non-overlapping questions that can be used to classify sentences from a first-person story. Phrase the questions in diverse ways.

Here are some example questions:
- Does the input contain emotional words?
- Is there stuttering or uncertainty in the input?
- Is there a first-person pronoun in the input?

Return only a bulleted list of questions and nothing else'''

ANS_WORDS = '''- Does the input contain emotional words?
- Is there stuttering or uncertainty in the input?
- Is there a first-person pronoun in the input?
- Does the sentence describe a physical action?
- Is the input expressing a wish or desire?
- Does the sentence mention a specific location?
- Is there a reference to a past event?
- Does the sentence include dialogue?
- Is there a mention of a future plan or intention?
- Does the input describe a sensory experience?
- Is there any mention of a family member?
- Does the sentence contain a question?
- Is the input expressing gratitude or thanks?
- Does the sentence reference a specific time of day?
- Is there a comparison or metaphor?
- Does the input describe a problem or challenge?
- Is there an expression of regret or apology?
- Does the sentence include a cultural reference or idiom?
- Is the input mentioning an animal?
- Does the sentence describe weather conditions?
- Is there a mention of a book, movie, or song?
- Does the input express surprise or disbelief?
- Is the sentence giving advice or a suggestion?
- Does the input involve planning or organizing?
- Is there a mention of a historical event?
- Does the sentence include technical or specialized terminology?
- Is the input reflecting on a personal change or growth?
- Does the sentence describe a physical sensation?
- Is there a mention of a hobby or leisure activity?
- Does the input discuss a health issue or concern?
- Is the sentence providing an explanation or rationale?
- Does the input contain humor or sarcasm?
- Is there a reference to a dream or aspiration?
- Does the sentence mention a mode of transportation?
- Is the input about a relationship with another person?
- Does the sentence include a number or statistic?
- Is there an expression of fear or anxiety?
- Does the input discuss a moral or ethical dilemma?
- Is the sentence describing a routine or habit?
- Does the input involve a comparison of before and after?
- Is there a mention of a specific food or drink?
- Does the sentence express confidence or certainty?
- Is the input about a discovery or realization?
- Does the sentence include a foreign word or phrase?
- Is there an expression of loyalty or commitment?
- Does the input mention a specific event or festival?
- Is the sentence describing a work or school-related task?
- Does the input express disappointment or dissatisfaction?
- Is there a mention of a physical object or item?
- Does the sentence involve making a choice or decision?
- Is the input about an interaction with technology?
- Does the sentence describe an artistic or creative activity?
- Is there a reference to physical health or fitness?
- Does the input mention a political or social issue?
- Is the sentence expressing admiration or praise?
- Does the input involve a financial transaction or discussion?
- Is there a mention of a place of worship or religious practice?
- Does the sentence describe a landscape or natural feature?
- Is the input about a personal achievement or milestone?
- Does the sentence involve a safety or security concern?
- Is there a mention of a game or sport?
- Does the input describe an emotional support or comfort?
- Is the sentence referencing legal matters or rights?
- Does the input discuss environmental concerns or issues?
- Is there a mention of a scientific fact or concept?
- Does the sentence involve a personal preference or taste?
- Is the input about overcoming a fear or challenge?
- Does the sentence mention a holiday or celebration?
- Is there an expression of surprise or unexpected outcome?
- Does the input discuss a collaboration or team effort?
- Is the sentence describing a misunderstanding or confusion?
- Does the input involve a critique or review of something?
- Is there a mention of a tradition or custom?
- Does the sentence express longing or nostalgia?
- Is the input about a practical joke or prank?
- Does the sentence involve a health or wellness practice?
- Is there a reference to a childhood memory?
- Does the input discuss a philosophical or existential question?
- Is the sentence mentioning a pet or domestic animal?
- Does the input express empathy or compassion?
- Is there a mention of a gadget or piece of technology?
- Does the sentence describe a challenge or obstacle overcome?
- Is the input about a social or community event?
- Does the sentence mention a unique or unusual experience?
- Is there a discussion of a lesson learned or wisdom gained?
- Does the input express a political or social opinion?
- Is the sentence describing a craft or DIY project?
- Does the input mention a fear or phobia?
- Is there a reference to a personal idol or role model?
- Does the sentence involve a negotiation or compromise?
- Is the input about a misunderstanding or assumption?
- Does the sentence describe a scenic view or landscape?
- Is there an expression of pride or self-esteem?
- Does the input mention a cultural or societal norm?
- Is the sentence involving a risk or dare?
- Does the input discuss a personal limitation or weakness?
- Is there a mention of a musical experience or concert?'''


def get_questions():
    qs_semantic = [q.strip('- ') for q in ANS_SEMANTIC.split('\n')]
    qs_story = [q.strip('- ') for q in ANS_STORY.split('\n')]
    qs_story_followup = [q.strip('- ') for q in ANS_STORY_FOLLOWUP.split('\n')]
    qs_words = [q.strip('- ') for q in ANS_WORDS.split('\n')]
    return list(set(qs_semantic + qs_story + qs_story_followup + qs_words))


if __name__ == "__main__":
    print(len(get_questions()), 'questions')
