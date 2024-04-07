import re
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
- Does the input contain repetition?
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


PROMPT_NEURO = '''Based on what you know from the neuroscience and psychology literature, generate a bulleted list of 100 diverse, non-overlapping yes/no questions that ask about properties of a sentence that might be important for predicting brain activity.

Return only a bulleted list of questions and nothing else'''

ANS_NEURO = '''- Does the sentence contain emotionally charged words?
- Is the sentence grammatically complex?
- Does the sentence include a metaphor?
- Does the sentence require inferential reasoning to understand?
- Is the sentence part of a narrative?
- Does the sentence describe a physical action?
- Does the sentence include numerical information?
- Is the sentence in the active voice?
- Is the sentence in the passive voice?
- Does the sentence involve a social interaction?
- Does the sentence contain a negation?
- Is the sentence a question?
- Does the sentence describe a sensory experience?
- Is the sentence humorous?
- Does the sentence contain a contradiction?
- Does the sentence reference a specific time or date?
- Is the sentence abstract rather than concrete?
- Does the sentence involve moral reasoning?
- Does the sentence contain a proper noun?
- Is the sentence structured as a list?
- Does the sentence use future tense?
- Does the sentence use past tense?
- Does the sentence include a conditional clause?
- Does the sentence describe an event with emotional significance?
- Is the sentence part of a scientific explanation?
- Does the sentence include a direct speech quotation?
- Does the sentence describe a hypothetical scenario?
- Does the sentence include an idiom?
- Is the sentence autobiographical?
- Does the sentence contain a cultural reference?
- Does the sentence describe a personal preference?
- Is the sentence a command?
- Does the sentence include technical or specialized terminology?
- Does the sentence involve spatial reasoning?
- Does the sentence relate to personal health or bodily functions?
- Is the sentence repetitive or redundant?
- Does the sentence include a comparison or simile?
- Does the sentence reference a famous person or event?
- Does the sentence use non-standard or slang language?
- Is the sentence intentionally vague or ambiguous?
- Does the sentence describe a natural phenomenon?
- Does the sentence involve mathematical reasoning?
- Is the sentence part of a legal document or text?
- Does the sentence contain words with strong visual imagery?
- Does the sentence use alliteration or rhyme?
- Is the sentence part of a dialogue?
- Does the sentence describe an ethical dilemma?
- Does the sentence use irony or sarcasm?
- Does the sentence reference historical events?
- Is the sentence emotionally neutral?
- Does the sentence describe a routine activity?
- Is the sentence related to food or eating?
- Does the sentence contain a pun?
- Does the sentence describe a common fear or phobia?
- Is the sentence part of a poem or song?
- Does the sentence describe a technological concept?
- Does the sentence include a threat or warning?
- Is the sentence humorous because of a play on words?
- Does the sentence describe a physical sensation (e.g., touch, taste)?
- Is the sentence designed to persuade or convince?
- Does the sentence reference a specific location or place?
- Does the sentence describe a fictional scenario?
- Is the sentence emotionally positive?
- Is the sentence emotionally negative?
- Does the sentence include an oxymoron?
- Does the sentence describe a relationship between people?
- Does the sentence use a common cliché?
- Is the sentence related to sports or physical exercise?
- Does the sentence describe a creative or artistic activity?
- Does the sentence use a unique or unusual word?
- Is the sentence related to politics or government?
- Does the sentence reference a religious or spiritual concept?
- Does the sentence describe an economic concept or issue?
- Is the sentence about a current event or recent news?
- Does the sentence contain a question that encourages reflection?
- Does the sentence describe a scientific experiment or finding?
- Is the sentence related to environmental issues?
- Does the sentence describe a character in a story?
- Does the sentence use an unusual syntactic structure?
- Is the sentence a well-known quote or saying?
- Does the sentence describe a dream or fantasy?
- Does the sentence involve planning or decision-making?
- Is the sentence related to education or learning?
- Does the sentence describe a cultural practice or tradition?
- Does the sentence include a rhetorical question?
- Is the sentence intended to evoke nostalgia?
- Does the sentence describe an animal or involve animals?
- Is the sentence about a game or leisure activity?
- Does the sentence describe a travel experience?
- Does the sentence involve time travel or hypothetical future scenarios?
- Is the sentence part of a mystery or puzzle?
- Does the sentence describe a common superstition or myth?
- Does the sentence use metaphor to describe emotions?
- Is the sentence related to health and fitness advice?
- Does the sentence describe a personal achievement or milestone?
- Is the sentence intended to be motivational or inspirational?
- Does the sentence involve a logical paradox?
- Does the sentence describe an art or craft technique?
- Is the sentence related to fashion or personal style?'''

PROMPT_NEURO_FOLLOWUP = '''Generate more diverse questions that may occur for a single sentence in a first-person narrative story'''

ANS_NEURO_FOLLOWUP = '''- Does the sentence express a personal fear or anxiety?
- Is the sentence reflective, involving self-analysis or introspection?
- Does the sentence reveal a secret or confidential information?
- Is the sentence focused on a memory from the past?
- Does the sentence express hope or aspiration for the future?
- Is the sentence describing a physical sensation experienced by the narrator?
- Does the sentence include dialogue or thoughts directed towards another character?
- Is the sentence expressing confusion or uncertainty?
- Does the sentence convey a decision or choice made by the narrator?
- Is the sentence describing an immediate reaction to an event or action?
- Does the sentence express regret or remorse for a past action?
- Is the sentence conveying a change in the narrator's emotional state?
- Does the sentence describe a lesson learned or insight gained by the narrator?
- Is the sentence describing a moment of realization or epiphany?
- Does the sentence express gratitude or appreciation for something or someone?
- Is the sentence conveying a sense of loneliness or isolation?
- Does the sentence express anger or frustration directed at a specific situation or person?
- Is the sentence describing a moment of peace or contentment?
- Does the sentence convey a sense of urgency or haste?
- Is the sentence expressing a wish or desire for something to be different?
- Does the sentence describe an interaction that is significant to the narrator's development?
- Is the sentence conveying the narrator's physical movement or action in detail?
- Does the sentence express the narrator's opinion or judgment about an event or character?
- Is the sentence highlighting a contrast between the narrator's past and present self?
- Does the sentence express a sense of belonging or connection to a place or community?
- Is the sentence describing a moment of vulnerability or weakness?
- Does the sentence convey the narrator's sense of humor or wit?
- Is the sentence expressing skepticism or disbelief towards something or someone?
- Does the sentence describe a recurring thought or obsession?
- Is the sentence conveying the narrator's admiration or respect for another character?
- Does the sentence express a philosophical or existential query or observation?
- Is the sentence highlighting a cultural or societal norm that impacts the narrator?
- Does the sentence express a moment of defiance or resistance by the narrator?
- Is the sentence conveying a sense of awe or wonder experienced by the narrator?
- Does the sentence describe the narrator's creative process or artistic inspiration?
- Is the sentence expressing the narrator's aspirations for societal change or impact?
- Does the sentence convey the narrator's relationship with time, such as impatience or nostalgia?
- Is the sentence describing a significant change in the narrator's environment or circumstances?
- Does the sentence express a moment of forgiveness, either giving or receiving?
- Is the sentence conveying a strategic or tactical thought by the narrator?
- Does the sentence express the narrator's attachment or detachment from material possessions?
- Is the sentence describing a ritual or routine important to the narrator?
- Does the sentence convey the narrator's interaction with nature or the outdoors?
- Is the sentence expressing the narrator's response to art or music?
- Does the sentence describe a moment of companionship or solitude?'''

PROMPT_RANDOM_DATA_EXAMPLES = '''Here are some example narrative sentences:
- away another pile and that lasted for about five minutes before
- life and um and i was so looking forward to all
- true okay and this guy scares the hell out of me
- in which he kept notes in the labor camp on a
- she could go in and out of so she can have
- as he does i look at him and i take the
- hit me in the head and came around and hit me
- con or a box containing the letters that i'd written them
- internship at an insurance company i remember my first day there
- sixteen years old and i'm a huge fan of the yankees
- the back of the head and he cried out spun around
- been dating this guy that i really like and um i
- feeling a little down about myself all i have to do
- great grandfather's decision but also understand what that cost the japanese
- they're not just figments of the imagination they're complete whole real
- to write it all down for him he turned and said
- were gon they that's why i got the cat scan i
- the boogeyman lived there was a place with all the crazy
- to point them to you and let you tell this story
- of whom had done exactly the same thing as me they'd
- appetite of his patrons and so when he started looking around
- that we would share every night um and each night we'd
- juggling and jesus yeah you can laugh that's that's the most
- okay so great there are farmers and there are gold miners
- birthday wish was to have blonde hair blue eyes and to
- like uh alright and she goes outside and uh ok ah
- to have affairs with married women or or signing up to
- free events to try and get people in the doors in
- read my letter but he wanted me to come visit him
- him so i agreed we set a date for seven pm
- a new one i was starting in los angeles and the
- best she said i wasn't just raised on a farm but
- of doing in a million years and i found out how
- so i think about ivy's hair now and how it's growing
- to have a child that's not true when you become a
- maybe every day you know thousands of times you know what's
- i told him i admired his confidence when i tell people
- not expecting much of an answer as far as i could
- most recent scan it appeared that my cancer was now in
- dating girls like from fifteen and i just ended up feeling
- western wall painted our faces with dead sea mud and fell
- those around them that were ricocheting across dc and i mean
- like powerlifting cinder blocks while screaming oy oy oy but it
- my mama had said you'll never know what it's like to
- friendship resulted in me the change in me and uh thank
- when we were nine years old we found something now had
- great i mean i've spent an entire week just practicing how
- i start looking at him he's fidgeting with stuff he's turning
- to go to sleep but you do gotta get off the
- many places around the world i actually helped foster new technologies
- from things big and small like finding love and stupid simple
- my hands on and i wrote this piece in this class
- his head and his face and besides he's going to be
- this is true my friends all drop trow they all went
- any more but i began to think all kinds of things
- a middle finger to all that is holy and good and
- i went home i was really nervous about my mom cause
- tape it closed so i'm like okay and i'm like can
- christmas and new year and um so i go down to
- the article i mean this is a local paper they didn't
- he says to me your lips are blue and so i
- minnesotans say chaska now that's only about an hour drive from
- communications audit assess your customer service capabilities these are things i
- we couldn't agree on where to go for dinner chinese or
- dj was driving and i was sitting in the back with
- food in the world there which um you may not know
- care my friend was dead and these two idiots had caused
- of my dogs that i knew now had died well once
- they also uh analyzed the blood and produced a poster for
- this new environment where the t roads are lined with beautiful
- far as blue squares go i can rewrite their memory and
- extra mile and ask my dad oh how is blessing where
- i didn't have my first real date until junior year of
- difficult he said and i was scared i did not want
- ten days two weeks every morning we have to go on
- you could tell he was really different he talked about his
- i sort of caught up a little bit and i told
- up to see the doctor now some of them might have
- were like three first three interviews by was by the colleagues
- image uh or track that the you know the cartoons when
- holding on for dear life and i'm clutching my souvenir right
- along um i had a really rough time um i lost


Generate a bulleted list of 100 diverse, non-overlapping yes/no questions that ask about properties of these narrative sentences that might be important for classifying them.
Make sure to focus on the given sentences.
Return only a bulleted list of questions and nothing else
Give an example from the above sentences in parentheses for each bullet.
Example:
- Does the sentence mention an age? (sixteen years old and i'm a huge fan of the yankees)'''


ANS_RANDOM_DATA_EXAMPLES = '''- Does the sentence contain a time duration? (that lasted for about five minutes before)
- Is there a mention of a specific location? (in which he kept notes in the labor camp on a)
- Does the sentence express fear or anxiety? (this guy scares the hell out of me)
- Is there a mention of a physical object? (a box containing the letters that i'd written them)
- Does the sentence involve a work or professional setting? (internship at an insurance company i remember my first day there)
- Is a specific age mentioned? (sixteen years old and i'm a huge fan of the Yankees)
- Does the sentence describe an action being done to someone? (hit me in the head and came around and hit me)
- Does it mention a relationship or social interaction? (been dating this guy that i really like and um i)
- Is there a mention of a personal emotion or feeling? (feeling a little down about myself all i have to do)
- Does the sentence involve historical or cultural references? (great grandfather's decision but also understand what that cost the Japanese)
- Does the sentence contain elements of the supernatural or fantastical? (the boogeyman lived there was a place with all the crazy)
- Is there a reference to a specific time of day? (we set a date for seven pm)
- Does the sentence mention a physical condition or health issue? (most recent scan it appeared that my cancer was now in)
- Is there a mention of a hobby or leisure activity? (like powerlifting cinder blocks while screaming oy oy oy but it)
- Does the sentence express gratitude or acknowledgment? (friendship resulted in me the change in me and uh thank)
- Is there a discovery or realization mentioned? (when we were nine years old we found something now had)
- Does the sentence describe practicing or learning a new skill? (great i mean i've spent an entire week just practicing how)
- Is there mention of technology or innovation? (many places around the world i actually helped foster new technologies)
- Does the sentence involve food or dining? (we couldn't agree on where to go for dinner chinese or)
- Is there a reference to an animal or pet? (of my dogs that i knew now had died well once)
- Does the sentence involve travel or movement? (dj was driving and i was sitting in the back with)
- Is a specific physical sensation described? (he says to me your lips are blue and so i)
- Does the sentence mention a specific educational or academic activity? (my hands on and i wrote this piece in this class)
- Is there a mention of a personal achievement or milestone? (i didn't have my first real date until junior year of)
- Does the sentence express a personal preference or desire? (birthday wish was to have blonde hair blue eyes and to)
- Is there a mention of a physical injury or harm? (the back of the head and he cried out spun around)
- Does the sentence involve a creative or artistic activity? (my mama had said you'll never know what it's like to)
- Is there a reference to a specific cultural or social event? (christmas and new year and um so i go down to)
- Does the sentence involve a conflict or disagreement? (we couldn't agree on where to go for dinner chinese or)
- Is a specific historical event or period mentioned? (great grandfather's decision but also understand what that cost the Japanese)
- Does the sentence describe an environmental setting in detail? (this new environment where the t roads are lined with beautiful)
- Is there a mention of performing a specific task or duty? (tape it closed so i'm like okay and i'm like can)
- Does the sentence involve a comparison or analogy? (okay so great there are farmers and there are gold miners)
- Is a significant life event mentioned? (to have a child that's not true when you become a)
- Does the sentence contain humor or sarcasm? (juggling and Jesus yeah you can laugh that's that's the most)
- Is there mention of a legal or official process? (to have affairs with married women or or signing up to)
- Does the sentence involve planning or scheduling? (him so i agreed we set a date for seven pm)
- Is a change in location or setting mentioned? (a new one i was starting in Los Angeles and the)
- Does the sentence describe a personal challenge or difficulty? (difficult he said and i was scared i did not want)
- Is there a mention of a scientific or medical procedure? (were gon they that's why I got the cat scan I)
- Does the sentence describe a physical appearance or feature? (birthday wish was to have blonde hair blue eyes and to)
- Is there a mention of a specific emotion or mental state? (not expecting much of an answer as far as i could)
- Does the sentence involve receiving or giving advice? (my mama had said you'll never know what it's like to)
- Is there a reference to a religious or spiritual practice? (western wall painted our faces with dead sea mud and fell)
- Does the sentence mention performing an experiment or research? (they also uh analyzed the blood and produced a poster for)
- Is a specific weather condition mentioned? (western wall painted our faces with dead sea mud and fell)
- Does the sentence involve a financial transaction or consideration? (free events to try and get people in the doors in)
- Is there mention of a specific book, movie, or song? (image uh or track that the you know the cartoons when)
- Does the sentence involve the use of technology or gadgets? (i start looking at him he's fidgeting with stuff he's turning)
- Is there a reference to a specific physical location or landmark? (minnesotans say chaska now that's only about an hour drive from)
- Does the sentence describe an emotional or psychological transformation? (friendship resulted in me the change in me and uh thank)
- Is there a mention of a specific cultural or national identity? (great grandfather's decision but also understand what that cost the Japanese)
- Does the sentence involve a mistake or error being made? (care my friend was dead and these two idiots had caused)
- Is there mention of a specific form of entertainment or media? (the article i mean this is a local paper they didn't)
- Does the sentence involve a gift or offering? (con or a box containing the letters that i'd written them)
- Is there a mention of a physical activity or sport? (like powerlifting cinder blocks while screaming oy oy oy but it)
- Does the sentence involve an expression of regret or apology? (any more but i began to think all kinds of things)
- Is there a reference to a specific cultural tradition or practice? (christmas and new year and um so i go down to)
- Does the sentence mention a specific form of communication or language? (communications audit assess your customer service capabilities these are things i)
- Is there mention of a natural or geographical feature? (western wall painted our faces with dead sea mud and fell)
- Does the sentence involve an act of creativity or invention? (many places around the world i actually helped foster new technologies)
- Is there a reference to a social or political issue? (a middle finger to all that is holy and good and)
- Does the sentence describe a change in personal belief or perspective? (great grandfather's decision but also understand what that cost the Japanese)
- Is there mention of a specific hobby or pastime? (my hands on and i wrote this piece in this class)
- Does the sentence involve a surprise or unexpected event? (when we were nine years old we found something now had)
- Is there a mention of a change in health or physical condition? (most recent scan it appeared that my cancer was now in)
- Does the sentence involve a plan or strategy being developed? (to point them to you and let you tell this story)
- Is there a reference to a personal or family tradition? (that we would share every night um and each night we'd)
- Does the sentence involve an expression of hope or aspiration? (to have a child that's not true when you become a)
- Is there a mention of a specific food or cuisine? (food in the world there which um you may not know)
- Does the sentence involve a personal reflection or introspection? (any more but i began to think all kinds of things)
- Is there a reference to a physical or mental health treatment? (were gon they that's why i got the cat scan i)
- Does the sentence describe a learning experience or lesson? (friendship resulted in me the change in me and uh thank)
- Is there mention of a specific form of art or craftsmanship? (my hands on and i wrote this piece in this class)
- Does the sentence involve a public or community event? (free events to try and get people in the doors in)
- Is there a reference to a specific historical figure or celebrity? (they're not just figments of the imagination they're complete whole real)
- Does the sentence mention a specific technological device or platform? (many places around the world i actually helped foster new technologies)
- Is there mention of a personal or family relationship? (been dating this guy that i really like and um i)
- Does the sentence involve a physical or environmental challenge? (this new environment where the t roads are lined with beautiful)
- Is there a reference to a life-changing decision or moment? (great grandfather's decision but also understand what that cost the Japanese)
- Does the sentence describe a routine or daily activity? (maybe every day you know thousands of times you know what's)
- Is there mention of a specific cultural artifact or symbol? (western wall painted our faces with dead sea mud and fell)
- Does the sentence involve an expression of personal identity or values? (i told him i admired his confidence when i tell people)
- Is there a reference to a specific professional field or discipline? (communications audit assess your customer service capabilities these are things i)
- Does the sentence involve a demonstration or protest? (a middle finger to all that is holy and good and)
- Is there mention of a significant personal or family event? (christmas and new year and um so i go down to)
- Does the sentence describe overcoming a challenge or obstacle? (difficult he said and i was scared i did not want)
- Is there a reference to a specific cultural or religious belief? (western wall painted our faces with dead sea mud and fell)
- Does the sentence involve an act of rebellion or defiance? (a middle finger to all that is holy and good and)
- Is there mention of a significant historical or societal change? (great grandfather's decision but also understand what that cost the Japanese)
- Does the sentence describe an act of care or nurturing? (care my friend was dead and these two idiots had caused)
- Is there a reference to a specific piece of legislation or law? (to have affairs with married women or or signing up to)
- Does the sentence involve an exploration or adventure? (many places around the world i actually helped foster new technologies)
- Is there mention of a personal crisis or emergency? (most recent scan it appeared that my cancer was now in)
- Does the sentence describe a moment of realization or epiphany? (when we were nine years old we found something now had)
- Is there a reference to a specific scientific concept or theory? (they also uh analyzed the blood and produced a poster for)
- Does the sentence involve an act of commemoration or remembrance? (con or a box containing the letters that i'd written them)
- Is there mention of a specific cultural icon or landmark? (western wall painted our faces with dead sea mud and fell)
- Does the sentence describe an attempt to solve a problem or resolve a situation? (we couldn't agree on where to go for dinner chinese or)
- Is there a reference to an act of learning from a mistake? (any more but i began to think all kinds of things)'''

PROMPT_RANDOM_DATA_EXAMPLES_2 = '''Here are some example narrative sentences:
- of them we took them to the recycling bin and the
- a helpless victim of a sickness defined by a condition and
- you sure you're gonna just add lesbian to that list like
- them said i'm going to shoot the dog and the other
- take all our kids and go cross country and on the
- boyfriend and he answers so i put a pound coin in
- had blood running down my arms and all over my little
- hours thanks in part to the addition of some lighter fluid
- doing the doctor's advice i enrolled to do actuarial science at
- subject is in school and tell him you know why i
- to the movies i pick them up from school in the
- voicemail on my phone from a man who identifies himself as
- wasn't good but somehow i managed to get through high school
- of leukemia in october of nineteen fifty five there is a
- didn't wanna be so logical i actually wanted something more magic
- gifts and clothes because that usually means going somewhere new and
- him if i had a cat scan he said no and
- by now it's around five o'clock but my brother is in
- and the minister from the church where the service was was
- taken your time you've done the things you wanted to do
- realized that i might have some options after all i picked
- apartment um and in that place we lived like two twenty
- clown troupe alright i knew that one wasn't cool um fifth
- so that's the oregon trail but i have to tell you
- crazy um it's crazy i'm naked with my shoes on like
- catch it bam you know and it this this went on
- that were that were written in response to the very adult
- time conservative evangelical christians but then there was our unofficial faith
- to georgia's death row during which she insisted on feeding me
- i knew it was dirty i waited a moment and then
- road trip with my family as we hit the expanse of
- i understand it's a dry town the president's brought a bottle
- surrounded by water my entire life i didn't like swimming and
- love it we are gonna love her good cause she's ivy
- with those very naughty nicholson boys but uh their mother knew
- sunday my one of my sisters said how can you do
- was rejected by a sandwich i ended an eight year relationship
- wave of loneliness hit things seemed to be going well in
- was they were not scared anymore and uh they were sort
- remember just looking at all the other kids' looks on their
- for a suitable day hike our mistake was continuing past the
- just want you to know that i'm gay and she looked
- one of my benson and hedges menthol light one twenties so
- became a college professor i wouldn't be halfway completed with my
- with him because he's a good conversationalist and i'm a good
- staring right at me and so i scream and i drop
- i to become best friends and now now i'm not saying
- drift and i'm getting a feel what he's saying you can
- ink work this is prison ink he has a prison guard
- a notice on the board asking for those scientists who wished
- another frozen moment i could see the lights peeking over the
- have a body jt was an avatar so let me take
- invitation she said and i see this thing about bringing a
- was bubbling and i was like i love you i love
- my life and know that i will never forget you and
- smiling i get the photo we step you know out of
- my husband's gay so you know it's oklahoma the the runner
- we possibly come back to england if we don't do the
- am i gonna see another day am i gonna see my
- said let's ca let's call the local newspaper let them take
- that sort of dumped out into a beautiful little brick patio
- i think their spiritual leader is my father who i haven't
- of power completely out of sync so class continues for the
- discussed the possibility of comfort care a euphemism for pulling the
- back in the car and he goes over and conferences with
- border and you know across the border different man you know
- protect himself like i was gonna take a swing but i
- and cover years because as elementary kids we went through regular
- clothes you know i gave them a gift of my child
- and negotiations in england we finally became an independent republic called
- squares go i can rewrite their memory and they're not afraid
- want to do with it i explain to him of course
- not not til you're sixteen okay and i was crushed like
- it was from igor he told of his escape just ahead
- and the woman and everybody began to undress the soldiers were
- staying he goes well last night i slept on a park
- i guess she felt sort of mismatched there too so there
- nine thirty or so they called mama and um {cg} i
- detaches from the rocket and then the tofu did four somersaults
- care of me i was still really sick i could barely
- lousy nobody else wants to watch them play so in the
- were treated to tap and ballet lessons by their parents uh

Generate a bulleted list of 100 diverse, non-overlapping yes/no questions that ask about properties of these narrative sentences that might be important for classifying them.
Make sure to focus on the given sentences.
Return only a bulleted list of questions and nothing else
Give an example from the above sentences in parentheses for each bullet.
Example:
- Does the sentence mention a number? (not not til you're sixteen okay and i was crushed like)'''

ANS_RANDOM_DATA_EXAMPLES_2 = '''- Does the sentence mention a specific time of day? (by now it's around five o'clock but my brother is in)
- Does the sentence involve family members? (sunday my one of my sisters said how can you do)
- Is there a mention of a specific location or place? (road trip with my family as we hit the expanse of)
- Does the sentence contain dialogue or reported speech? (them said i'm going to shoot the dog and the other)
- Is there a mention of a specific emotion? (wave of loneliness hit things seemed to be going well in)
- Does the sentence involve an animal? (i to become best friends and now now i'm not saying)
- Is there a reference to a specific date or historical event? (of leukemia in october of nineteen fifty five there is a)
- Does the sentence include a mention of technology? (voicemail on my phone from a man who identifies himself as)
- Is there a mention of a physical activity or action? (catch it bam you know and it this this went on)
- Does the sentence involve a vehicle or form of transportation? (take all our kids and go cross country and on the)
- Is there a reference to an educational setting or activity? (doing the doctor's advice i enrolled to do actuarial science at)
- Does the sentence contain any mention of food or drink? (to georgia's death row during which she insisted on feeding me)
- Is a health condition or medical situation mentioned? (a helpless victim of a sickness defined by a condition and)
- Does the sentence refer to a specific cultural or religious belief? (time conservative evangelical christians but then there was our unofficial faith)
- Is there an explicit mention of a hobby or leisure activity? (were treated to tap and ballet lessons by their parents uh)
- Does the narrative involve a crisis or emergency situation? (and the woman and everybody began to undress the soldiers were)
- Is there a mention of a physical object or item? (one of my benson and hedges menthol light one twenties so)
- Does the sentence involve a financial transaction or mention of money? (boyfriend and he answers so i put a pound coin in)
- Is a specific profession or job role mentioned? (became a college professor i wouldn't be halfway completed with my)
- Does the sentence contain a reference to weather or environmental conditions? (surrounded by water my entire life i didn't like swimming and)
- Is there a mention of a specific artistic or creative activity? (ink work this is prison ink he has a prison guard)
- Does the sentence involve planning or preparation for an event? (invitation she said and i see this thing about bringing a)
- Is a sports activity or event mentioned? (lousy nobody else wants to watch them play so in the)
- Does the sentence include a mention of a legal or illegal activity? (them said i'm going to shoot the dog and the other)
- Is there a reference to a physical injury or accident? (had blood running down my arms and all over my little)
- Does the sentence involve a mention of a geographical feature (e.g., lake, mountain)? (for a suitable day hike our mistake was continuing past the)
- Is the narrative focused on a personal discovery or realization? (realized that i might have some options after all i picked)
- Does the sentence involve a social or community event? (and the minister from the church where the service was was)
- Is there a mention of a specific piece of clothing or accessory? (crazy um it's crazy i'm naked with my shoes on like)
- Does the sentence describe a character's physical appearance or attire? (with him because he's a good conversationalist and i'm a good)
- Is there a mention of a specific behavior or mannerism? (staring right at me and so i scream and i drop)
- Does the sentence involve a technological or scientific concept? (a notice on the board asking for those scientists who wished)
- Is there a reference to a health or wellness practice? (discussed the possibility of comfort care a euphemism for pulling the)
- Does the sentence involve travel to a different country or state? (we possibly come back to england if we don't do the)
- Is there a mention of a specific age or life stage? (not not til you're sixteen okay and i was crushed like)
- Does the sentence contain a reference to a specific literary or artistic work? (that were that were written in response to the very adult)
- Is a specific game or toy mentioned? (squares go i can rewrite their memory and they're not afraid)
- Does the sentence involve a mention of a natural disaster or extreme weather event? (another frozen moment i could see the lights peeking over the)
- Is there a reference to a specific historical period or era? (and negotiations in england we finally became an independent republic called)
- Does the sentence describe a change in a relationship or social status? (was rejected by a sandwich i ended an eight year relationship)
- Is there a mention of performing arts or entertainment (e.g., movie, theater)? (to the movies i pick them up from school in the)
- Does the sentence involve a mention of a health professional or medical advice? (him if i had a cat scan he said no and)
- Is there a reference to a political or governmental action? (border and you know across the border different man you know)
- Does the sentence contain a mention of a specific plant or animal species? (love it we are gonna love her good cause she's ivy)
- Is a specific holiday or festive occasion mentioned? (gifts and clothes because that usually means going somewhere new and)
- Does the sentence describe an act of creativity or creation? (detaches from the rocket and then the tofu did four somersaults)
- Is there a mention of a specific building or structure? (that sort of dumped out into a beautiful little brick patio)
- Does the sentence involve an experience of loss or grief? (my life and know that i will never forget you and)
- Is a specific body part mentioned? (had blood running down my arms and all over my little)
- Does the sentence contain a mention of a specific historical figure or celebrity? (with those very naughty nicholson boys but uh their mother knew)
- Is there a reference to a specific social or political issue? (my husband's gay so you know it's oklahoma the the runner)
- Does the sentence describe a process of learning or education? (just want you to know that i'm gay and she looked)
- Is there a mention of a specific musical or artistic genre? (clown troupe alright i knew that one wasn't cool um fifth)
- Does the sentence involve a specific tradition or cultural practice? (and cover years because as elementary kids we went through regular)
- Is a specific health or safety measure mentioned? (protect himself like i was gonna take a swing but i)
- Does the sentence involve a mention of a fictional character or scenario? (have a body jt was an avatar so let me take)
- Is there a reference to a specific scientific or technological invention? (drift and i'm getting a feel what he's saying you can)
- Does the sentence contain a mention of a specific business or brand? (one of my benson and hedges menthol light one twenties so)
- Is there a mention of a specific educational degree or qualification? (doing the doctor's advice i enrolled to do actuarial science at)
- Does the sentence involve a mention of a specific political or social movement? (time conservative evangelical christians but then there was our unofficial faith)
- Is there a reference to a specific piece of legislation or law? (border and you know across the border different man you know)
- Does the sentence contain a mention of a specific event or occasion? (invitation she said and i see this thing about bringing a)
- Is there a mention of a specific natural resource or environmental concern? (surrounded by water my entire life i didn't like swimming and)
- Does the sentence involve a personal challenge or obstacle? (wasn't good but somehow i managed to get through high school)
- Is there a mention of a specific tool or device? (hours thanks in part to the addition of some lighter fluid)
- Does the sentence involve a mention of a specific cultural or societal norm? (i guess she felt sort of mismatched there too so there)
- Is there a reference to a specific religious figure or deity? (i think their spiritual leader is my father who i haven't)
- Does the sentence describe a specific method or technique? (and the woman and everybody began to undress the soldiers were)
- Is there a mention of a specific personal or family tradition? (sunday my one of my sisters said how can you do)
- Does the sentence involve a specific mental or cognitive process? (didn't wanna be so logical i actually wanted something more magic)
- Is there a mention of a specific dietary habit or preference? (care of me i was still really sick i could barely)
- Does the sentence describe an act of communication or messaging? (voicemail on my phone from a man who identifies himself as)
- Is there a reference to a specific leisure or entertainment device? (lousy nobody else wants to watch them play so in the)
- Does the sentence contain a mention of a specific architectural style or element? (that sort of dumped out into a beautiful little brick patio)
- Is a specific scientific concept or theory mentioned? (a notice on the board asking for those scientists who wished)
- Does the sentence involve a mention of a cultural or historical artifact? (that were that were written in response to the very adult)
- Is there a reference to a specific educational institution or program? (doing the doctor's advice i enrolled to do actuarial science at)
- Does the sentence describe an experience of conflict or confrontation? (and the woman and everybody began to undress the soldiers were)
- Is there a mention of a specific psychological condition or mental health issue? (a helpless victim of a sickness defined by a condition and)
- Does the sentence involve a specific crafting or construction activity? (detaches from the rocket and then the tofu did four somersaults)
- Is there a reference to a specific social or cultural identity? (you sure you're gonna just add lesbian to that list like)
- Does the sentence describe an experience of surprise or shock? (staring right at me and so i scream and i drop)
- Is there a mention of a specific leisure activity or pastime? (were treated to tap and ballet lessons by their parents uh)
- Does the sentence involve a specific form of communication or language use? (drift and i'm getting a feel what he's saying you can)
- Is there a reference to a specific personal achievement or milestone? (became a college professor i wouldn't be halfway completed with my)
- Does the sentence contain a mention of a specific social or community role? (and the minister from the church where the service was was)
- Is there a mention of a specific environmental or ecological issue? (surrounded by water my entire life i didn't like swimming and)'''

PROMPT_BOOST_1 = '''Here are some example narrative sentences:
- at the time and i realized as that came out didn't sound good um
- got two feet away i lifted my can
- i put her name like ivy and she'd be like all the rage at parades
- driving there are no stop lights i have to
- my idea of flirty banter is sending someone
- a bipolar and if you don't know what that is it's just basically a chemical
- no did they find somebody with a weapon no well why did
- us with espionage which was farcical given we had little money no
- and he bought me um a jeans jacket and he would
- that fatima zohra escort me to the town's ramshackle internet
- the village where we were to montevideo or to california
- it's like finding an old cassette tape that you made
- becoming friends with people that maybe i don't
- and wept for a minute and gave thanks as i was really
- you had to go through this concrete corridor that sort of dumped out
- and bizarrely the first thing i thought about was henley royal regatta
- but things at home start to get a little tricky
- you know or sane people do sane people do
- uh which weren't the united states yet uh across the rockies leaving everything
- that parrotfish those parrotfish that gather the together i'd
- the coin
- most of the time everybody only talks about how really happy
- i stood in his doorway she um explained who i was and
- to be in the world that the books showed
- one of these ways was to strike up seemingly natural conversations
- up in her eyes and i said mama what's wrong
- ang among my friends for having tripped over my own feet when i was in college in boston and actually
- and white shirts and it's an italian
- far away from that as possible my parents were uh dyed in the wool liberals
- family and that night is so beautiful
- got my own set of controls he has his own set of controls we
- i'm really excited so i'm in the van going to camp
- the tribal rituals that are used for the treatment of depression here
- love for the first time such intense feelings can make you feel small
- obama bobs his head in time to the music and betty white gets her card and
- i was dating or anything personal going on in my life and
- but i wasn't very happy because i kept thinking about my family
- to a degree of intimacy that includes all kinds of intimate sounds
- and we come to a stoplight and
- in the middle of my life i have spent years
- and then he called and i picked up with the intention
- we used to go to almost every day after school but
- but they really want to confirm if blessing is pregnant
- but as a college professor it's rarely up to me and i
- i told wesley that i was glad he was reading this book because
- was like a maniac you know always screaming goddamn fucking bitch you know
- amazing course that has you spending your days doing
- how's it fucking feel down there bitch
- and think about my missing friend
- i saved up and we got a guide dog and
- and i'm really really nervous because not only am i scared
- old unsent greeting cards throw away another pile
- female open minded easy going down
- could face another execution date in a matter of days
- and then one day
- and i figured there's two ways of looking at it either we should be very grateful that
- a page out of the program that listed all the different you know yankee executives and i wrote
- speeches for people who do not support gay marriage
- there was like a complete silence
- uh in what was called uh the duck and cover years
- game for the start of the game so
- that ancient red rock they wanna see that desert city i mean there are songs
- caught by surprise she succumbed to my rescue
- that i thought of as home right and
- using pamela's age and some of her defining
- and he notices me for the first time sitting in the car and he points
- and i'm also thinking this conversation
- and agitated and leaning against the railing
- and all these gray and so my salary got went up
- year olds got me thinking about the parameters of acceptable
- i had been at a school in the township where we had something called hot seating
- or five jeeps full of mps
- options i could somehow be cool
- because the soldier out outside were very impatient
- all the things that that give us joy
- that was at lackland air force base in texas nineteen sixty
- and um i went they paid for this gold this is true they pay for this thing called
- just don't make pies
- one wanted to answer but i was low seniority so they gave it you know my boss gave it
- sharing beachfront with the kennedy compound
- and he tells me about berlin one evening

Generate a bulleted list of 100 diverse, non-overlapping yes/no questions that ask about properties of these narrative sentences that might be important for classifying them.
Make sure to focus on the given sentences and try to ask questions that combine properties of multiple sentences above.
Return only a bulleted list of questions and nothing else'''

ANS_BOOST_1 = '''- Does the sentence involve a realization or sudden understanding?
- Is there a physical action or movement described?
- Does the sentence include a specific name or designation (like a person's name or a place)?
- Is there mention of transportation or travel?
- Does the sentence include flirting or romantic interaction?
- Is there a mention of a mental health condition or emotional state?
- Does the sentence question the actions or decisions of others?
- Is espionage, surveillance, or conspiracy mentioned?
- Does the sentence describe a gift or an act of giving?
- Is there a mention of using technology or the internet?
- Does the sentence involve geographical movement between cities or countries?
- Is nostalgia or reminiscing about the past a theme?
- Does the sentence describe forming new relationships or friendships?
- Are emotions or reactions to a situation described?
- Does the sentence describe navigating through a specific space or location?
- Is there a mention of a specific event or occasion?
- Does the sentence describe family dynamics or home life?
- Is the sentence about personal habits or behaviors in social situations?
- Does the sentence describe a journey or adventure?
- Is there a description of aquatic life or underwater scenes?
- Is the sentence about making a choice or decision?
- Does the sentence express a generally positive or happy sentiment?
- Does the sentence involve an introduction or explaining one's identity to someone?
- Does the sentence describe a desire to explore or learn from books?
- Is there an attempt to engage someone in conversation mentioned?
- Does the sentence describe a moment of emotional vulnerability?
- Does the sentence recount a humorous or embarrassing personal anecdote?
- Is clothing or personal style a focus of the sentence?
- Does the sentence mention political or ideological beliefs?
- Does the sentence revolve around a family event or gathering?
- Is there mention of operating machinery or vehicles?
- Does the sentence describe excitement or anticipation for an event?
- Are traditional or cultural practices mentioned?
- Is the experience of first love or deep emotion described?
- Are celebrities or public figures mentioned?
- Is personal or private life details being shared?
- Is there an expression of dissatisfaction with personal circumstances?
- Does the sentence discuss aspects of a close relationship?
- Is there a mention of stopping or waiting at an intersection?
- Does the sentence reflect on personal history or life decisions?
- Does the sentence describe initiating a phone call or conversation?
- Is there a mention of a routine or habitual activity?
- Is pregnancy or the potential for it a topic?
- Does the sentence describe academic or professional challenges?
- Is there a recommendation or endorsement of a book?
- Does the sentence include aggressive or violent language?
- Is the sentence about an educational or learning experience?
- Is derogatory language or insults used?
- Does the sentence reflect on a loss or absence of someone?
- Is there mention of acquiring a service animal?
- Does the sentence express fear or anxiety about an upcoming event?
- Is decluttering or disposing of items described?
- Does the sentence describe someone as being approachable or sociable?
- Is there an impending legal or judicial event mentioned?
- Does the sentence describe a moment of change or decision?
- Is gratitude or appreciation for a situation expressed?
- Does the sentence involve collecting or writing down information?
- Does the sentence discuss political or social opinions?
- Is there a sudden shift in the atmosphere or mood mentioned?
- Does the sentence describe a historical period or context?
- Is anticipation for a game or event described?
- Does the sentence describe a desire to visit a natural or scenic location?
- Is there a rescue or assistance in a difficult situation?
- Does the sentence describe a feeling of belonging or home?
- Are physical characteristics or attributes of a person detailed?
- Does the sentence describe a first encounter or noticing someone?
- Does the sentence involve contemplation or reflection on a conversation?
- Is there a description of emotional or physical discomfort?
- Does the sentence describe a financial or career change?
- Does the sentence discuss considerations of social or age-appropriate behavior?
- Is there a mention of an educational method or school activity?
- Are military or law enforcement personnel mentioned?
- Does the sentence suggest looking for ways to be perceived as cool?
- Is there impatience or urgency expressed by characters?
- Does the sentence describe finding joy or happiness in something?
- Is a specific military base or location mentioned?
- Does the sentence describe receiving financial support for something?
- Is there advice against a specific action or behavior?
- Does seniority or hierarchy influence a decision or action?
- Is there a mention of a famous family or property?
- Does the sentence recount a detailed personal experience in a specific city or place?'''

PROMPT_BOOST_2 = '''Here are some example narrative sentences:
- on top of that my my wife got pregnant at the time and i realized as that
- off guard but when he got two feet away
- pounds of baby that i'm analyzing her skull and she goes
- therapy he's had shock treatments he's had every combination
- feeling creepy for staring at these strangers and also envious
- by he was also a bipolar and if
- to play and every so often i turn around to look at mike course
- unsettled i sensed i was now exactly where i should be
- and we would he taught me how to chop wood and he bought me um
- me to the town's ramshackle internet cafe so i could check my e-mail
- we were happy we kept all the time in touch with the family
- just lost his wife
- straight at me and almost
- and the snacks that doesn't even start with the clothing so we overpacked
- over blueprints and the building
- in the morning and the streets of delhi were quiet i'd never seen them quiet before they're normally
- education director and i asked him which class he thought i should take and he
- god's gift to the planet you know or
- trail and he says well there's a water fountain right there and
- resist we called him puff like puff the magic dragon he
- to the car park and i ask to be let down they offer
- about how really happy adoption is and we are
- and i saw that it was only a few miles
- not feel like somebody who had suffered liked someone
- of church that was trying to be really hip and modern it met in a strip mall and
- that water was starting to well up in her eyes and i said
- chloe kept making remarks about how only a few of our friends have been killed by customers
- graduated on to a two room school house in the center of chilmark
- an evangelical christian i grew up as far away from that as possible my parents
- uh to this day you know i still think that he died on purpose
- through it and i'm just like this is so cool i i say
- it's like super cool and um i'm really excited so
- thing and she came out with her very prized possession which was a
- intense feelings can make you feel small in their grasp or
- so we feel good about the joke but but we still need a birthday card so uh one
- ones that make you lose sleep and
- and you know and i
- the weirdest thing is that his friends all went naked
- i shut the door i call out to my wife i said hey babe
- my clothes my books my bookcases
- me twice and i never replied and then he called and
- oh one and we're waiting outside the room
- your space and your family
- rarely up to me and i go new places and i meet new people
- he was reading this book because he needed to understand
- ride back he's like fucking like you know that dude from leave it to beaver
- numbers so you can imagine my joy when i heard
- there bitch with no one writing about you in the new yorker
- seconds and i just fell to the side somebody took my place
- please don't send any more please don't send any more because my
- and my classmate right in front of me is stephanie wilson
- now we have this moment where we're laughing together at this image of my mom like digging through the
- waiting for me from my adult female open minded
- in a matter of days how could i say no to him
- german concentration camp and then
- we should be very grateful that ancient catholic
- i went home that night and i tore a page out of the program that listed all the different
- i got to do at the white house uh some of the proudest moments of my life
- so i had no option but to find a job
- our gear into a small bering airplane
- about keeping promises and about self sacrifice
- he's done his cop training he's like we have evidence
- in distress style caught by surprise
- me to be there that she in fact needed me there for support my brothers too
- i laughed
- with his friends we would just find some little nook or a corner or like go in the backyard
- thinking this conversation about being
- i think it's very difficult to really live any day as
- it was all of a sudden i started feeling things i started feeling hot
- justified or even worse satisfyingly resolved
- where we had something called hot seating which meant that forty eight of us came
- seems like seconds four or five jeeps full of mps
- fifth grade was the last year i can remember believing that the world needed
- outside were very impatient uh waiting for
- i do the camp thing i do what's at camp and i don't know if you know this but there are a lot
- like out cold alright and the surgeon's like do gives him like
- he was no help and um somehow though sam and i survived our
- now probably making sure everyone she shares the afterlife
- eddie heka told you about and
- well where i was from but i felt that somehow or other this
- chicken fights but it's kinda weird because there's only three of us so
- rid of the fear one problem is i don't actually

Generate a bulleted list of 100 specific, non-overlapping yes/no questions that ask about aspects of the above sentences that are important for classifying them.
Focus on the given sentences and form questions that combine shared properties from multiple sentences above.
Return only a bulleted list of questions and nothing else'''

ANS_BOOST_2 = '''- Does the sentence involve a significant life event (e.g., pregnancy, death)?
- Is there a mention of a specific emotional state (e.g., feeling creepy, envious, unsettled)?
- Does the sentence reference a mental health condition (e.g., bipolar disorder, need for therapy)?
- Is there a physical activity or skill being learned (e.g., chopping wood)?
- Does the sentence involve technology or modern conveniences (e.g., internet cafe)?
- Is there a clear expression of happiness or contentment?
- Does the sentence involve a significant other or family member?
- Is there an animal mentioned explicitly or implied (e.g., dragon)?
- Does the sentence contain a reference to a specific location (e.g., Delhi, a church)?
- Is there a mention of an educational setting or learning environment (e.g., school house, class)?
- Does the sentence involve a creative or recreational activity (e.g., playing, joking)?
- Is there a sense of loss or grief expressed?
- Does the sentence involve planning or preparation (e.g., overpacking, looking at blueprints)?
- Is there a mention of a medical or health-related situation?
- Does the sentence reference a historical or significant event (e.g., White House, concentration camp)?
- Is the focus on a personal achievement or a proud moment?
- Does the sentence involve a job or work-related activity?
- Is there an element of surprise or being caught off guard?
- Does the sentence discuss a social issue or a cause (e.g., adoption, self-sacrifice)?
- Is law enforcement or military mentioned or implied?
- Does the sentence involve support or needing support from others?
- Is there a mention of a casual or informal gathering (e.g., with friends, backyard)?
- Does the sentence discuss feelings of physical discomfort or sensations (e.g., feeling hot)?
- Is there a reference to a specific cultural or religious aspect?
- Does the sentence involve an outdoor activity or setting (e.g., trail, camp)?
- Is a specific object of value or importance mentioned (e.g., prized possession)?
- Does the sentence discuss a conflict or problem-solving situation?
- Is there a mention of a fictional or imaginative concept (e.g., magic dragon, afterlife)?
- Does the narrative involve travel or moving to a new place?
- Is there a focus on personal introspection or realization?
- Does the sentence involve a health or medical professional (e.g., surgeon)?
- Is there a mention of a specific time of day or atmospheric condition (e.g., morning, quiet streets)?
- Does the sentence involve a physical or emotional challenge?
- Is the concept of teaching or mentoring explicitly mentioned?
- Does the sentence reference a specific form of communication or interaction (e.g., calling, sending messages)?
- Is there a mention of a specific type of establishment (e.g., strip mall, cafe)?
- Does the narrative involve an aspect of performance or presentation (e.g., acting, at the White House)?
- Is the sentence focused on a personal decision or choice?
- Does the narrative involve an unusual or unexpected behavior (e.g., going naked)?
- Is there a mention of specific types of items or belongings (e.g., books, clothing)?
- Does the sentence reference a financial aspect or consideration?
- Is there a mention of a social or community event (e.g., camp)?
- Does the sentence involve an act of creativity or artistic expression?
- Is there a focus on a personal or family tradition?
- Does the narrative involve a discovery or new experience?
- Is there a mention of a specific feeling of discomfort or distress?
- Does the sentence involve a specific form of transportation (e.g., airplane)?
- Is the concept of time or timing important to the narrative?
- Does the sentence involve a legal or judicial aspect?
- Is there a mention of a specific type of relationship (e.g., classmate, adult female)?
- Does the narrative involve a form of entertainment or leisure activity?
- Is there a specific geographical or historical detail mentioned?
- Does the sentence involve an aspect of physical or environmental change?
- Is the focus on a personal or internal conflict?
- Does the narrative involve an act of helping or assisting others?
- Is there a mention of a specific physical sensation or experience (e.g., falling, feeling small)?
- Does the sentence involve an academic or scholarly activity?
- Is there a reference to a specific type of document or written material (e.g., program, New Yorker article)?
- Does the narrative involve a change in personal beliefs or perspective?
- Is there a mention of a specific health treatment or procedure?
- Does the sentence involve an aspect of personal security or safety?
- Is there a reference to a specific form of media or technology usage?
- Does the narrative involve a reference to a specific age or life stage?
- Is the focus on a specific type of environment or setting (e.g., in distress, quiet streets)?
- Does the sentence involve a mention of food or dietary considerations?
- Is there a focus on emotional support or empathy?
- Does the narrative involve a specific task or duty being performed?
- Is there a mention of a personal habit or routine?
- Does the sentence involve a reference to a specific type of clothing or accessory?
- Is the concept of privacy or personal space discussed?
- Does the narrative involve an aspect of cultural or social identity?
- Is there a mention of a specific type of animal or pet?
- Does the sentence involve an element of risk or danger?
- Is there a reference to a specific type of game or sport?
- Does the narrative involve a financial transaction or exchange?
- Is there a focus on a specific type of emotion or feeling (e.g., gratitude, satisfaction)?
- Does the sentence involve an aspect of personal growth or development?
- Is there a mention of a specific type of music or song?
- Does the narrative involve a form of competition or challenge?
- Is there a reference to a specific type of building or structure?
- Does the sentence involve a personal reflection or insight?
- Is the focus on a specific type of event or occasion (e.g., birthday)?
- Does the narrative involve a specific type of craft or handiwork?
- Is there a mention of a specific type of tool or equipment?
- Does the sentence involve a form of academic or intellectual exploration?
- Is there a focus on a specific type of landscape or natural feature?
- Does the narrative involve a specific type of personal care or hygiene activity?
- Is there a reference to a specific type of artistic or creative work?
- Does the sentence involve a mention of a specific type of professional or expert?
- Is the focus on a specific form of social interaction or relationship dynamics?
- Does the narrative involve a specific type of health or fitness activity?
- Is there a mention of a specific historical period or era?
- Does the sentence involve a specific type of technology or device?
- Is the narrative focused on a personal challenge or adversity?
- Is there a mention of a specific type of group or organization?
- Does the sentence involve an aspect of personal or household maintenance?
- Is there a focus on a specific type of personal achievement or milestone?
- Does the narrative involve a specific type of outdoor or nature activity?
- Is there a mention of a specific type of weather or climate condition?'''


def _split_bulleted_str(s, remove_parentheticals=False):
    qs = [q.strip('- ') for q in s.split('\n')]
    if remove_parentheticals:
        qs = [re.sub(r'\(.*?\)', '', q).strip() for q in qs]
    return qs


def _rewrite_to_focus_on_end(question, suffix='last'):
    if suffix == 'last':
        focus_text = 'In the last word of the input, '
    elif suffix == 'ending':
        focus_text = 'At the end of the input, '
    elif suffix == 'last10':
        focus_text = 'In the last ten words of the input, '
    else:
        raise ValueError(suffix)
    # replace nouns
    question = question.lower().replace('the sentence', 'the text').replace(
        'the story', 'the text').replace('the narrative', 'the text')
    question = question.replace(' in the input?', '?').replace(
        ' in the input text?', '?')
    return focus_text + question


def get_questions(version='v1', suffix=None, full=False):
    '''Different versions
    -last, -ending adds suffixes from last
    '''
    if len(version.split('-')) > 1:
        version, suffix = version.split('-')

    if version == 'v1':
        qs_semantic = _split_bulleted_str(ANS_SEMANTIC)
        qs_story = _split_bulleted_str(ANS_STORY)
        qs_story_followup = _split_bulleted_str(
            ANS_STORY_FOLLOWUP)
        qs_words = _split_bulleted_str(ANS_WORDS)
        qs = qs_semantic + qs_story + qs_story_followup + qs_words
        qs_remove = []
    elif version == 'v2':
        qs_neuro = _split_bulleted_str(ANS_NEURO)
        qs_neuro_followup = _split_bulleted_str(ANS_NEURO_FOLLOWUP)
        qs = qs_neuro + qs_neuro_followup
        qs_remove = get_questions(version='v1', suffix=suffix)
    elif version == 'v3':
        qs_random_data = _split_bulleted_str(
            ANS_RANDOM_DATA_EXAMPLES, remove_parentheticals=True)
        qs_random_data_2 = _split_bulleted_str(
            ANS_RANDOM_DATA_EXAMPLES_2, remove_parentheticals=True)
        qs = qs_random_data + qs_random_data_2
        qs_v1 = get_questions(version='v1', suffix=suffix)
        qs_v2 = get_questions(version='v2', suffix=suffix)
        qs_remove = qs_v1 + qs_v2
    elif version == 'v4':
        qs_boost_1 = _split_bulleted_str(ANS_BOOST_1)
        qs_boost_2 = _split_bulleted_str(ANS_BOOST_2)
        qs = qs_boost_1 + qs_boost_2
        qs_v1 = get_questions(version='v1', suffix=suffix)
        qs_v2 = get_questions(version='v2', suffix=suffix)
        qs_v3 = get_questions(version='v3', suffix=suffix)
        qs_remove = qs_v1 + qs_v2 + qs_v3
    elif version == 'all':
        qs_v1 = get_questions(version='v1', suffix=suffix)
        qs_v2 = get_questions(version='v2', suffix=suffix)
        qs_v3 = get_questions(version='v3', suffix=suffix)
        qs_v4 = get_questions(version='v4', suffix=suffix)
        qs = qs_v1 + qs_v2 + qs_v3 + qs_v4
        qs_remove = []
    if suffix is not None:
        qs = [_rewrite_to_focus_on_end(q, suffix) for q in qs]

    qs_added = sorted(list(set(qs) - set(qs_remove)))
    if full:
        # be careful to always add things in the right order!!!!
        return qs_remove + qs_added
    return qs_added


def get_question_num(question_version):
    if '-' in question_version:
        return int(question_version.split('-')[0][1:])
    else:
        return int(question_version[1])


if __name__ == "__main__":
    print('v1 has', len(get_questions(version='v1')), 'questions')
    print('v2 adds', len(get_questions(version='v2')), 'questions')
    print('v3 adds', len(get_questions(version='v3')), 'questions')
    print('v4 adds', len(get_questions(version='v4')), 'questions')
    print('total questions', len(get_questions('all')))
    for q in get_questions('v4-end'):
        print(q)
