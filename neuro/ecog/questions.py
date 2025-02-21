import re
from pprint import pprint

PROMPT_O1_DEC26 = '''List 100 semantic questions to ask about text spoken in a movie that may help predict ECOG responses when listening to that word. Make all questions have yes or no answers.

Examples:
Does the input mention a number?
Does the input mention a location?
Does the input contain a negation?
Does the input describe a physical action?
Does the input contain a measurement?'''

RESPONSE_RAW_01_DEC26 = '''1. **Does the text mention a number?**  
2. **Does the text refer to a specific time of day (e.g., morning, afternoon, evening)?**  
3. **Does the text mention a location (e.g., city, country, building)?**  
4. **Does the text contain a negation (e.g., “no,” “not,” “never”)?**  
5. **Does the text describe a physical action (e.g., “run,” “jump,” “eat”)?**  
6. **Does the text contain a measurement (e.g., weight, distance, height)?**  
7. **Does the text include any ordinal numbers (e.g., “first,” “second”)?**  
8. **Does the text contain a reference to money or cost?**  
9. **Does the text mention a family relationship (e.g., mother, father, cousin)?**  
10. **Does the text reference a person’s name (proper noun)?**  
11. **Does the text mention an animal?**  
12. **Does the text mention a color?**  
13. **Does the text include a comparison (e.g., “bigger than,” “smaller than”)?**  
14. **Does the text contain an emotion word (e.g., “happy,” “sad,” “angry”)?**  
15. **Does the text include a direct address (e.g., “Hey, John,” “Listen, Sarah”)?**  
16. **Does the text describe weather or climate (e.g., “It’s raining,” “Sunny day”)?**  
17. **Does the text reference a date, year, or era (e.g., “in 1999,” “the 18th century”)?**  
18. **Does the text contain an exclamation (e.g., “Wow!,” “Oh my God!”)?**  
19. **Does the text mention a mode of transportation (e.g., car, train, plane)?**  
20. **Does the text mention a brand name or product?**  
21. **Does the text include a question (i.e., is it phrased as a question)?**  
22. **Does the text mention a body part (e.g., hand, head, leg)?**  
23. **Does the text mention a sense perception (e.g., seeing, hearing, smelling)?**  
24. **Does the text include a quantifier (e.g., “many,” “few,” “some”)?**  
25. **Does the text make a request or command (imperative form)?**  
26. **Does the text mention any food or drink?**  
27. **Does the text reference a feeling of uncertainty (e.g., “unsure,” “doubtful,” “maybe”)?**  
28. **Does the text mention future intentions (e.g., “will,” “going to”)?**  
29. **Does the text reference the past (e.g., “used to,” “once,” “back then”)?**  
30. **Does the text contain a superlative (e.g., “greatest,” “smallest,” “fastest”)?**  
31. **Does the text name a specific event (e.g., “birthday,” “wedding,” “conference”)?**  
32. **Does the text reference a fictional element (e.g., mythical creature, fictional place)?**  
33. **Does the text include a promise or vow (e.g., “I swear,” “I promise”)?**  
34. **Does the text include a threat?**  
35. **Does the text contain a greeting (e.g., “Hello,” “Hi,” “Good morning”)?**  
36. **Does the text mention a problem or conflict?**  
37. **Does the text include any form of apology (e.g., “I’m sorry,” “My apologies”)?**  
38. **Does the text contain slang or colloquial expressions?**  
39. **Does the text mention a role or job title (e.g., “doctor,” “teacher,” “police officer”)?**  
40. **Does the text reference a historical figure or event (e.g., “World War II,” “Abraham Lincoln”)?**  
41. **Does the text describe a physical characteristic (e.g., “tall,” “short,” “bright”)?**  
42. **Does the text reference any sport or game (e.g., “football,” “chess”)?**  
43. **Does the text mention technology or devices (e.g., “phone,” “computer”)?**  
44. **Does the text indicate possession (e.g., “my,” “your,” “their”)?**  
45. **Does the text mention a personal opinion or belief (e.g., “I think,” “I believe”)?**  
46. **Does the text include any rhetorical questions (e.g., “Who wouldn’t want that?”)?**  
47. **Does the text reference music or art (e.g., “song,” “painting,” “concert”)?**  
48. **Does the text describe an obligation or necessity (e.g., “must,” “need to”)?**  
49. **Does the text mention a hypothetical situation (e.g., “if,” “in case”)?**  
50. **Does the text reference specific shapes (e.g., circle, square)?**  
51. **Does the text mention a weapon or tool (e.g., “gun,” “knife,” “hammer”)?**  
52. **Does the text contain figurative language (e.g., metaphor, simile)?**  
53. **Does the text express gratitude (e.g., “Thank you,” “I appreciate it”)?**  
54. **Does the text mention a deadline or time limit?**  
55. **Does the text describe a specific physical sensation (e.g., “pain,” “tingling”)?**  
56. **Does the text mention reading or writing activities?**  
57. **Does the text reference a list or enumeration (e.g., “There are three reasons…”)?**  
58. **Does the text ask for clarification (e.g., “What do you mean?”)?**  
59. **Does the text use any intensifiers (e.g., “very,” “extremely,” “highly”)?**  
60. **Does the text mention any form of entertainment (e.g., movie, theater, game)?**  
61. **Does the text express agreement or alignment (e.g., “I agree,” “exactly”)?**  
62. **Does the text mention a mistake or error?**  
63. **Does the text refer to personal preference (e.g., “I like,” “I prefer,” “I love”)?**  
64. **Does the text describe an event that happened recently (“just now,” “recently”)?**  
65. **Does the text mention a plan or strategy (e.g., “Let’s do this,” “We should…“)?**  
66. **Does the text include a description of movement in space (e.g., “enter,” “exit,” “approach”)?**  
67. **Does the text mention a change of state (e.g., “became,” “turned into,” “transformed”)?**  
68. **Does the text mention a conditional outcome (e.g., “if this happens, then…”)?**  
69. **Does the text have a sarcastic or ironic statement?**  
70. **Does the text mention a conflict between two or more people?**  
71. **Does the text reference health or illness?**  
72. **Does the text include any moral or ethical judgment (e.g., “That’s wrong,” “That’s right”)?**  
73. **Does the text describe a learned skill or ability (e.g., “I can drive,” “I can code”)?**  
74. **Does the text mention any legal terms (e.g., “contract,” “law,” “sue”)?**  
75. **Does the text include profanity or swear words?**  
76. **Does the text mention a specific brand of technology or software?**  
77. **Does the text reference an upcoming event (e.g., “next week,” “tomorrow night”)?**  
78. **Does the text include an idiom (e.g., “break a leg,” “hit the sack”)?**  
79. **Does the text explicitly reference fear or anxiety (e.g., “I’m scared,” “I’m worried”)?**  
80. **Does the text reference hope or optimism (e.g., “hopefully,” “I hope”)?**  
81. **Does the text describe or imply deception (e.g., “lying,” “tricking,” “faking”)?**  
82. **Does the text mention an academic or intellectual topic (e.g., “science,” “philosophy”)?**  
83. **Does the text reference an internal thought process (e.g., “I wondered,” “I realized”)?**  
84. **Does the text mention a personal achievement or success?**  
85. **Does the text mention a setback or failure (e.g., “we lost,” “it failed”)?**  
86. **Does the text use any formal or polite language (e.g., “Sir,” “Madam,” “please”)?**  
87. **Does the text reference a non-physical concept (e.g., “love,” “justice,” “freedom”)?**  
88. **Does the text mention any type of group or organization (e.g., “company,” “team,” “committee”)?**  
89. **Does the text reference gossip or rumor (e.g., “I heard that…,” “they say…”)?**  
90. **Does the text mention a timeline or schedule (e.g., “at 3 pm,” “in two days”)?**  
91. **Does the text reference a memory or recall (e.g., “I remember,” “I recall”)?**  
92. **Does the text contain an allusion to literature or famous quotes?**  
93. **Does the text describe a property or characteristic of an object (e.g., “sharp,” “heavy,” “rusty”)?**  
94. **Does the text include a direct comparison using “like” or “as” (simile)?**  
95. **Does the text depict a cause-effect relationship (e.g., “because,” “therefore”)?**  
96. **Does the text reference a schedule conflict or time clash (e.g., “I’m busy then”)?**  
97. **Does the text mention any type of resource scarcity (e.g., “low on food,” “short on time”)?**  
98. **Does the text suggest cooperation or collaboration (e.g., “Let’s work together,” “We can do this together”)?**  
99. **Does the text reference an injury or harm (physical or emotional)?**  
100. **Does the text mention an alternative option or choice (e.g., “Alternatively,” “Another way”)?**'''

QS_O1_DEC26 = [
    'Does the text include a comparison?',
    'Does the text describe a physical action?',
    'Does the text reference a fictional element?',
    'Does the text indicate possession?',
    'Does the text mention reading or writing activities?',
    'Does the text contain figurative language?',
    'Does the text mention a mode of transportation?',
    'Does the text mention a conditional outcome?',
    'Does the text mention a problem or conflict?',
    'Does the text name a specific event?',
    'Does the text refer to a specific time of day?',
    'Does the text reference a list or enumeration?',
    'Does the text contain an emotion word?',
    'Does the text reference a feeling of uncertainty?',
    'Does the text mention a color?',
    'Does the text reference a person’s name?',
    'Does the text contain a greeting?',
    'Does the text describe weather or climate?',
    'Does the text contain a measurement?',
    'Does the text reference a memory or recall?',
    'Does the text reference an injury or harm?',
    'Does the text refer to personal preference?',
    'Does the text include any form of apology?',
    'Does the text reference music or art?',
    'Does the text describe an event that happened recently?',
    'Does the text mention a setback or failure?',
    'Does the text describe a property or characteristic of an object?',
    'Does the text mention any legal terms?',
    'Does the text include any ordinal numbers?',
    'Does the text mention a mistake or error?',
    'Does the text reference a schedule conflict or time clash?',
    'Does the text mention a personal opinion or belief?',
    'Does the text use any formal or polite language?',
    'Does the text ask for clarification?',
    'Does the text mention a role or job title?',
    'Does the text contain slang or colloquial expressions?',
    'Does the text reference the past?',
    'Does the text include profanity or swear words?',
    'Does the text include a quantifier?',
    'Does the text include a question?',
    'Does the text mention an academic or intellectual topic?',
    'Does the text contain an exclamation?',
    'Does the text mention a personal achievement or success?',
    'Does the text mention a location?',
    'Does the text contain a reference to money or cost?',
    'Does the text depict a cause-effect relationship?',
    'Does the text describe an obligation or necessity?',
    'Does the text mention a timeline or schedule?',
    'Does the text describe a physical characteristic?',
    'Does the text contain a negation?',
    'Does the text mention technology or devices?',
    'Does the text mention a hypothetical situation?',
    'Does the text mention a specific brand of technology or software?',
    'Does the text include a direct comparison using “like” or “as”?',
    'Does the text include a description of movement in space?',
    'Does the text express gratitude?',
    'Does the text have a sarcastic or ironic statement?',
    'Does the text reference gossip or rumor?',
    'Does the text explicitly reference fear or anxiety?',
    'Does the text describe or imply deception?',
    'Does the text include a direct address?',
    'Does the text mention a deadline or time limit?',
    'Does the text mention a number?',
    'Does the text mention future intentions?',
    'Does the text mention any type of resource scarcity?',
    'Does the text mention any food or drink?',
    'Does the text mention a plan or strategy?',
    'Does the text reference any sport or game?',
    'Does the text use any intensifiers?',
    'Does the text express agreement or alignment?',
    'Does the text reference specific shapes?',
    'Does the text mention a sense perception?',
    'Does the text include any rhetorical questions?',
    'Does the text describe a learned skill or ability?',
    'Does the text mention a conflict between two or more people?',
    'Does the text reference an upcoming event?',
    'Does the text mention any type of group or organization?',
    'Does the text reference a non-physical concept?',
    'Does the text include a threat?',
    'Does the text mention any form of entertainment?',
    'Does the text reference health or illness?',
    'Does the text make a request or command?',
    'Does the text include a promise or vow?',
    'Does the text mention a body part?',
    'Does the text mention a brand name or product?',
    'Does the text mention an animal?',
    'Does the text contain a superlative?',
    'Does the text include an idiom?',
    'Does the text include any moral or ethical judgment?',
    'Does the text reference a date, year, or era?',
    'Does the text mention an alternative option or choice?',
    'Does the text reference an internal thought process?',
    'Does the text contain an allusion to literature or famous quotes?',
    'Does the text reference a historical figure or event?',
    'Does the text mention a weapon or tool?',
    'Does the text mention a change of state?',
    'Does the text describe a specific physical sensation?',
    'Does the text reference hope or optimism?',
    'Does the text mention a family relationship?',
    'Does the text suggest cooperation or collaboration?']

PROMPT_O1_DEC26_2 = '''List another 100 new and non-overlapping questions about semantics '''

RESPONSE_RAW_01_DEC26_2 = '''1. **Does the text mention or imply flirting or romantic advances?**  
2. **Does the text mention or describe jealousy or envy between characters?**  
3. **Does the text suggest that someone is ignoring or avoiding another person?**  
4. **Does the text convey a sense of rivalry between two or more characters?**  
5. **Does the text describe or imply emotional support given to someone?**  
6. **Does the text mention the idea of trust or distrust between characters?**  
7. **Does the text mention a surprise intended for someone else (e.g., a surprise party)?**  
8. **Does the text include direct criticism or complaint about another person?**  
9. **Does the text mention conflict resolution or making peace after a disagreement?**  
10. **Does the text refer to someone’s personal or emotional boundaries?**  
11. **Does the text allude to a coded message or hidden meaning (beyond simple deception)?**  
12. **Does the text mention or describe a misunderstanding or miscommunication?**  
13. **Does the text explicitly describe how something is said (e.g., whispered, shouted)?**  
14. **Does the text reference a communication barrier (e.g., language differences)?**  
15. **Does the text use an onomatopoeic word (e.g., “bang,” “sizzle,” “whoosh”)?**  
16. **Does the text contain a trailing off in speech (e.g., “I just… can’t do it”)?**  
17. **Does the text mention someone being bilingual or switching languages?**  
18. **Does the text describe the speaker’s volume or pitch (“speaking softly,” “yelling”)?**  
19. **Does the text mention directly quoting or repeating another person’s words verbatim?**  
20. **Does the text mention physical gestures as part of communication (e.g., nodding, waving)?**  
21. **Does the text mention a social tradition or cultural custom (e.g., a ritual)?**  
22. **Does the text reference or describe a taboo or something considered socially unacceptable?**  
23. **Does the text mention or describe a scenario requiring social etiquette (e.g., proper greetings)?**  
24. **Does the text describe a charitable or philanthropic act (e.g., donating, volunteering)?**  
25. **Does the text mention or describe a protest, demonstration, or public rally?**  
26. **Does the text address social status or class differences (e.g., “upper class,” “working class”)?**  
27. **Does the text mention or describe stereotypes or prejudices against a group?**  
28. **Does the text mention or describe a negotiation or attempt at compromise?**  
29. **Does the text mention someone being excluded or left out of a group activity?**  
30. **Does the text describe courtesy or politeness gestures (beyond just “thank you,” “please”)?**  
31. **Does the text describe the ambiance or atmosphere of a location (e.g., “tense,” “lively”)?**  
32. **Does the text mention or describe interior design or decoration (e.g., “walls covered in art”)?**  
33. **Does the text mention or describe the lighting conditions (e.g., “dimly lit,” “brightly lit”)?**  
34. **Does the text mention or describe background sounds or noise in the environment?**  
35. **Does the text specifically describe the temperature (e.g., “cold,” “hot”) but not the weather?**  
36. **Does the text mention the presence or absence of other people in the setting (“the room was empty”)?**  
37. **Does the text describe the location of an object within a scene (“on the table,” “under the chair”)?**  
38. **Does the text distinguish between being indoors vs outdoors?**  
39. **Does the text describe cleanliness or disarray of a place (e.g., “messy,” “spotless”)?**  
40. **Does the text mention a sense of belonging (or not) in that environment?**  
41. **Does the text mention or describe someone feeling bored or uninterested?**  
42. **Does the text describe regret or remorse (not just an apology)?**  
43. **Does the text mention or describe someone feeling lonely or isolated?**  
44. **Does the text describe confusion or being perplexed (e.g., “I’m so confused”)?**  
45. **Does the text reference curiosity or intrigue about a subject (“I’m curious about…”)?**  
46. **Does the text mention or describe admiration or awe (“I’m amazed,” “That’s incredible”)?**  
47. **Does the text describe relief or feeling relieved after tension (“Phew, that’s over!”)?**  
48. **Does the text mention or describe determination or resolve (“I won’t quit,” “I’m determined”)?**  
49. **Does the text mention or describe being distracted or having trouble focusing?**  
50. **Does the text describe indifference or apathy (“I don’t care at all”)?**  
51. **Does the text mention or describe a failed attempt at an action (“I tried but couldn’t”)?**  
52. **Does the text reference an accidental event (“I slipped,” “the vase broke by accident”)?**  
53. **Does the text mention or describe rescuing or saving someone from harm?**  
54. **Does the text mention or describe a chase or pursuit (e.g., “they ran after him”)?**  
55. **Does the text mention or describe hiding or concealment (“she hid behind the door”)?**  
56. **Does the text mention searching for something or someone (“we’ve been looking everywhere”)?**  
57. **Does the text mention blocking or shielding (“he blocked the punch”)?**  
58. **Does the text mention building or constructing something (“they built a shelter”)?**  
59. **Does the text mention destroying or breaking something intentionally?**  
60. **Does the text describe a repeated or habitual movement (“he paced back and forth”)?**  
61. **Does the text mention analyzing or evaluating a situation (“we assessed the risks”)?**  
62. **Does the text describe learning or discovering new information (“I just found out”)?**  
63. **Does the text describe teaching or instructing someone else (“I’ll show you how”)?**  
64. **Does the text mention rationalizing or justifying an action or belief (“I did it because…”)?**  
65. **Does the text reference something as common knowledge or widely known (“everyone knows that”)?**  
66. **Does the text describe differentiating or distinguishing between two similar things?**  
67. **Does the text mention a rumor being confirmed or disproven (“it turns out that…”)?**  
68. **Does the text describe training or practice to improve a skill (not just stating a skill)?**  
69. **Does the text mention analyzing data or statistics (e.g., “the numbers show…”)?**  
70. **Does the text describe brainstorming or generating ideas collectively?**  
71. **Does the text mention or describe thirst or dehydration (“I’m so thirsty”)?**  
72. **Does the text mention a body reflex (e.g., sneezing, coughing, blinking)?**  
73. **Does the text mention or describe aging or getting older (“I’m too old for this”)?**  
74. **Does the text refer to genetic traits or heredity (“it runs in my family”)?**  
75. **Does the text mention pregnancy or childbirth?**  
76. **Does the text reference physical exercise or a workout routine?**  
77. **Does the text describe physical fatigue or exhaustion (“I’m worn out”)?**  
78. **Does the text mention a medical procedure (e.g., surgery, therapy)?**  
79. **Does the text describe a phobia or intense, irrational fear (beyond basic anxiety)?**  
80. **Does the text reference physical stamina or endurance in some context?**  
81. **Does the text mention or discuss the idea of fate or destiny?**  
82. **Does the text mention or discuss chaos, randomness, or chance?**  
83. **Does the text refer to balance or equilibrium in an abstract sense?**  
84. **Does the text discuss existential or philosophical ideas about life’s meaning?**  
85. **Does the text reference duality?**  
86. **Does the text mention or discuss whether reality might be an illusion or not?**  
87. **Does the text mention or discuss something being inevitable or unavoidable?**  
88. **Does the text refer to responsibility as an abstract principle (“it’s my responsibility”)?**  
89. **Does the text mention or discuss an afterlife or spiritual realm (e.g., heaven, reincarnation)?**  
90. **Does the text mention or discuss free will or personal autonomy (“I can choose my own path”)?**  
91. **Does the text mention or reference gambling or betting (“I bet you,” “place your bets”)?**  
92. **Does the text mention or reference a legal punishment or penalty (“fined,” “sentenced”)?**  
93. **Does the text mention historical artifacts or antiques (not just events) (“an ancient vase”)?**  
94. **Does the text reference a man-made disaster (e.g., oil spill, industrial accident)?**  
95. **Does the text explicitly use or reference a pun or wordplay for humor?**  
96. **Does the text mention illusions, hallucinations, or imaginary experiences (“I saw something unreal”)?**  
97. **Does the text mention or reference making or receiving a phone call specifically?**  
98. **Does the text mention or reference a dream or daydream?**  
99. **Does the text mention or reference a test or exam (academic or otherwise)?**  
100. **Does the text mention or reference someone’s childhood memories or events?**'''

QS_O1_DEC26_2 = ['Does the text mention or describe a protest, demonstration, or public rally?',
                 'Does the text describe the ambiance or atmosphere of a location?',
                 'Does the text describe relief or feeling relieved after tension?',
                 'Does the text mention or describe admiration or awe?',
                 'Does the text mention historical artifacts or antiques ?',
                 'Does the text describe a repeated or habitual movement?',
                 'Does the text suggest that someone is ignoring or avoiding another person?',
                 'Does the text mention or describe jealousy or envy between characters?',
                 'Does the text reference a man-made disaster?',
                 'Does the text describe physical fatigue or exhaustion?',
                 'Does the text discuss existential or philosophical ideas about life’s '
                 'meaning?',
                 'Does the text mention physical gestures as part of communication?',
                 'Does the text mention a social tradition or cultural custom?',
                 'Does the text mention a surprise intended for someone else?',
                 'Does the text describe a charitable or philanthropic act?',
                 'Does the text describe or imply emotional support given to someone?',
                 'Does the text include direct criticism or complaint about another person?',
                 'Does the text refer to balance or equilibrium in an abstract sense?',
                 'Does the text reference something as common knowledge or widely known?',
                 'Does the text mention or discuss whether reality might be an illusion or '
                 'not?',
                 'Does the text mention or describe a scenario requiring social etiquette?',
                 'Does the text mention pregnancy or childbirth?',
                 'Does the text mention or describe stereotypes or prejudices against a group?',
                 'Does the text convey a sense of rivalry between two or more characters?',
                 'Does the text reference an accidental event?',
                 'Does the text mention the idea of trust or distrust between characters?',
                 'Does the text mention analyzing data or statistics?',
                 'Does the text describe a phobia or intense, irrational fear?',
                 'Does the text mention or discuss chaos, randomness, or chance?',
                 'Does the text mention rationalizing or justifying an action or belief?',
                 'Does the text explicitly use or reference a pun or wordplay for humor?',
                 'Does the text mention the presence or absence of other people in the '
                 'setting?',
                 'Does the text reference a communication barrier?',
                 'Does the text allude to a coded message or hidden meaning?',
                 'Does the text mention or imply flirting or romantic advances?',
                 'Does the text mention or describe the lighting conditions?',
                 'Does the text mention or describe a failed attempt at an action?',
                 'Does the text mention or describe a negotiation or attempt at compromise?',
                 'Does the text mention or describe a chase or pursuit?',
                 'Does the text mention or discuss the idea of fate or destiny?',
                 'Does the text mention or reference gambling or betting?',
                 'Does the text mention conflict resolution or making peace after a '
                 'disagreement?',
                 'Does the text mention illusions, hallucinations, or imaginary experiences?',
                 'Does the text reference physical stamina or endurance in some context?',
                 'Does the text refer to responsibility as an abstract principle?',
                 'Does the text describe courtesy or politeness gestures?',
                 'Does the text describe confusion or being perplexed?',
                 'Does the text mention or describe thirst or dehydration?',
                 'Does the text reference duality?',
                 'Does the text describe training or practice to improve a skill?',
                 'Does the text mention someone being excluded or left out of a group '
                 'activity?',
                 'Does the text mention blocking or shielding?',
                 'Does the text mention or discuss free will or personal autonomy?',
                 'Does the text mention or reference a legal punishment or penalty?',
                 'Does the text mention or describe being distracted or having trouble '
                 'focusing?',
                 'Does the text refer to genetic traits or heredity?',
                 'Does the text mention or reference a test or exam?',
                 'Does the text mention destroying or breaking something intentionally?',
                 'Does the text distinguish between being indoors vs outdoors?',
                 'Does the text describe the speaker’s volume or pitch?',
                 'Does the text describe indifference or apathy?',
                 'Does the text mention building or constructing something?',
                 'Does the text use an onomatopoeic word?',
                 'Does the text describe cleanliness or disarray of a place?',
                 'Does the text mention or describe someone feeling lonely or isolated?',
                 'Does the text reference or describe a taboo or something considered socially '
                 'unacceptable?',
                 'Does the text contain a trailing off in speech?',
                 'Does the text describe learning or discovering new information?',
                 'Does the text refer to someone’s personal or emotional boundaries?',
                 'Does the text describe differentiating or distinguishing between two similar '
                 'things?',
                 'Does the text mention a sense of belonging  in that environment?',
                 'Does the text mention or discuss an afterlife or spiritual realm?',
                 'Does the text mention or describe someone feeling bored or uninterested?',
                 'Does the text describe the location of an object within a scene?',
                 'Does the text reference curiosity or intrigue about a subject?',
                 'Does the text describe teaching or instructing someone else?',
                 'Does the text mention directly quoting or repeating another person’s words '
                 'verbatim?',
                 'Does the text mention or describe aging or getting older?',
                 'Does the text mention or reference making or receiving a phone call '
                 'specifically?',
                 'Does the text mention or reference a dream or daydream?',
                 'Does the text mention or describe interior design or decoration?',
                 'Does the text mention searching for something or someone?',
                 'Does the text mention or describe hiding or concealment?',
                 'Does the text describe regret or remorse?',
                 'Does the text mention someone being bilingual or switching languages?',
                 'Does the text mention or describe background sounds or noise in the '
                 'environment?',
                 'Does the text explicitly describe how something is said?',
                 'Does the text mention or discuss something being inevitable or unavoidable?',
                 'Does the text mention or describe rescuing or saving someone from harm?',
                 'Does the text specifically describe the temperature  but not the weather?',
                 'Does the text mention or describe a misunderstanding or miscommunication?',
                 'Does the text mention or describe determination or resolve?',
                 'Does the text describe brainstorming or generating ideas collectively?',
                 'Does the text mention a body reflex?',
                 'Does the text mention analyzing or evaluating a situation?',
                 'Does the text mention a medical procedure?',
                 'Does the text address social status or class differences?',
                 'Does the text mention a rumor being confirmed or disproven?',
                 'Does the text reference physical exercise or a workout routine?',
                 'Does the text mention or reference someone’s childhood memories or events?']


def clean_response(RESPONSE):

    qs = RESPONSE.split('\n')
    qs_clean = []

    for q in qs:
        # remove leading number
        q = q.split('. ')[1]

        # remove asterisks
        q = q.replace('*', '')

        # remove parenthetical statements
        q = re.sub(r'\([^)]*\)', '', q)

        # remove leading/trailing whitespace
        q = q.strip()

        # remove space before question mark
        q = q.replace(' ?', '?')

        print(q)

        assert q.endswith('?')

        qs_clean.append(q)

    qs_clean = list(set(qs_clean))
    return qs_clean


if __name__ == '__main__':
    qs_1 = clean_response(RESPONSE_RAW_01_DEC26)
    qs_2 = clean_response(RESPONSE_RAW_01_DEC26_2)
    print(len(qs_1), len(qs_2), len(set(qs_1 + qs_2)))
