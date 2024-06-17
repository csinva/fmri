
import warnings

TRAIN_01_PUBLIC = ['adollshouse', 'gpsformylostidentity', 'singlewomanseekingmanwich', 'adventuresinsayingyes', 'hangtime', 'sloth', 'afatherscover', 'haveyoumethimyet', 'souls', 'againstthewind', 'howtodraw', 'stagefright', 'alternateithicatom', 'ifthishaircouldtalk', 'stumblinginthedark', 'avatar', 'inamoment', 'superheroesjustforeachother', 'backsideofthestorm', 'itsabox', 'sweetaspie', 'becomingindian', 'jugglingandjesus', 'swimmingwithastronauts', 'beneaththemushroomcloud', 'kiksuya', 'thatthingonmyarm', 'birthofanation', 'leavingbaghdad', 'theadvancedbeginner', 'bluehope', 'legacy', 'theclosetthatateeverything', 'breakingupintheageofgoogle', 'lifeanddeathontheoregontrail', 'thecurse', 'buck', 'life', 'thefreedomridersandme', 'catfishingstrangerstofindmyself',
                   'lifereimagined', 'theinterview', 'cautioneating', 'listo', 'thepostmanalwayscalls', 'christmas1940', 'mayorofthefreaks', 'theshower', 'cocoonoflove', 'metsmagic', 'thetiniestbouquet', 'comingofageondeathrow', 'mybackseatviewofagreatromance', 'thetriangleshirtwaistconnection', 'exorcism', 'myfathershands', 'threemonths', 'eyespy', 'myfirstdaywiththeyankees', 'thumbsup', 'firetestforlove', 'naked', 'tildeath', 'food', 'notontheusualtour', 'treasureisland', 'forgettingfear', 'odetostepfather', 'undertheinfluence', 'onlyonewaytofindout', 'vixenandtheussr', 'gangstersandcookies', 'penpal', 'waitingtogo', 'goingthelibertyway', 'quietfire', 'whenmothersbullyback', 'goldiethegoldfish', 'reachingoutbetweenthebars', 'golfclubbing', 'shoppinginchina', 'wildwomenanddancingqueens']
TRAIN_02_PUBLIC = ['adollshouse', 'hangtime', 'sloth', 'adventuresinsayingyes', 'haveyoumethimyet', 'souls', 'afatherscover', 'howtodraw', 'stagefright', 'againstthewind', 'ifthishaircouldtalk', 'stumblinginthedark', 'alternateithicatom', 'inamoment', 'superheroesjustforeachother', 'avatar', 'itsabox', 'sweetaspie', 'backsideofthestorm', 'jugglingandjesus', 'swimmingwithastronauts', 'becomingindian', 'kiksuya', 'thatthingonmyarm', 'beneaththemushroomcloud', 'leavingbaghdad', 'theadvancedbeginner', 'birthofanation', 'legacy', 'theclosetthatateeverything', 'breakingupintheageofgoogle', 'lifeanddeathontheoregontrail', 'thecurse', 'buck', 'life', 'thefreedomridersandme', 'catfishingstrangerstofindmyself', 'lifereimagined', 'theinterview', 'cautioneating', 'listo',
                   'thepostmanalwayscalls', 'christmas1940', 'mayorofthefreaks', 'theshower', 'cocoonoflove', 'metsmagic', 'thetiniestbouquet', 'comingofageondeathrow', 'mybackseatviewofagreatromance', 'thetriangleshirtwaistconnection', 'exorcism', 'myfathershands', 'threemonths', 'eyespy', 'myfirstdaywiththeyankees', 'thumbsup', 'firetestforlove', 'naked', 'tildeath', 'food', 'notontheusualtour', 'treasureisland', 'forgettingfear', 'odetostepfather', 'undertheinfluence', 'onlyonewaytofindout', 'vixenandtheussr', 'gangstersandcookies', 'penpal', 'waitingtogo', 'goingthelibertyway', 'quietfire', 'whenmothersbullyback', 'goldiethegoldfish', 'reachingoutbetweenthebars', 'golfclubbing', 'shoppinginchina', 'wildwomenanddancingqueens']
TRAIN_03_PUBLIC = [
    'adollshouse', 'gpsformylostidentity', 'singlewomanseekingmanwich', 'adventuresinsayingyes', 'hangtime', 'sloth', 'afatherscover', 'haveyoumethimyet', 'souls', 'againstthewind', 'howtodraw', 'stagefright', 'alternateithicatom', 'ifthishaircouldtalk', 'stumblinginthedark', 'avatar', 'inamoment', 'superheroesjustforeachother', 'backsideofthestorm', 'itsabox', 'sweetaspie', 'becomingindian', 'jugglingandjesus', 'swimmingwithastronauts', 'beneaththemushroomcloud', 'kiksuya', 'thatthingonmyarm', 'birthofanation', 'leavingbaghdad', 'theadvancedbeginner', 'bluehope', 'legacy', 'theclosetthatateeverything', 'breakingupintheageofgoogle', 'lifeanddeathontheoregontrail', 'thecurse', 'buck', 'life', 'thefreedomridersandme', 'catfishingstrangerstofindmyself', 'lifereimagined', 'theinterview',
    'cautioneating', 'listo', 'thepostmanalwayscalls', 'christmas1940', 'mayorofthefreaks', 'theshower', 'cocoonoflove', 'metsmagic', 'thetiniestbouquet', 'comingofageondeathrow', 'mybackseatviewofagreatromance', 'thetriangleshirtwaistconnection', 'exorcism', 'myfathershands', 'threemonths', 'eyespy', 'myfirstdaywiththeyankees', 'thumbsup', 'firetestforlove', 'naked', 'tildeath', 'food', 'notontheusualtour', 'treasureisland', 'forgettingfear', 'odetostepfather', 'undertheinfluence', 'onlyonewaytofindout', 'vixenandtheussr', 'gangstersandcookies', 'penpal', 'waitingtogo', 'goingthelibertyway', 'quietfire', 'whenmothersbullyback', 'goldiethegoldfish', 'reachingoutbetweenthebars', 'golfclubbing', 'shoppinginchina', 'wildwomenanddancingqueens'
]
TRAIN_04_PUBLIC = ['adollshouse', 'adventuresinsayingyes', 'alternateithicatom', 'avatar', 'buck', 'exorcism', 'eyespy', 'hangtime', 'haveyoumethimyet', 'howtodraw', 'inamoment', 'itsabox', 'legacy',
                   'myfirstdaywiththeyankees', 'naked', 'odetostepfather', 'sloth', 'souls', 'stagefright', 'swimmingwithastronauts', 'thatthingonmyarm', 'theclosetthatateeverything', 'tildeath', 'undertheinfluence']
TRAIN_05_PUBLIC = ['adollshouse', 'adventuresinsayingyes', 'alternateithicatom', 'avatar', 'buck', 'exorcism', 'eyespy', 'hangtime', 'haveyoumethimyet', 'howtodraw', 'inamoment', 'itsabox', 'legacy',
                   'life', 'myfirstdaywiththeyankees', 'naked', 'odetostepfather', 'sloth', 'souls', 'stagefright', 'swimmingwithastronauts', 'thatthingonmyarm', 'theclosetthatateeverything', 'tildeath', 'undertheinfluence']
TRAIN_06_PUBLIC = TRAIN_05_PUBLIC
TRAIN_07_PUBLIC = TRAIN_05_PUBLIC
TRAIN_08_PUBLIC = TRAIN_05_PUBLIC
TEST_PUBLIC = [
    "wheretheressmoke", "fromboyhoodtofatherhood"
]
TRAIN_01_HUGE = ['itsabox', 'odetostepfather', 'inamoment', 'hangtime', 'ifthishaircouldtalk', 'goingthelibertyway', 'golfclubbing', 'thetriangleshirtwaistconnection', 'igrewupinthewestborobaptistchurch', 'tetris', 'becomingindian', 'canplanetearthfeedtenbillionpeoplepart1', 'thetiniestbouquet', 'swimmingwithastronauts', 'lifereimagined', 'forgettingfear', 'stumblinginthedark', 'backsideofthestorm', 'food', 'theclosetthatateeverything', 'notontheusualtour', 'exorcism', 'adventuresinsayingyes', 'thefreedomridersandme', 'cocoonoflove', 'waitingtogo', 'thepostmanalwayscalls', 'googlingstrangersandkentuckybluegrass', 'mayorofthefreaks', 'learninghumanityfromdogs', 'shoppinginchina', 'souls', 'cautioneating', 'comingofageondeathrow', 'breakingupintheageofgoogle', 'gpsformylostidentity', 'eyespy', 'treasureisland', 'thesurprisingthingilearnedsailingsoloaroundtheworld', 'theadvancedbeginner', 'goldiethegoldfish', 'life', 'thumbsup', 'seedpotatoesofleningrad',
                 'theshower', 'adollshouse', 'canplanetearthfeedtenbillionpeoplepart2', 'sloth', 'howtodraw', 'quietfire', 'metsmagic', 'penpal', 'thecurse', 'canadageeseandddp', 'thatthingonmyarm', 'buck', 'wildwomenanddancingqueens', 'againstthewind', 'indianapolis', 'alternateithicatom', 'bluehope', 'kiksuya', 'afatherscover', 'haveyoumethimyet', 'firetestforlove', 'catfishingstrangerstofindmyself', 'christmas1940', 'tildeath', 'lifeanddeathontheoregontrail', 'vixenandtheussr', 'undertheinfluence', 'beneaththemushroomcloud', 'jugglingandjesus', 'superheroesjustforeachother', 'sweetaspie', 'naked', 'singlewomanseekingmanwich', 'avatar', 'whenmothersbullyback', 'myfathershands', 'reachingoutbetweenthebars', 'theinterview', 'stagefright', 'legacy', 'canplanetearthfeedtenbillionpeoplepart3', 'listo', 'gangstersandcookies', 'birthofanation', 'mybackseatviewofagreatromance', 'lawsthatchokecreativity', 'threemonths', 'whyimustspeakoutaboutclimatechange', 'leavingbaghdad']
TRAIN_02_HUGE = ['itsabox', 'odetostepfather', 'inamoment', 'afearstrippedbare', 'findingmyownrescuer', 'hangtime', 'ifthishaircouldtalk', 'goingthelibertyway', 'golfclubbing', 'thetriangleshirtwaistconnection', 'igrewupinthewestborobaptistchurch', 'tetris', 'becomingindian', 'canplanetearthfeedtenbillionpeoplepart1', 'thetiniestbouquet', 'swimmingwithastronauts', 'lifereimagined', 'forgettingfear', 'stumblinginthedark', 'backsideofthestorm', 'food', 'theclosetthatateeverything', 'escapingfromadirediagnosis', 'notontheusualtour', 'exorcism', 'adventuresinsayingyes', 'thefreedomridersandme', 'cocoonoflove', 'waitingtogo', 'thepostmanalwayscalls', 'googlingstrangersandkentuckybluegrass', 'mayorofthefreaks', 'learninghumanityfromdogs', 'shoppinginchina', 'souls', 'cautioneating', 'comingofageondeathrow', 'breakingupintheageofgoogle', 'gpsformylostidentity', 'marryamanwholoveshismother', 'eyespy', 'treasureisland', 'thesurprisingthingilearnedsailingsoloaroundtheworld', 'theadvancedbeginner', 'goldiethegoldfish', 'life',
                 'thumbsup', 'seedpotatoesofleningrad', 'theshower', 'adollshouse', 'canplanetearthfeedtenbillionpeoplepart2', 'sloth', 'howtodraw', 'quietfire', 'metsmagic', 'penpal', 'thecurse', 'canadageeseandddp', 'thatthingonmyarm', 'buck', 'thesecrettomarriage', 'wildwomenanddancingqueens', 'againstthewind', 'indianapolis', 'alternateithicatom', 'bluehope', 'kiksuya', 'afatherscover', 'haveyoumethimyet', 'firetestforlove', 'catfishingstrangerstofindmyself', 'christmas1940', 'tildeath', 'lifeanddeathontheoregontrail', 'vixenandtheussr', 'undertheinfluence', 'beneaththemushroomcloud', 'jugglingandjesus', 'superheroesjustforeachother', 'sweetaspie', 'naked', 'singlewomanseekingmanwich', 'avatar', 'whenmothersbullyback', 'myfathershands', 'reachingoutbetweenthebars', 'theinterview', 'stagefright', 'legacy', 'canplanetearthfeedtenbillionpeoplepart3', 'listo', 'gangstersandcookies', 'birthofanation', 'mybackseatviewofagreatromance', 'lawsthatchokecreativity', 'threemonths', 'whyimustspeakoutaboutclimatechange', 'leavingbaghdad']
TRAIN_03_HUGE = ['itsabox', 'odetostepfather', 'inamoment', 'afearstrippedbare', 'findingmyownrescuer', 'hangtime', 'ifthishaircouldtalk', 'goingthelibertyway', 'golfclubbing', 'thetriangleshirtwaistconnection', 'igrewupinthewestborobaptistchurch', 'tetris', 'becomingindian', 'canplanetearthfeedtenbillionpeoplepart1', 'thetiniestbouquet', 'swimmingwithastronauts', 'lifereimagined', 'forgettingfear', 'stumblinginthedark', 'backsideofthestorm', 'food', 'theclosetthatateeverything', 'escapingfromadirediagnosis', 'notontheusualtour', 'exorcism', 'adventuresinsayingyes', 'thefreedomridersandme', 'cocoonoflove', 'waitingtogo', 'thepostmanalwayscalls', 'googlingstrangersandkentuckybluegrass', 'mayorofthefreaks', 'learninghumanityfromdogs', 'shoppinginchina', 'souls', 'cautioneating', 'comingofageondeathrow', 'breakingupintheageofgoogle', 'gpsformylostidentity', 'marryamanwholoveshismother', 'eyespy', 'treasureisland', 'thesurprisingthingilearnedsailingsoloaroundtheworld', 'theadvancedbeginner', 'goldiethegoldfish', 'life',
                 'thumbsup', 'seedpotatoesofleningrad', 'theshower', 'adollshouse', 'canplanetearthfeedtenbillionpeoplepart2', 'sloth', 'howtodraw', 'quietfire', 'metsmagic', 'penpal', 'thecurse', 'canadageeseandddp', 'thatthingonmyarm', 'buck', 'thesecrettomarriage', 'wildwomenanddancingqueens', 'againstthewind', 'indianapolis', 'alternateithicatom', 'bluehope', 'kiksuya', 'afatherscover', 'haveyoumethimyet', 'firetestforlove', 'catfishingstrangerstofindmyself', 'christmas1940', 'tildeath', 'lifeanddeathontheoregontrail', 'vixenandtheussr', 'undertheinfluence', 'beneaththemushroomcloud', 'jugglingandjesus', 'superheroesjustforeachother', 'sweetaspie', 'naked', 'singlewomanseekingmanwich', 'avatar', 'whenmothersbullyback', 'myfathershands', 'reachingoutbetweenthebars', 'theinterview', 'stagefright', 'legacy', 'canplanetearthfeedtenbillionpeoplepart3', 'listo', 'gangstersandcookies', 'birthofanation', 'mybackseatviewofagreatromance', 'lawsthatchokecreativity', 'threemonths', 'whyimustspeakoutaboutclimatechange', 'leavingbaghdad']

TEST_HUGE = [
    "wheretheressmoke", "fromboyhoodtofatherhood", "onapproachtopluto"
]
DICT_PUBLIC = {
    'train': {
        "UTS01": TRAIN_01_PUBLIC,
        "UTS02": TRAIN_02_PUBLIC,
        "UTS03": TRAIN_03_PUBLIC,
        'UTS04': TRAIN_04_PUBLIC,
        'UTS05': TRAIN_05_PUBLIC,
        'UTS06': TRAIN_06_PUBLIC,
        'UTS07': TRAIN_07_PUBLIC,
        'UTS08': TRAIN_08_PUBLIC,
        'shared': list(set(TRAIN_01_PUBLIC).intersection(TRAIN_02_PUBLIC, TRAIN_03_PUBLIC))
    },
    'test': {
        k: TEST_PUBLIC
        for k in ['UTS01', 'UTS02', 'UTS03', 'UTS04', 'UTS05', 'UTS06', 'UTS07', 'UTS08', 'shared']
    }
}
DICT_HUGE = {
    'train': {
        'UTS01': TRAIN_01_HUGE,
        'UTS02': TRAIN_02_HUGE,
        'UTS03': TRAIN_03_HUGE,
        'shared': list(set(TRAIN_01_HUGE).intersection(TRAIN_02_HUGE, TRAIN_03_HUGE)),
    },
    'test': {
        "UTS01": TEST_HUGE,
        "UTS02": TEST_HUGE,
        "UTS03": TEST_HUGE,
        'shared': TEST_HUGE
    }
}

TEST_BRAINDRIVE = {
    'UTS02': ['GenStory1', 'GenStory2', 'GenStory3', 'GenStory4', 'GenStory5',
              'GenStory6', 'GenStory7', 'GenStory8', 'GenStory9', 'GenStory10',
              'GenStory27', 'GenStory28', 'GenStory29'],
    'UTS03': ['GenStory12', 'GenStory13', 'GenStory14', 'GenStory15',
              'GenStory16', 'GenStory17'],
}


def get_story_names(subject: str = "UTS01", train_or_test="train", use_huge=False, use_brain_drive=False, all=False):

    if use_brain_drive:
        warnings.warn('Using BrainDrive stories')
        if all:
            return sum(TEST_BRAINDRIVE.values(), [])

        story_names = TEST_BRAINDRIVE[subject]
        return story_names

    if all:
        warnings.warn('Loading all stories, ignoring subject / train_or_test')
        # get set of all stories
        all_stories = []
        for k in ['train', 'test']:
            for k2 in DICT_PUBLIC[k]:
                all_stories += DICT_PUBLIC[k][k2]
            for k2 in DICT_HUGE[k]:
                all_stories += DICT_HUGE[k][k2]
        story_names = list(set(all_stories))
        return story_names

    if use_huge and subject in DICT_HUGE['train'].keys():
        story_names = DICT_HUGE[train_or_test][subject]
    else:
        story_names = DICT_PUBLIC[train_or_test][subject]
    # story_names = [s for s in story_names if not 'canplanetearth' in s]

    return story_names


if __name__ == "__main__":
    for subject in ['UTS01', 'UTS02', 'UTS03', 'UTS04', 'UTS05', 'UTS06', 'UTS07', 'UTS08', 'shared']:
        print(f"Subject: {subject}")
        print(
            f'\tTrain public len: {len(get_story_names(subject, "train", use_huge=False))}')
        print(
            f'\tTrain huge len: {len(get_story_names(subject, "train", use_huge=True))}')
        print(f'\tnum in {subject} (huge) but not in UTS03 (public)', len(
            set(get_story_names(subject, "train", use_huge=True)) - set(get_story_names("UTS03", "train", use_huge=False))))
        print(f'\tnum in {subject} (huge) but not in UTS03 (huge)', len(
            set(get_story_names(subject, "train", use_huge=True)) - set(get_story_names("UTS03", "train", use_huge=True))))
        print(f'\tnum in {subject} (huge) but not in UTS05', len(
            set(get_story_names(subject, "train", use_huge=True)) - set(get_story_names("UTS05", "train", use_huge=False))))
    print('all_stories', len(get_story_names(all=True)))
