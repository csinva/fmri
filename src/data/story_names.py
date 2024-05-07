
def get_story_names(subject: str = "UTS01", train_or_test="train", huge=False):
    TRAIN_01_PUBLIC = ['adollshouse', 'gpsformylostidentity', 'singlewomanseekingmanwich', 'adventuresinsayingyes', 'hangtime', 'sloth', 'afatherscover', 'haveyoumethimyet', 'souls', 'againstthewind', 'howtodraw', 'stagefright', 'alternateithicatom', 'ifthishaircouldtalk', 'stumblinginthedark', 'avatar', 'inamoment', 'superheroesjustforeachother', 'backsideofthestorm', 'itsabox', 'sweetaspie', 'becomingindian', 'jugglingandjesus', 'swimmingwithastronauts', 'beneaththemushroomcloud', 'kiksuya', 'thatthingonmyarm', 'birthofanation', 'leavingbaghdad', 'theadvancedbeginner', 'bluehope', 'legacy', 'theclosetthatateeverything', 'breakingupintheageofgoogle', 'lifeanddeathontheoregontrail', 'thecurse', 'buck', 'life', 'thefreedomridersandme', 'catfishingstrangerstofindmyself',
                       'lifereimagined', 'theinterview', 'cautioneating', 'listo', 'thepostmanalwayscalls', 'christmas1940', 'mayorofthefreaks', 'theshower', 'cocoonoflove', 'metsmagic', 'thetiniestbouquet', 'comingofageondeathrow', 'mybackseatviewofagreatromance', 'thetriangleshirtwaistconnection', 'exorcism', 'myfathershands', 'threemonths', 'eyespy', 'myfirstdaywiththeyankees', 'thumbsup', 'firetestforlove', 'naked', 'tildeath', 'food', 'notontheusualtour', 'treasureisland', 'forgettingfear', 'odetostepfather', 'undertheinfluence', 'onlyonewaytofindout', 'vixenandtheussr', 'gangstersandcookies', 'penpal', 'waitingtogo', 'goingthelibertyway', 'quietfire', 'whenmothersbullyback', 'goldiethegoldfish', 'reachingoutbetweenthebars', 'golfclubbing', 'shoppinginchina', 'wildwomenanddancingqueens']
    TRAIN_02_PUBLIC = ['adollshouse', 'hangtime', 'sloth', 'adventuresinsayingyes', 'haveyoumethimyet', 'souls', 'afatherscover', 'howtodraw', 'stagefright', 'againstthewind', 'ifthishaircouldtalk', 'stumblinginthedark', 'alternateithicatom', 'inamoment', 'superheroesjustforeachother', 'avatar', 'itsabox', 'sweetaspie', 'backsideofthestorm', 'jugglingandjesus', 'swimmingwithastronauts', 'becomingindian', 'kiksuya', 'thatthingonmyarm', 'beneaththemushroomcloud', 'leavingbaghdad', 'theadvancedbeginner', 'birthofanation', 'legacy', 'theclosetthatateeverything', 'breakingupintheageofgoogle', 'lifeanddeathontheoregontrail', 'thecurse', 'buck', 'life', 'thefreedomridersandme', 'catfishingstrangerstofindmyself', 'lifereimagined', 'theinterview', 'cautioneating', 'listo',
                       'thepostmanalwayscalls', 'christmas1940', 'mayorofthefreaks', 'theshower', 'cocoonoflove', 'metsmagic', 'thetiniestbouquet', 'comingofageondeathrow', 'mybackseatviewofagreatromance', 'thetriangleshirtwaistconnection', 'exorcism', 'myfathershands', 'threemonths', 'eyespy', 'myfirstdaywiththeyankees', 'thumbsup', 'firetestforlove', 'naked', 'tildeath', 'food', 'notontheusualtour', 'treasureisland', 'forgettingfear', 'odetostepfather', 'undertheinfluence', 'onlyonewaytofindout', 'vixenandtheussr', 'gangstersandcookies', 'penpal', 'waitingtogo', 'goingthelibertyway', 'quietfire', 'whenmothersbullyback', 'goldiethegoldfish', 'reachingoutbetweenthebars', 'golfclubbing', 'shoppinginchina', 'wildwomenanddancingqueens']
    TRAIN_03_PUBLIC = [
        'adollshouse', 'gpsformylostidentity', 'singlewomanseekingmanwich', 'adventuresinsayingyes', 'hangtime', 'sloth', 'afatherscover', 'haveyoumethimyet', 'souls', 'againstthewind', 'howtodraw', 'stagefright', 'alternateithicatom', 'ifthishaircouldtalk', 'stumblinginthedark', 'avatar', 'inamoment', 'superheroesjustforeachother', 'backsideofthestorm', 'itsabox', 'sweetaspie', 'becomingindian', 'jugglingandjesus', 'swimmingwithastronauts', 'beneaththemushroomcloud', 'kiksuya', 'thatthingonmyarm', 'birthofanation', 'leavingbaghdad', 'theadvancedbeginner', 'bluehope', 'legacy', 'theclosetthatateeverything', 'breakingupintheageofgoogle', 'lifeanddeathontheoregontrail', 'thecurse', 'buck', 'life', 'thefreedomridersandme', 'catfishingstrangerstofindmyself', 'lifereimagined', 'theinterview',
        'cautioneating', 'listo', 'thepostmanalwayscalls', 'christmas1940', 'mayorofthefreaks', 'theshower', 'cocoonoflove', 'metsmagic', 'thetiniestbouquet', 'comingofageondeathrow', 'mybackseatviewofagreatromance', 'thetriangleshirtwaistconnection', 'exorcism', 'myfathershands', 'threemonths', 'eyespy', 'myfirstdaywiththeyankees', 'thumbsup', 'firetestforlove', 'naked', 'tildeath', 'food', 'notontheusualtour', 'treasureisland', 'forgettingfear', 'odetostepfather', 'undertheinfluence', 'onlyonewaytofindout', 'vixenandtheussr', 'gangstersandcookies', 'penpal', 'waitingtogo', 'goingthelibertyway', 'quietfire', 'whenmothersbullyback', 'goldiethegoldfish', 'reachingoutbetweenthebars', 'golfclubbing', 'shoppinginchina', 'wildwomenanddancingqueens'
    ]
    TEST_PUBLIC = [
        "wheretheressmoke", "fromboyhoodtofatherhood"
    ]
    TRAIN_01_HUGE = ['itsabox', 'odetostepfather', 'inamoment', 'hangtime', 'ifthishaircouldtalk', 'goingthelibertyway', 'golfclubbing', 'thetriangleshirtwaistconnection', 'igrewupinthewestborobaptistchurch', 'tetris', 'becomingindian', 'canplanetearthfeedtenbillionpeoplepart1', 'thetiniestbouquet', 'swimmingwithastronauts', 'lifereimagined', 'forgettingfear', 'stumblinginthedark', 'backsideofthestorm', 'food', 'theclosetthatateeverything', 'notontheusualtour', 'exorcism', 'adventuresinsayingyes', 'thefreedomridersandme', 'cocoonoflove', 'waitingtogo', 'thepostmanalwayscalls', 'googlingstrangersandkentuckybluegrass', 'mayorofthefreaks', 'learninghumanityfromdogs', 'shoppinginchina', 'souls', 'cautioneating', 'comingofageondeathrow', 'breakingupintheageofgoogle', 'gpsformylostidentity', 'eyespy', 'treasureisland', 'thesurprisingthingilearnedsailingsoloaroundtheworld', 'theadvancedbeginner', 'goldiethegoldfish', 'life', 'thumbsup', 'seedpotatoesofleningrad',
                     'theshower', 'adollshouse', 'canplanetearthfeedtenbillionpeoplepart2', 'sloth', 'howtodraw', 'quietfire', 'metsmagic', 'penpal', 'thecurse', 'canadageeseandddp', 'thatthingonmyarm', 'buck', 'wildwomenanddancingqueens', 'againstthewind', 'indianapolis', 'alternateithicatom', 'bluehope', 'kiksuya', 'afatherscover', 'haveyoumethimyet', 'firetestforlove', 'catfishingstrangerstofindmyself', 'christmas1940', 'tildeath', 'lifeanddeathontheoregontrail', 'vixenandtheussr', 'undertheinfluence', 'beneaththemushroomcloud', 'jugglingandjesus', 'superheroesjustforeachother', 'sweetaspie', 'naked', 'singlewomanseekingmanwich', 'avatar', 'whenmothersbullyback', 'myfathershands', 'reachingoutbetweenthebars', 'theinterview', 'stagefright', 'legacy', 'canplanetearthfeedtenbillionpeoplepart3', 'listo', 'gangstersandcookies', 'birthofanation', 'mybackseatviewofagreatromance', 'lawsthatchokecreativity', 'threemonths', 'whyimustspeakoutaboutclimatechange', 'leavingbaghdad']
    TRAIN_02_HUGE = []
    TRAIN_03_HUGE = []
    TEST_HUGE = [
        "wheretheressmoke", "fromboyhoodtofatherhood", "onapproachtopluto"
    ]
    DICT_PUBLIC = {
        'train': {
            "UTS01": TRAIN_01_PUBLIC,
            "UTS02": TRAIN_02_PUBLIC,
            "UTS03": TRAIN_03_PUBLIC,
            'shared': list(set(TRAIN_01_PUBLIC).intersection(TRAIN_02_PUBLIC, TRAIN_03_PUBLIC))
        },
        'test': {
            "UTS01": TEST_PUBLIC,
            "UTS02": TEST_PUBLIC,
            "UTS03": TEST_PUBLIC,
            'shared': TEST_PUBLIC
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

    if huge:
        return DICT_HUGE[train_or_test][subject]
    else:
        return DICT_PUBLIC[train_or_test][subject]


if __name__ == "__main__":
    for subject in ["UTS01", "UTS02", "UTS03", "shared"]:
        print(f"Subject: {subject}")
        print(
            f'\tTrain public len: {len(get_story_names(subject, "train", huge=False))}')
        print(
            f'\tTrain huge len: {len(get_story_names(subject, "train", huge=True))}')
        print(f'num in {subject} but not in UTS03', len(
            set(get_story_names(subject, "train")) - set(get_story_names("UTS03", "train"))))
