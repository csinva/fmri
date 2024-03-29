
def get_train_story_texts(subject: str = "UTS01", train_or_test="train"):
    TRAIN_STORIES_01 = [
        "itsabox",
        "odetostepfather",
        "inamoment",
        "hangtime",
        "ifthishaircouldtalk",
        "goingthelibertyway",
        "golfclubbing",
        "thetriangleshirtwaistconnection",
        "igrewupinthewestborobaptistchurch",
        "tetris",
        "becomingindian",
        "canplanetearthfeedtenbillionpeoplepart1",
        "thetiniestbouquet",
        "swimmingwithastronauts",
        "lifereimagined",
        "forgettingfear",
        "stumblinginthedark",
        "backsideofthestorm",
        "food",
        "theclosetthatateeverything",
        "notontheusualtour",
        "exorcism",
        "adventuresinsayingyes",
        "thefreedomridersandme",
        "cocoonoflove",
        "waitingtogo",
        "thepostmanalwayscalls",
        "googlingstrangersandkentuckybluegrass",
        "mayorofthefreaks",
        "learninghumanityfromdogs",
        "shoppinginchina",
        "souls",
        "cautioneating",
        "comingofageondeathrow",
        "breakingupintheageofgoogle",
        "gpsformylostidentity",
        "eyespy",
        "treasureisland",
        "thesurprisingthingilearnedsailingsoloaroundtheworld",
        "theadvancedbeginner",
        "goldiethegoldfish",
        "life",
        "thumbsup",
        "seedpotatoesofleningrad",
        "theshower",
        "adollshouse",
        "canplanetearthfeedtenbillionpeoplepart2",
        "sloth",
        "howtodraw",
        "quietfire",
        "metsmagic",
        "penpal",
        "thecurse",
        "canadageeseandddp",
        "thatthingonmyarm",
        "buck",
        "wildwomenanddancingqueens",
        "againstthewind",
        "indianapolis",
        "alternateithicatom",
        "bluehope",
        "kiksuya",
        "afatherscover",
        "haveyoumethimyet",
        "firetestforlove",
        "catfishingstrangerstofindmyself",
        "christmas1940",
        "tildeath",
        "lifeanddeathontheoregontrail",
        "vixenandtheussr",
        "undertheinfluence",
        "beneaththemushroomcloud",
        "jugglingandjesus",
        "superheroesjustforeachother",
        "sweetaspie",
        "naked",
        "singlewomanseekingmanwich",
        "avatar",
        "whenmothersbullyback",
        "myfathershands",
        "reachingoutbetweenthebars",
        "theinterview",
        "stagefright",
        "legacy",
        "canplanetearthfeedtenbillionpeoplepart3",
        "listo",
        "gangstersandcookies",
        "birthofanation",
        "mybackseatviewofagreatromance",
        "lawsthatchokecreativity",
        "threemonths",
        "whyimustspeakoutaboutclimatechange",
        "leavingbaghdad",
    ]
    TRAIN_STORIES_02_03 = [
        "itsabox",
        "odetostepfather",
        "inamoment",
        "afearstrippedbare",
        "findingmyownrescuer",
        "hangtime",
        "ifthishaircouldtalk",
        "goingthelibertyway",
        "golfclubbing",
        "thetriangleshirtwaistconnection",
        "igrewupinthewestborobaptistchurch",
        "tetris",
        "becomingindian",
        "canplanetearthfeedtenbillionpeoplepart1",
        "thetiniestbouquet",
        "swimmingwithastronauts",
        "lifereimagined",
        "forgettingfear",
        "stumblinginthedark",
        "backsideofthestorm",
        "food",
        "theclosetthatateeverything",
        "escapingfromadirediagnosis",
        "notontheusualtour",
        "exorcism",
        "adventuresinsayingyes",
        "thefreedomridersandme",
        "cocoonoflove",
        "waitingtogo",
        "thepostmanalwayscalls",
        "googlingstrangersandkentuckybluegrass",
        "mayorofthefreaks",
        "learninghumanityfromdogs",
        "shoppinginchina",
        "souls",
        "cautioneating",
        "comingofageondeathrow",
        "breakingupintheageofgoogle",
        "gpsformylostidentity",
        "marryamanwholoveshismother",
        "eyespy",
        "treasureisland",
        "thesurprisingthingilearnedsailingsoloaroundtheworld",
        "theadvancedbeginner",
        "goldiethegoldfish",
        "life",
        "thumbsup",
        "seedpotatoesofleningrad",
        "theshower",
        "adollshouse",
        "canplanetearthfeedtenbillionpeoplepart2",
        "sloth",
        "howtodraw",
        "quietfire",
        "metsmagic",
        "penpal",
        "thecurse",
        "canadageeseandddp",
        "thatthingonmyarm",
        "buck",
        "thesecrettomarriage",
        "wildwomenanddancingqueens",
        "againstthewind",
        "indianapolis",
        "alternateithicatom",
        "bluehope",
        "kiksuya",
        "afatherscover",
        "haveyoumethimyet",
        "firetestforlove",
        "catfishingstrangerstofindmyself",
        "christmas1940",
        "tildeath",
        "lifeanddeathontheoregontrail",
        "vixenandtheussr",
        "undertheinfluence",
        "beneaththemushroomcloud",
        "jugglingandjesus",
        "superheroesjustforeachother",
        "sweetaspie",
        "naked",
        "singlewomanseekingmanwich",
        "avatar",
        "whenmothersbullyback",
        "myfathershands",
        "reachingoutbetweenthebars",
        "theinterview",
        "stagefright",
        "legacy",
        "canplanetearthfeedtenbillionpeoplepart3",
        "listo",
        "gangstersandcookies",
        "birthofanation",
        "mybackseatviewofagreatromance",
        "lawsthatchokecreativity",
        "threemonths",
        "whyimustspeakoutaboutclimatechange",
        "leavingbaghdad",
    ]
    TEST_STORIES = ["wheretheressmoke",
                    "onapproachtopluto", "fromboyhoodtofatherhood"]
    story_names_train = {
        "UTS01": TRAIN_STORIES_01,
        "UTS02": TRAIN_STORIES_02_03,
        "UTS03": TRAIN_STORIES_02_03,
    }
    if train_or_test == "train":
        return story_names_train[subject]
    elif train_or_test == "test":
        return TEST_STORIES
