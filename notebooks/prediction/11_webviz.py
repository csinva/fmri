import cortex

import numpy as np
np.random.seed(1234)

# gather multiple datasets
volume1 = cortex.Volume.random(
    subject='UTS01', xfmname='UTS01_auto', priority=1)
volume2 = cortex.Volume.random(
    subject='UTS02', xfmname='UTS02_auto', priority=2)
# volume2 = cortex.Volume.random(subject='S1', xfmname='fullhead', priority=2)
# volume3 = cortex.Volume.random(subject='S1', xfmname='fullhead', priority=3)
volumes = {
    'UTS01': volume1,
    'UTS02': volume2,
    # 'Third Dataset': volume3,
}

# create viewer
# cortex.webgl.show(data=volumes)
cortex.webgl.make_static(outpath='static_webviz', data=volume1, recache=True)

user_input = input("Please enter something and press Enter to finish: ")

# Print the user's input
print(f"Ending because user input entered: {user_input}")

# a webserver such as nginx can then be used to host the static viewer
