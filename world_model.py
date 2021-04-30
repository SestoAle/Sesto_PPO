import numpy as np
import json
from PIL import Image

with open("arrays/{}.json".format("bug_detector_pos_buffer")) as f:
    buffer = json.load(f)

img = Image.new('RGB', (20, 20))
for k in buffer.keys():
    k_value = list(map(float, k.split(" ")))
    k_value = np.asarray(k_value)
    k_value = (((k_value + 1) / 2) * 19)
    k_value = k_value.astype(int)
    try:
        img.putpixel(k_value, (155,155,55))
    except Exception as e:
        print(k)
        print(k_value)
        print(e)
        input('...')
img.save('sqr.png')