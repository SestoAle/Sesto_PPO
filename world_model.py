import numpy as np
import json
from PIL import Image

with open("arrays/{}.json".format("bug_detector_rnd_pos_buffer")) as f:
    buffer = json.load(f)

img = Image.new('RGB', (20, 20))
for k in buffer.keys():
    k_value = list(map(float, k.split(" ")))
    k_value = np.asarray(k_value)
    k_value = (((k_value + 1) / 2) * 19)
    k_value = k_value.astype(int)
    img.putpixel(k_value[:2], (155,155,55))

img.save('sqr.png')