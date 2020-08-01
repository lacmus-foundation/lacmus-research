# Compares times of getting image size by PIL.Image and cv2.imread

import cv2
from PIL import Image
import os
import time

dir = '../data/laddv4/spring/JPEGImages/'
files = list(os.listdir(dir))

ratio = 0.0
start = time.time()
for f in files:
    image = Image.open(dir + f)
    ratio += float(image.width) / float(image.height)
end = time.time()
print('Getting images sizes from PIL: ', (end - start) / len(files), 'ms per image')

start = time.time()
for f in files:
    img = cv2.imread(dir + f)
    height, width, _ = img.shape
    ratio += float(width) / float(height)
end = time.time()
print('Getting images sizes from cv2: ', (end - start) / len(files), 'ms per image')

# Output:
# Getting images sizes from PIL:  0.001029537684881865 ms per image
# Getting images sizes from cv2:  0.22250559614665472 ms per image



