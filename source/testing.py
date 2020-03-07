#!/usr/bin/env python
# -*- coding: utf-8 -*-
from PIL import Image

img = Image.open('image.jpg')
width, height = img.size
half = width / 2
thirds = height / 3
cropped = []
cropped.append(img.crop((0, 0, half, thirds)))
cropped.append(img.crop((0, thirds, half, 2 * thirds)))
cropped.append(img.crop((0, 2 * thirds, half, height)))
cropped.append(img.crop((half, 0, width, thirds)))
cropped.append(img.crop((half, thirds, width, 2 * thirds)))
cropped.append(img.crop((half, 2 * thirds, width, height)))

for x in range(6):
    cropped[x].save(
        "C:/Users/Abe/Documents/GitHub/CAPSTONE/source/testing/crop" + str(
            x + 1) + ".jpg")