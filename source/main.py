#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os
from capstone import ROOT_DIR, CROP_DIR, WORD_DIR

sys.path.append(ROOT_DIR)
sys.path.append((os.path.join(ROOT_DIR, "Tensorflow")))

from Tensorflow.object_detection import text_detection
from Tensorflow.text_recognition import predict


# Clears all files in a given path
def clear_images(path):
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

test = text_detection.TextDetection()
images = test.split_image()
shoop = predict.Predict()
shoop.run_prediction()
word_list = shoop.get_prediction()
print(word_list)
print(len(word_list))
# Delete temporary images
#clear_images(CROP_DIR)
clear_images(WORD_DIR)
