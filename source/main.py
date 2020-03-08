#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os
from capstone import ROOT_DIR

sys.path.append(ROOT_DIR)
sys.path.append((os.path.join(ROOT_DIR, "Tensorflow")))

from Tensorflow.object_detection import text_detection
from Tensorflow.text_recognition import predict


test = text_detection.TextDetection()
images = test.split_image()
shoop = predict.Predict()
shoop.run_prediction()
