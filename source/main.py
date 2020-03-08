#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os

ABS_PATH = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.join(ABS_PATH, '..')
sys.path.append(ROOT_PATH)
sys.path.append((os.path.join(ROOT_PATH, "Tensorflow/models/research")))

from Tensorflow.models.research.object_detection import text_detection
from Tensorflow.models.research.Text_recognition_checkpoint import predict


test = text_detection.TextDetection()
images = test.split_image()
shoop = predict.Predict()
shoop.run_prediction()
