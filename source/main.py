#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append("..")
from Tensorflow.models.research.object_detection import text_detection
from Tensorflow.models.research.Text_recognition_checkpoint import predict

test = text_detection.TextDetection()
images = test.split_image()
shoop = predict.Predict()
shoop.run_prediction()
