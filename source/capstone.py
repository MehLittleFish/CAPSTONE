# -------All the constants in this capstone---------

import sys, os

# -------Root Directory of the program------------
ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

# -------Image Directories--------------
TARGET_DIR = os.path.join(ROOT_DIR, "images/test") # Relative path to target images
CROP_DIR = os.path.join(ROOT_DIR, "images/temporary/cropped")
WORD_DIR = os.path.join(ROOT_DIR, "images/temporary/words")

# --------Text Detection Constants--------
# Path to frozen detection graph. This is the actual model that is used for the object detection.
DETECTION_MODEL_PATH = os.path.join(ROOT_DIR, "Tensorflow/object_detection/text_graph/frozen_inference_graph.pb")
# List of the strings that is used to add correct label for each box.
DETECTION_LABEL_PATH = os.path.join(ROOT_DIR, "Tensorflow/object_detection/data/text_label_map.pbtxt")

# --------Text Recognition Constants--------
# Path to frozen recognition graph. This is the actual model that is used for the text recognition.
RECOGNITION_MODEL_PATH = os.path.join(ROOT_DIR, "Tensorflow/text_recognition/frozen-graph/frozen_graph.pb")