import numpy as np
import sys, os
import tensorflow as tf

from source.capstone import TARGET_FILE_NAME, TARGET_DIR, CROP_DIR, WORD_DIR, DETECTION_MODEL_PATH, DETECTION_LABEL_PATH
from matplotlib import pyplot as plt
from PIL import Image

from Tensorflow.object_detection.utils import ops as utils_ops
from Tensorflow.object_detection.utils import label_map_util
from Tensorflow.object_detection.utils import visualization_utils as vis_util


class TextDetection:
    NUM_CLASSES = 1
    counter = 0
    words = []
    temp_words = []
    detection_graph = tf.Graph()

    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()

        with tf.compat.v1.gfile.GFile(DETECTION_MODEL_PATH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    label_map = label_map_util.load_labelmap(DETECTION_LABEL_PATH)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                use_display_name=False)
    category_index = label_map_util.create_category_index(categories)

    TARGET_IMAGE_PATH = os.path.join(TARGET_DIR, TARGET_FILE_NAME)

    # Size, in inches, of the output images.
    IMAGE_SIZE = (24, 16)

    @staticmethod
    def load_image_into_numpy_array(image):
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)

    @staticmethod
    def run_inference_for_single_image(image, graph):
        with graph.as_default():
            with tf.compat.v1.Session() as sess:
                # Get handles to input and output tensors
                ops = tf.compat.v1.get_default_graph().get_operations()
                all_tensor_names = {output.name for op in ops for output in op.outputs}
                tensor_dict = {}

                for key in [
                    'num_detections', 'detection_boxes', 'detection_scores',
                    'detection_classes', 'detection_masks']:
                    tensor_name = key + ':0'

                    if tensor_name in all_tensor_names:
                        tensor_dict[key] = tf.compat.v1.get_default_graph().get_tensor_by_name(
                            tensor_name)

                if 'detection_masks' in tensor_dict:
                    # The following processing is only for single image
                    detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                    detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                    # Reframe is required to translate mask from box coordinates to
                    # image coordinates and fit the image size.
                    real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                    detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                    detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                        detection_masks, detection_boxes, image.shape[0], image.shape[1])
                    detection_masks_reframed = tf.cast(
                        tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                    # Follow the convention by adding back the batch dimension
                    tensor_dict['detection_masks'] = tf.expand_dims(
                        detection_masks_reframed, 0)

                image_tensor = tf.compat.v1.get_default_graph().get_tensor_by_name('image_tensor:0')

                # Run inference
                output_dict = sess.run(tensor_dict,
                                       feed_dict={image_tensor: np.expand_dims(image, 0)})

                # all outputs are float32 numpy arrays, so convert types as appropriate
                output_dict['num_detections'] = int(output_dict['num_detections'][0])
                output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
                output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                output_dict['detection_scores'] = output_dict['detection_scores'][0]

                if 'detection_masks' in output_dict:
                    output_dict['detection_masks'] = output_dict['detection_masks'][0]

        return output_dict

    def split_image(self):
        img = Image.open(self.TARGET_IMAGE_PATH)
        width, height = img.size
        x = width / 3
        y = height / 3
        # Crops images into quadrants
        cropped = [img.crop((0, 0, 2 * x, 2 * y)),
                   img.crop((x, 0, width, 2 * y)),
                   img.crop((0, y, 2 * x, 2 * height)),
                   img.crop((x, y, width, height))]

        # Run object detection on each portion of image
        i = 0
        for q in cropped:
            self.run_detection(q, i, x, y)
            i += 1
        self.clean_list(x, y)
        self.save_the_things(img)

    # Main method for text detection
    def run_detection(self, cropped, iteration, x, y):
        image = cropped
        im_width, im_height = image.size

        # Checks how much of a margin to add depending on iteration
        if iteration == 1:
            w = x
            h = 0
        elif iteration == 2:
            w = 0
            h = y
        elif iteration == 3:
            w = x
            h = y
        else:
            w = 0
            h = 0
        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        image_np = self.load_image_into_numpy_array(image)
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        # Actual detection.
        output_dict = self.run_inference_for_single_image(image_np, self.detection_graph)
        # Visualization of the results of a detection
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            self.category_index,
            instance_masks=output_dict.get('detection_masks'),
            use_normalized_coordinates=True,
            line_thickness=3)

        for i in range(0, output_dict['num_detections']):
            self.words.append(
                (output_dict['detection_boxes'][i][1] * im_width + w,
                 output_dict['detection_boxes'][i][0] * im_height + h,
                 output_dict['detection_boxes'][i][3] * im_width + w,
                 output_dict['detection_boxes'][i][2] * im_height + h))

    def clean_list(self, w, h):
        c1 = 0
        c2 = 0
        # iterate through list and find any overlapping boxes, merge and delete
        for i in range(0, len(self.words)-1):
            index = i + 1
            while index < len(self.words):
                if self.do_overlap(self.words[i], self.words[index]):
                    self.words[i] = self.merge(self.words[i], self.words[index])
                    del (self.words[index])
                else:
                    index += 1

        # Sorts the words from left most top most
        self.words = sorted(self.words, key=lambda k: [k[0], k[1]])
        self.words = sorted(self.words, key=lambda k: [k[1], k[0]])

        while True:
            for i in range(0, len(self.words) - 1):
                if self.words[i][0] > self.words[i + 1][0]:
                    if 20 > self.words[i + 1][1] - self.words[i][1] > -20:
                        self.words[i], self.words[i + 1] = self.words[i + 1], self.words[i]
                        c1 += 1

            if c1 == c2:
                break

            c2 = c1

    def is_isolated(self, x, w, h):
        # checks if its any of the four corners
        if (self.words[x][2] < w and self.words[x][3] < h) \
                or (self.words[x][0] > 2 * w and self.words[x][3] < h) \
                or (self.words[x][2] < w and self.words[x][1] > 2 * h) \
                or (self.words[x][0] > 2 * w and self.words[x][1] > 2 * h):
            return True
        else:
            return False

    @staticmethod
    def do_overlap(x, y):
        # Tolerance of overlap, if barely touching it won't be considered an overlap
        t = 0
        # If one rectangle is on left side of other
        if x[0] > y[2] - t or y[0] > x[2] - t:
            return False

        # If one rectangle is above other
        if x[1] > y[3] - t or y[1] > x[3] - t:
            return False

        return True

    @staticmethod
    def merge(x, y):
        c = [min(x[0], y[0]), min(x[1], y[1]), max(x[2], y[2]), max(x[3], y[3])]
        return c

    def save_the_things(self, img):
        for i in range(0, len(self.words)):
            img2 = img.crop(self.words[i])
            img2.save(
                WORD_DIR + "/image" + str(i) + ".jpg")


q = TextDetection()
q.split_image()
