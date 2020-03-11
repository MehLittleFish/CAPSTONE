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

    # Section the target image into grids of 2X3
    # Section grids will pass though the model
    def split_image(self):
        img = Image.open(self.TARGET_IMAGE_PATH)
        width, height = img.size
        cropped = []
        sixth = height/6
        twelve = height / 12
        cropped.append(img.crop((0, 0, width, sixth)))
        cropped.append(img.crop((0, twelve, width, 1.5*sixth)))
        cropped.append(img.crop((0, sixth, width, 2*sixth)))
        cropped.append(img.crop((0, (3 * (height / 12)), width, 2.5*sixth)))
        cropped.append(img.crop((0, (4 * (height / 12)), width, 3*sixth)))
        cropped.append(img.crop((0, (5 * (height / 12)), width, 3.5*sixth)))
        cropped.append(img.crop((0, (6 * (height / 12)), width, 4*sixth)))
        cropped.append(img.crop((0, (7 * (height / 12)), width, 4.5*sixth)))
        cropped.append(img.crop((0, (8 * (height / 12)), width, 5*sixth)))
        cropped.append(img.crop((0, (9 * (height / 12)), width, 5.5*sixth)))
        cropped.append(img.crop((0, (10 * (height / 12)), width, height)))
        
        # Save each section as a new image
        for x in range(0, len(cropped)):
            #cropped[x].save(
                #CROP_DIR + "/crop" + str(x + 1) + ".jpg")
            self.run_detection(cropped[x], x)
        self.save_the_things(img)

    # Main method for text detection
    def run_detection(self, cropped, iteration):
        image = cropped
        im_width, im_height = image.size
        coor = []
        c1 = 0
        c2 = 0
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
            coor.append(
                (output_dict['detection_boxes'][i][1] * im_width, output_dict['detection_boxes'][i][0] * im_height + (iteration*im_height/2),
                 output_dict['detection_boxes'][i][3] * im_width, output_dict['detection_boxes'][i][2] * im_height + (iteration*im_height/2)))
        
        # Sort the cropped words into the order they would be read in
        # NOTE This implementation doesn't work anymore with 'split_image' method
        coor = sorted(coor, key=lambda k: [k[0], k[1]])
        coor = sorted(coor, key=lambda k: [k[1], k[0]])
        print (coor)
        
        while True:
            for i in range(0, len(coor) - 1):
                if coor[i][0] > coor[i + 1][0]:
                    if 20 > coor[i + 1][1] - coor[i][1] > -20:
                        coor[i], coor[i + 1] = coor[i + 1], coor[i]
                        c1 += 1
            
            if c1 == c2:
                break

            c2 = c1
        self.merge_boxy_bois(iteration, im_height, coor)
        # Save cropped words into images
        """
        for i in range(0, output_dict['num_detections']):
            img2 = image.crop(coor[i])
            img2.save(
                WORD_DIR + "/image" + str(
                    self.counter + 1) + ".jpg")
            self.counter += 1
            #print(self.counter)

        #plt.figure(figsize=self.IMAGE_SIZE)
        #plt.imshow(image_np)
        #plt.show()
        """
    #TODO Add a function that checks coordinates, should reorganize the sentences so they actually make sense
    def merge_boxy_bois(self, iteration, height, coor):
        top = []
        bottom = []
        if iteration == 0:
            for x1 in coor:
                if x1[3] > height/2:
                    print("2")
                    self.temp_words.append(x1)
                else:
                    self.words.append(x1)
        else:
            for x1 in coor:
                if x1[3] < (iteration - 1)*(height / 2)+height:
                    top.append(x1)
                else:
                    bottom.append(x1)
            found = False
            for x1 in top:
                for y in self.temp_words:
                    if self.do_over_lap(x1, y):
                        self.words.append(self.merge(x1, y))
                        found = True
                        break
                if found == False:
                    self.words.append(x1)
                found = False
            self.temp_words = bottom


    @staticmethod
    def do_over_lap(x2, y):

        # If one rectangle is on left side of other
        if x2[0] > y[2] or y[0] > x2[2]:
            return False

        # If one rectangle is above other
        if x2[1] > y[3] or y[1] > x2[3]:
            return False

        return True

    @staticmethod
    def merge(x, y):
        c = []
        c.append(min(x[0], y[0]))
        c.append(min(x[1], y[1]))
        c.append(max(x[2], y[2]))
        c.append(max(x[3], y[3]))
        return c

    def save_the_things(self, img):
        for i in range(0, len(self.words)):
            img2 = img.crop(self.words[i])
            img2.save(
                WORD_DIR + "/image" + str(
                    self.counter + 1) + ".jpg")
            self.counter += 1

x = TextDetection()
x.split_image()