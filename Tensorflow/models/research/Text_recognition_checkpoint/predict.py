import tensorflow as tf
import os


class Predict:
    frozen_model_filename = "C:/Users/Abe/Documents/GitHub/CAPSTONE/Tensorflow/models/research/Text_recognition_checkpoint/frozen-graph/frozen_graph.pb"
    path = os.path.dirname(os.path.realpath(__file__))
    img_directory = "C:/Users/Abe/Documents/GitHub/CAPSTONE/Tensorflow/models/research/object_detection/test_images/Test_boi"

    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)
    with tf.gfile.GFile(frozen_model_filename, 'rb') as f:
        graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')
    sess.run(tf.global_variables_initializer())

    x = sess.graph.get_tensor_by_name('input_image_as_bytes:0')
    y = sess.graph.get_tensor_by_name('prediction:0')
    allProbs = sess.graph.get_tensor_by_name('probability:0')

    @staticmethod
    def get_image(path):
        with open(path, 'rb') as img_file:
            img = img_file.read()
            return img

    def run_prediction(self):
        for i in range(1, len(os.listdir(self.img_directory)) + 1):
            img = self.get_image(self.img_directory + '/image' + str(i) + '.jpg')
            (y_out, probs_output) = self.sess.run([self.y, self.allProbs], feed_dict={self.x: [img]})
            print(str(y_out)[2:-1])

