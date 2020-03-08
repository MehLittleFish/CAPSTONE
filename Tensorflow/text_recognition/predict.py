import tensorflow as tf
import os

ABS_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ABS_DIR, "frozen-graph")
IMG_DIR = os.path.join(ABS_DIR, "../../images/temporary/words")

class Predict:
    frozen_model_filename = os.path.join(MODEL_DIR, "frozen_graph.pb")
    config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
    sess = tf.compat.v1.Session(config=config)

    with tf.io.gfile.GFile(frozen_model_filename, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()

    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')
    sess.run(tf.compat.v1.global_variables_initializer())

    x = sess.graph.get_tensor_by_name('input_image_as_bytes:0')
    y = sess.graph.get_tensor_by_name('prediction:0')
    allProbs = sess.graph.get_tensor_by_name('probability:0')

    @staticmethod
    def get_image(path):
        with open(path, 'rb') as img_file:
            img = img_file.read()
            return img

    def run_prediction(self):
        for i in range(1, len(os.listdir(IMG_DIR)) + 1):
            img = self.get_image(IMG_DIR + '/image' + str(i) + '.jpg')
            (y_out, probs_output) = self.sess.run([self.y, self.allProbs], feed_dict={self.x: [img]})
            print(str(y_out)[2:-1])
