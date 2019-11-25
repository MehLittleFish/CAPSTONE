import tensorflow as tf
import os

frozen_model_filename = "C:\\Users\A.Robert\Documents\\Uoit\CAPSTONE\Tensorflow\models\\research\Text_recognition_checkpoint\\frozen-graph/frozen_graph.pb"
img_directory="C:\\Users\A.Robert\Documents\\Uoit\CAPSTONE\Tensorflow\workspace\\training_demo\images\cropped_images"

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

def getImage(path):
    with open(path, 'rb') as img_file:
        img = img_file.read()
        return img
for i in range(1, len(os.listdir(img_directory)) + 1):
    img=getImage(img_directory + '\image' + str(i) + '.jpg')
    (y_out, probs_output) = sess.run([y,allProbs], feed_dict={x: [img]})
    print(str(y_out)[2:-1])
