import argparse
import numpy as np
from subprocess import call
import pandas as pd
import tensorflow as tf

from keras.datasets import mnist
tf.logging.set_verbosity(tf.logging.INFO)
tf.reset_default_graph()

FLAGS=None			
iter =25000

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)
feature_columns = [tf.feature_column.numeric_column(key='image', shape=(784,))]

classifier = tf.estimator.DNNClassifier(
                hidden_units = [1024, 512, 256],
                feature_columns = feature_columns,
                model_dir = None,
                n_classes = 10)
                
def main(_):
    classifier.train(input_fn = tf.estimator.inputs.numpy_input_fn(
        dict({'image':x_train}), np.array(y_train,np.int32),
        shuffle=True), steps= 1000)
    classifier.evaluate(input_fn = tf.estimator.inputs.numpy_input_fn(
        dict({'image':x_train}), np.array(y_train,np.int32),
        shuffle=True), steps= 100)
    file = open('test.csv', "r")
    wf = open('mnist_kaggle_submit_file.txt', "w")
    wf.write("ImageId,Label\n")
    for i, line in file.readlines():
        op = classifier.predict(input_fn = tf.estimator.inputs.numpy_input_fn(
            dict({'image':line})))
        ll = str(i+1) + "," + str(op) + "\n"                        
        wf.write(ll)
    file.close()
    wf.close()
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', type=bool,
                        default=False,
                        help='path to model dir')
    parser.add_argument('--test', type=bool,
                        default=False,
                        help='path to model dir')
    parser.add_argument('--phase', type=bool,
                        default=True,
                        help='path to model dir')
    FLAGS, unparsed = parser.parse_known_args()

tf.app.run()
