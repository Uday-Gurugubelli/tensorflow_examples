import argparse
import numpy as np
from subprocess import call
import pandas as pd
import tensorflow as tf
#from tensorflow.examples.tutorials.mnist import input_data
#from keras.datasets import mnist
tf.logging.set_verbosity(tf.logging.INFO)
tf.reset_default_graph()
#mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)

batch=100
FLAGS=None

lbls = np.loadtxt("./train.csv",
                  delimiter=",", skiprows=1, usecols=[0]).astype(int)
#print(lbls.size)
#print(lbls.max().astype(int))
#labels = np.zeros(lbls.size, lbls.max().astype(int)+1)
#labels[np.arange(lbls.size), lbls] = 1
lb_values = np.max(lbls) + 1
labels = np.eye(lb_values)[lbls]

cols = [i for i in range(785)]
cols = cols[1:]
features = np.loadtxt("./train.csv", delimiter=",", skiprows=1, usecols=cols)

cols = [i for i in range(784)]
test_features = np.loadtxt("./test.csv", delimiter=",", skiprows=1, usecols=cols)

#(x_train, y_train), (x_test, y_test) = mnist.load_data()
#x_train = x_train.reshape(-1, 784)
#x_test = x_test.reshape(-1, 784)
feature_columns = [tf.feature_column.numeric_column(key = 'features', shape=(784,))]

classifier = tf.estimator.DNNClassifier(
                hidden_units = [1024, 512, 256],
                feature_columns = feature_columns,
                model_dir = None,
                n_classes = 10)

def main(_):
    #file = open('./test.csv', "r")
    #fl = file.readline()
    wf = open('./mnist_kaggle_submit_file.txt', "w")
    wf.write("ImageId,Label\n")

    classifier.train(input_fn = tf.estimator.inputs.numpy_input_fn(
        dict({"features":features}), lbls,
        batch_size=100, num_epochs=25, shuffle=True), steps= 10000)
    classifier.evaluate(input_fn = tf.estimator.inputs.numpy_input_fn(
        dict({'features':features}), lbls, shuffle=True), steps= 1)
    pred = classifier.predict(input_fn = tf.estimator.inputs.numpy_input_fn(
        dict({'features':test_features}), batch_size=100, shuffle=True))
    for i, prd in enumerate(pred):
        #print(i, prd)
        [opp] = prd['classes'].astype(int)
        #print(opp)
        ll = str(i+1) + "," + str(opp) + "\n"                        
        wf.write(ll)
    #file.close()
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
