import argparse
import numpy as np
from subprocess import call
import pandas as pd
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
tf.logging.set_verbosity(tf.logging.INFO)
tf.reset_default_graph()
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
dft = pd.read_csv("./train.csv")
fdi = pd.read_csv("./test.csv")
batch=1000
FLAGS=None			
iter =25000	
def nn(x):
	x = tf.layers.dense(inputs=x, units=256, activation=tf.nn.tanh,
				kernel_initializer=None, #tf.random_normal_initializer(0, 1),
				bias_initializer=tf.zeros_initializer(),
		                #kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0005),
		                #bias_regularizer=None
				)#tf.contrib.layers.l2_regularizer(0.001))
	x = tf.layers.batch_normalization(x, training=FLAGS.phase)
	x = tf.layers.dropout(x, 0.70)
	x = tf.layers.dense(inputs=x, units=128, activation=tf.nn.elu,
				kernel_initializer=None, #tf.random_normal_initializer(0, 1),
				bias_initializer=tf.zeros_initializer(),
		                #kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0005),
		                #bias_regularizer=None
				)#tf.contrib.layers.l2_regularizer(0.001))
	x = tf.layers.batch_normalization(x, training=FLAGS.phase)
	x = tf.layers.dropout(x, 0.80)
	x = tf.layers.dense(inputs=x, units=64, activation=tf.nn.relu,
				kernel_initializer=None, #tf.random_normal_initializer(0, 1),
				bias_initializer=tf.zeros_initializer(),
		                #kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0005),
		                #bias_regularizer=None
				)#tf.contrib.layers.l2_regularizer(0.001))
	x = tf.layers.dense(inputs=x, units=10, activation=None)#tf.nn.softmax)
	return x
    
def model_fn(features, labels, mode):
	y = nn(features)
	train_op = None
	loss = tf.convert_to_tensor(0.)
   	predictions = None
   	eval_metric_ops = None
 	global_step = tf.train.get_global_step()
	if(mode == tf.estimator.ModeKeys.EVAL or
       	mode == tf.estimator.ModeKeys.TRAIN):
		loss = tf.losses.softmax_cross_entropy(
					onehot_labels=labels, logits=y)# + tf.losses.get_regularization_loss()
   	if(mode == tf.estimator.ModeKeys.TRAIN):
		lr = tf.train.exponential_decay(0.0001, global_step, 100, 0.10000, staircase=False)
		train_op = tf.train.AdamOptimizer(0.00001).minimize(loss, global_step = global_step)

   	if(mode == tf.estimator.ModeKeys.PREDICT):
		predictions = {"pred_labels": tf.argmax(y, 1)}
   	if(mode == tf.estimator.ModeKeys.EVAL):
		eval_metric_ops = {"accuracy": tf.metrics.accuracy(tf.argmax(labels,1), tf.argmax(y,1)),
                          "precision": tf.metrics.precision(tf.argmax(labels,1), tf.argmax(y,1)),
                          "recall": tf.metrics.recall(tf.argmax(labels,1), tf.argmax(y,1))}
	
	return tf.estimator.EstimatorSpec(mode=mode, loss=loss, 
					train_op=train_op, predictions=predictions, eval_metric_ops = eval_metric_ops)

def input_fn():
   	t1, t2 = mnist.train.next_batch(batch)
	t2 = tf.reshape(t2, [batch, 10])
	mean, var = tf.nn.moments(tf.convert_to_tensor(t1), 1, keep_dims=True)
	t1 = (tf.reshape(t1, [batch, 784]) -mean)/tf.sqrt(var)
	return t1, t2

def ip_fn():
	data = dft.sample(batch,replace=False)
	d = tf.cast(tf.contrib.learn.extract_pandas_data(data), tf.float32)
	labels, features = tf.split(d, [1,784], 1)
	labels = tf.reshape(tf.one_hot(tf.cast(labels, tf.int32), 10), [batch, 10])
	mean, var = tf.nn.moments(features, 1, keep_dims=True)
	features = (tf.reshape(features, [batch, 784])-mean)/tf.sqrt(var)
	return features, labels

def ip_fn_infer():
    a=[1.]
    FLAGS=None
    file_q = tf.train.string_input_producer(["./test.csv"],
                                                num_epochs=1, shuffle=False)
    reader = tf.TextLineReader(skip_header_lines=1)
    key, val = reader.read(file_q)
    record_defaults = [a for i in range(784)]
    rev = tf.decode_csv(
                    val, record_defaults, field_delim=',')
    batch = 1000
    min_after_dequeue=5000
    capacity = min_after_dequeue + 3 * batch
    feat_id, feat_rev = tf.train.batch([rev, rev], batch_size=batch,
                                    capacity = capacity,
                                    #min_after_dequeue = min_after_dequeue,
                                    allow_smaller_final_batch=True)

    mean, var = tf.nn.moments(feat_rev, 1, keep_dims=True)
    features = (tf.reshape(feat_rev, [batch, 784])-mean)/tf.sqrt(var)
    labels = None
    return features, labels

def main(_):
	runConfig = tf.estimator.RunConfig()
	
	est = tf.estimator.Estimator(model_fn = model_fn,
                            model_dir="/home/udaygurugubelli/dnn_kaggle_sig_fulldata/model_dnn/",
							config = runConfig.replace(save_summary_steps=500,
                                                save_checkpoints_steps=500))
                                                #save_checkpoints_secs=100))

	if(FLAGS.test == False):
		for i in range(iter):
			if(FLAGS.eval == False):
				est.train(input_fn = ip_fn, steps=1)
				est.train(input_fn = input_fn, steps=1)
			if((i+1) % 2 == 0):
				est.evaluate(input_fn=ip_fn, steps=1)
				est.evaluate(input_fn=input_fn, steps=1)

	if FLAGS.test == True:
		pred = est.predict(input_fn=ip_fn_infer)
		sub_file = "kaggle_mnist_sub.csv"

		f_hand = open(sub_file, "w")
		f_hand.write("ImageId,Label\n")
		for i, p in enumerate(pred):
			pp = p["pred_labels"]
			line = str(i+1) + "," + str(pp) + "\n"
			f_hand.write(line)

		f_hand.close()  

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

