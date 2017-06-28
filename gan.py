import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
tf.logging.set_verbosity(tf.logging.INFO)
tf.reset_default_graph()

mnist = input_data.read_data_sets("MNIST_data/")
img_dim = 784
z_dim=200
total_damples = len(mnist)

def generator(x):
    x = tf.layers.dense(inputs=x, units=256, activation=tf.nn.relu)
    x = tf.layers.dense(inputs=x, units=img_dim, activation=tf.nn.sigmoid)
    return x
def discriminator(x):
    x = tf.layers.dense(inputs=x, units=256, activation=tf.nn.relu)
    x = tf.layers.dense(inputs=x, units=1, activation=tf.nn.sigmoid)
    return x
def model_fn(features, labels, mode):

    gen_sample = generator(features)
    disc_real = discriminator(gen_sample)

    if(mode == tf.estimator.ModeKeys.EVAL or
            mode == tf.estimator.ModeKeys.TRAIN):

        loss = -tf.reduce_mean(labels*tf.log(disc_real) + labels*tf.log(1. - disc_real))
    train_op = None
    global_step = tf.train.get_global_step()
    if(mode == tf.estimator.ModeKeys.TRAIN):
        train_op = tf.train.GradientDescentOptimizer(0.01).minimize(
                            loss, global_step = global_step)

    predictions = {"classes": features,
                   "labels ": labels}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss,
                                    train_op=train_op, predictions=predictions)

def my_input_fn():
    len = 10000
    rand = np.random.randint(0, 2)
    if(rand == 1):
        x = tf.random_uniform([len, 784], 0, 1)
        y = tf.convert_to_tensor(
                    tf.random_uniform([len, 1], 0, 10, tf.int32))
        y = tf.cast(tf.tile(y, [1,784]), tf.float32)
    else:
        x, y1 = mnist.train.next_batch(len)
        y = tf.reshape(y1, [len, 1])
        y = tf.cast(tf.tile(y, [1,784]), tf.float32)

    g, h = tf.train.slice_input_producer([x, y])
    
    t1, t2  = tf.train.shuffle_batch_join(tensors_list = [[g, h]],
                               batch_size=100,
                               capacity=1000,
                               min_after_dequeue=500)
    
    return tf.convert_to_tensor(t1), tf.convert_to_tensor(t2)


estimator = tf.estimator.Estimator(model_fn = model_fn)
for i in range(100):
    estimator.train(input_fn = my_input_fn, steps = 1)

