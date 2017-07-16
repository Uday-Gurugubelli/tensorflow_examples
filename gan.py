import numpy as np
import tensorflow as tf
imort matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

tf.logging.set_verbosity(tf.logging.INFO)
tf.reset_default_graph()

FALGS=None
img_dim = 784
z_dim = 784
batch=100
test_batch = 20
niter=500

mnist = input_data.read_data_sets("MNIST_data/")
test_data = np.random.uniform(0., 1., [test_batch, z_dim]).astype(np.float32)

def generator(ip, reuse=False):
    with tf.variable_scope('Generator', reuse=reuse):
        net = tf.layers.dense(inputs=ip, units=512, activation=tf.nn.tanh,
                        kernel_initializer= tf.random_uniform_initializer(-1,1),
                        bias_initializer=tf.zeros_initializer(),
                        kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001),
                        bias_regularizer=None)
        net = tf.layers.dense(inputs=net, units=1024, activation=tf.nn.tanh,
                        kernel_initializer= tf.random_uniform_initializer(-1,1),
                        bias_initializer=tf.zeros_initializer(),
                        kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001),
                        bias_regularizer=None)
        net = tf.layers.dense(inputs=net, units=img_dim, activation=tf.nn.sigmoid)
        return net
def discriminator(ip, reuse=False):
    with tf.variable_scope('Discriminator', reuse=reuse):
        net = tf.layers.dense(inputs=ip, units=512, activation=tf.nn.tanh, 
                        kernel_initializer= tf.random_uniform_initializer(-1,1),
                        bias_initializer=tf.zeros_initializer(),
                        kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001),
                        bias_regularizer=None)
        net = tf.layers.dense(inputs=net, units=1024, activation=tf.nn.tanh,
                        kernel_initializer= tf.random_uniform_initializer(-1,1),
                        bias_initializer=tf.zeros_initializer(),
                        kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001),
                        bias_regularizer=None)
        net = tf.layers.dense(inputs=net, units=1, activation=tf.nn.sigmoid)
        return net

def model_fn(features, labels, mode):

    fake = tf.random_uniform([batch, z_dim], 0, 1)
    
    gen_sample = generator(fake)
    gen_samp = tf.stop_gradient(gen_sample)
    disc_fake = discriminator(gen_sample)
    disc_real = discriminator(features, reuse=True)
    
    gen_sample_g = generator(fake, reuse=True)
    disc_fake_g = discriminator(gen_sample_g, reuse=True)
    
    train_op = None
    predictions = None
    loss = tf.convert_to_tensor(0.)
    global_step = tf.train.get_global_step()
    if(mode == tf.estimator.ModeKeys.EVAL or
            mode == tf.estimator.ModeKeys.TRAIN):
        loss_disc = -tf.reduce_mean(tf.log(disc_real) + tf.log(1. - disc_fake))
        loss_gen = -tf.reduce_mean(tf.log(disc_fake_g))
        loss = loss_disc+loss_gen+tf.losses.get_regularization_loss()
      
    if(mode == tf.estimator.ModeKeys.TRAIN):
        train_op = tf.train.GradientDescentOptimizer(0.001).minimize(
                            loss, global_step = global_step)
    if(mode == tf.estimator.ModeKeys.PREDICT):  
        predictions = {"pred ": tf.convert_to_tensor(gen_sample_g)}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss,
                                    train_op=train_op, predictions=predictions)

def input_fn():
    t1, t2 = mnist.train.next_batch(batch)
    return tf.convert_to_tensor(t1), tf.convert_to_tensor(t2)

def plt_ip_fn():
    global test_data
    [t] = tf.train.slice_input_producer([test_data], num_epochs=1)
    x = tf.train.shuffle_batch([t], batch_size=test_batch, capacity = 500,
                                        min_after_dequeue=100)
    y = None
    return x, y

def main(_):
    estimator = tf.estimator.Estimator(model_fn = model_fn)
    for i in range(niter):
        estimator.train(input_fn = input_fn, steps = 100)
    
    pred = estimator.predict(input_fn = plt_ip_fn)
    op = []
    for i, p in enumerate(pred):
        op.append(p["pred"])
        print(p["pred"])
    
    f, a = plt.subplots(2, 10, figsize=(10, 4))

    for i in range(10):
        for j in range(2):
            a[j][i].imshow(np.reshape(op[i], (28, 28)))
    f.show()
    plt.draw()
    plt.waitforbuttonpress()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str,
                        default='/tmp/gan_model',
                        help='path to model dir')
    FLAGS, unparsed = parser.parse_known_args()

tf.app.run()
