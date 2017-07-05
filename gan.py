import numpy as np
import tensorflow as tf
imort matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

tf.logging.set_verbosity(tf.logging.INFO)
tf.reset_default_graph()

FALGS=None
img_dim = 784
batch=100
niter=100

mnist = input_data.read_data_sets("MNIST_data/")
test_data = np.random.uniform(0., 1., [batch, img_dim])

def generator(ip):
    net = tf.layers.dense(inputs=ip, units=512, activation=tf.nn.relu)
    net = tf.layers.dense(inputs=net, units=1024, activation=tf.nn.relu)
    net = tf.layers.dense(inputs=net, units=img_dim, activation=tf.nn.sigmoid)
    return net
def discriminator(x):
    x = tf.layers.dense(inputs=x, units=512, activation=tf.nn.relu)
    x = tf.layers.dense(inputs=x, units=1024, activation=tf.nn.relu)
    x = tf.layers.dense(inputs=x, units=1, activation=tf.nn.sigmoid)
    return x

def model_fn(features, labels, mode):

    gen_sample = generator(features)
    disc_fake = discriminator(gen_sample)
    disc_real = discriminator(gen_sample)
    
    train_op = None
    predictions = None
    loss = tf.convert_to_tensor(0.)
    global_step = tf.train.get_global_step()
    if(mode == tf.estimator.ModeKeys.EVAL or
            mode == tf.estimator.ModeKeys.TRAIN):
        loss = -tf.reduce_mean(tf.log(disc_real) + tf.log(1. - disc_fake))
      
    if(mode == tf.estimator.ModeKeys.TRAIN):
        train_op = tf.train.GradientDescentOptimizer(0.01).minimize(
                            loss, global_step = global_step)
    gen_pred = generator(features)    
    predictions = {"pred_labels ": tf.convert_to_tensor(gen_pred)}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss,
                                    train_op=train_op, predictions=predictions)

def input_fn():
    t1, t2 = mnist.train.next_batch(batch)
    return tf.convert_to_tensor(t1), tf.convert_to_tensor(t2)

def plt_ip_fn():
    [t] = tf.train.slice_input_producer([test_data])
    x = tf.train.shuffle_batch([t], batch_size=batch, capacity = 500,
                                        min_after_dequeue=100)
    y = None
    return x, y

def main(_):
    estimator = tf.estimator.Estimator(model_fn = model_fn)
    for i in range(niter):
        estimator.train(input_fn = input_fn, steps = 1)
        
    estimator.evaluate(input_fn = input_fn, steps = 1)
    
    pred = estimator.predict(input_fn = plt_ip_fn)
    for i, p in enumerate(pred):
        print(p["labels"])
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str,
                        default='/tmp/gan_model',
                        help='path to model dir')
    FLAGS, unparsed = parser.parse_known_args()
    
    
    main(sys.argv[0])
