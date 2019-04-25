import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

tf.logging.set_verbosity(tf.logging.INFO)
tf.reset_default_graph()

FALGS=None
img_dim = 784
z_dim = 784
batch=100
test_batch = 20
niter=10000

mnist = input_data.read_data_sets("MNIST_data/")
test_data = np.random.uniform(0., 1., [test_batch, z_dim]).astype(np.float32)

def generator(ip, trainable=False):
    
    net = tf.layers.dense(inputs=ip, units=512, activation=tf.nn.tanh, 
                    trainable=trainable)
    net = tf.layers.dense(inputs=net, units=1024, activation=tf.nn.tanh, 
                    trainable=trainable)
    net = tf.layers.dense(inputs=net, units=img_dim, activation=tf.nn.sigmoid, 
                    trainable=trainable)    
    return net
    
def discriminator(ip, trainable=False):
    
    net = tf.layers.dense(inputs=ip, units=512, activation=tf.nn.tanh, 
                    trainable=trainable)
    net = tf.layers.dense(inputs=net, units=1024, activation=tf.nn.tanh,
                    trainable=trainable)
    net = tf.layers.dense(inputs=net, units=1, activation=tf.nn.sigmoid,
                    trainable=trainable)
    return net

def dg(f):

    gen_smp = generator(f, True)
    disc_fake = discriminator(gen_smp, False)
    return gen_smp, disc_fake
  
def disc(f):

    gen_smp = generator(f, False)
    disc_fake = discriminator(gen_smp, True)
    return gen_smp, disc_fake

def input_disc():
    zf = tf.random_uniform([batch, z_dim], 0, 1)
    zr, t2 = mnist.train.next_batch(batch)
    zr = tf.reshape(zr, [batch, img_dim])
    x = tf.concat([zr, zf], 0)
    print(x)
    y1 = [0.9 for _ in range(batch)]
    y2 = tf.zeros([batch,1])
    print(y2)
    y = tf.concat([tf.reshape(y1, (100,1)),y2], 0)
    print(y)
    return x, y

def input_gan():
    x = tf.random_uniform([batch, z_dim], 0, 1)
    y = tf.ones([batch, 1])
    print(x,y)
    return x, y

def model_g(features, labels, mode):

    gen_imgs, disc_fake_g = dg(features) 
        
    train_op = None
    loss = tf.convert_to_tensor(0.)
    predictions = None
    eval_metric_ops = None
    global_step = tf.train.get_global_step()
    if(mode == tf.estimator.ModeKeys.EVAL or
        mode == tf.estimator.ModeKeys.TRAIN):

        loss_gen = tf.losses.sigmoid_cross_entropy(labels, disc_fake_g)
        loss = loss_gen+tf.losses.get_regularization_loss()

        train_op = tf.train.AdamOptimizer(0.001).minimize(
                              loss, global_step = global_step)
            
    if(mode == tf.estimator.ModeKeys.PREDICT):
        predictions = {"predictions": gen_imgs}
        
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss,
                                        train_op=train_op, predictions=predictions,
                                        eval_metric_ops = eval_metric_ops)   

def model_d(features, labels, mode):
     
    _, disc_fake = disc(features)

    train_op = None
    loss = tf.convert_to_tensor(0.)
    predictions = None
    eval_metric_ops = None
    global_step = tf.train.get_global_step()
    if(mode == tf.estimator.ModeKeys.EVAL or
            mode == tf.estimator.ModeKeys.TRAIN):

        loss_disc = tf.losses.sigmoid_cross_entropy(labels, disc_fake)
        loss = loss_disc+tf.losses.get_regularization_loss()

        train_op = tf.train.AdamOptimizer(0.001).minimize(
                              loss, global_step = global_step)
        
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss,
                                        train_op=train_op, predictions=predictions,
                                        eval_metric_ops = eval_metric_ops)

def main(_):
    dir_d = os.path.join(FLAGS.model_dir, "gan_d_model")
    dir_g = os.path.join(FLAGS.model_dir, "gan_g_model")
    
    est_d = tf.estimator.Estimator(model_fn = model_d, model_dir = dir_d)
    est_g = tf.estimator.Estimator(model_fn = model_g, model_dir = dir_g)
    for i in range(niter):
        est_d.train(input_fn = input_fn, steps = 10)
        est_g.train(input_fn = input_fn, steps = 10)
    
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
