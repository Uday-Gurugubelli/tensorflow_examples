import numpy as np
import tensorflow as tf
imort matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

tf.logging.set_verbosity(tf.logging.INFO)
tf.reset_default_graph()

FALGS=None
img_dim = 784
z_dim = 100
batch=100
test_batch = 20
niter=10000

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

def dg(f):

    gen_smp = generator(f['zf'], reuse=False)
    gen_smp = tf.stop_gradient(gen_smp)
    disc_fake = discriminator(gen_smp, reuse=False)
    disc_real = discriminator(f['zr'], reuse=True)

    gen_smp_g = generator(f['zf'], reuse=True)
    disc_gen_g = discriminator(gen_smp_g, reuse=True)
    
    return disc_real, disc_fake, disc_gen_g, gen_smp

def model_g(features, labels, mode):

    a, b, disc_fake_g, c = dg(features) 
        
    train_op = None
    loss = tf.convert_to_tensor(0.)
    predictions = None
    eval_metric_ops = None
    global_step = tf.train.get_global_step()
    if(mode == tf.estimator.ModeKeys.EVAL or
        mode == tf.estimator.ModeKeys.TRAIN):

        l_real = tf.ones([batch, 1])

        loss_gen = tf.losses.sigmoid_cross_entropy(l_real, disc_fake_g)
        loss = loss_gen+tf.losses.get_regularization_loss(scope='Generator')

        train_op = tf.train.GradientDescentOptimizer(0.1).minimize(
                              loss, global_step = global_step)
            
    if(mode == tf.estimator.ModeKeys.PREDICT):
        a, b, c, pred_sample = dg(features)
        predictions = {"pred_sample": pred_sample}
        
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss,
                                        train_op=train_op, predictions=predictions,
                                        eval_metric_ops = eval_metric_ops)                                                                        

def model_d(features, labels, mode):
     
    disc_real, disc_fake, c, d = dg(features)

    l_real = tf.ones([batch, 1])
    l_fake = tf.zeros([batch, 1])

    train_op = None
    loss = tf.convert_to_tensor(0.)
    predictions = None
    eval_metric_ops = None
    global_step = tf.train.get_global_step()
    if(mode == tf.estimator.ModeKeys.EVAL or
            mode == tf.estimator.ModeKeys.TRAIN):

        loss_real = tf.losses.sigmoid_cross_entropy(l_real, disc_real)
        loss_fake = tf.losses.sigmoid_cross_entropy(l_fake, disc_fake)
        loss_disc = loss_real+loss_fake
        loss = loss_disc+tf.losses.get_regularization_loss(scope='Discriminator')

        train_op = tf.train.GradientDescentOptimizer(0.1).minimize(
                              loss, global_step = global_step)
        
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss,
                                        train_op=train_op, predictions=predictions,
                                        eval_metric_ops = eval_metric_ops)

def input_fn():
    zf = tf.random_uniform([batch, f_dim], 0, 1)
    zr, t2 = mnist.train.next_batch(batch)
    zr = tf.reshape(zr, [batch, img_dim])
    x = {'zr':zr, 'zf':zf}
    y = None
    return x, y

def plt_ip_fn():
    zf = np.random.uniform(0., 1., [test_batch, f_dim]).astype(np.float32)
    zr = np.random.uniform(0., 1., [test_batch, img_dim]).astype(np.float32)
    x = {'zr':zr, 'zf':zf}
    y = None
    return tf.estimator.inputs.numpy_input_fn(x, y, batch_size=test_batch, num_epochs=1,
                                            shuffle=False)

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
