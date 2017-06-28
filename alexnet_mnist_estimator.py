import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

tf.reset_default_graph()
tf.logging.set_verbosity(tf.logging.INFO)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
global_step = tf.Variable(0, name="global_step", trainable=False)

def alexnet_model_fn(mode, features, labels):
    
    features = tf.reshape(features, [-1, 28, 28, 1])
    
    network = tf.layers.conv2d(inputs=features,
                           filters=16,
                           kernel_size=3,
                           padding='SAME')                               
    network = tf.layers.max_pooling2d(network, pool_size=[2,2], strides=2)
    network = tf.nn.local_response_normalization(network)
    
    network = tf.layers.conv2d(inputs=network,
                           filters=32,
                           kernel_size=3,
                           padding='SAME')                               
    network = tf.layers.max_pooling2d(network, pool_size=[2,2], strides=2)
    network = tf.nn.local_response_normalization(network)
    
    network = tf.layers.conv2d(inputs=network,
                           filters=64,
                           kernel_size=2,
                           padding='SAME')
    network = tf.layers.conv2d(inputs=network,
                           filters=128, 
                           kernel_size=2,
                           padding='SAME')

    network = tf.layers.conv2d(inputs=network,
                           filters=256,
                           kernel_size=2,
                           padding='SAME')
    network = tf.layers.max_pooling2d(network, pool_size=[2,2], strides=2)
    network = tf.nn.local_response_normalization(network)

    network_flat = tf.reshape(network, [-1, 3*3*256]) 

    dense_nw = tf.layers.dense(inputs=network_flat, units=1024)    
    dropout = tf.layers.dropout(inputs=dense_nw, rate=0.5)

    dense_nw = tf.layers.dense(inputs=dropout, units=1024)
    dropout = tf.layers.dropout(inputs=dense_nw, rate=0.5)

    dense_nw_logits = tf.layers.dense(inputs=dropout, units=10,
                                  activation=None)
    
    train_op=None
    global_step = tf.train.get_global_step()
    if (mode == tf.estimator.ModeKeys.EVAL or
             mode == tf.estimator.ModeKeys.TRAIN):
        
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                        labels=labels, logits=dense_nw_logits))    
    if (mode == tf.estimator.ModeKeys.TRAIN):
        train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss,
                                        global_step = global_step)
                        
    predictions = {"classes": tf.argmax(
                        input=labels, axis=1),
                        "probabilities": dense_nw_logits}
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions,
                        loss=loss, train_op=train_op)

def my_input_fn():
    x ,y = mnist.train.next_batch(100)
    x1 = tf.convert_to_tensor(x)
    y1 = tf.convert_to_tensor(y)
    return x1 ,y1

def my_input_fn_test():
    x ,y = mnist.test.next_batch(100)
    x1 = tf.convert_to_tensor(x)
    y1 = tf.convert_to_tensor(y)
    return x1 ,y1

mnist_classifier = tf.estimator.Estimator(
                            alexnet_model_fn,
                            model_dir=None)
for i in range(5):
    mnist_classifier.train(input_fn=my_input_fn, steps=10, max_steps=None)

mnist_classifier.evaluate(input_fn=my_input_fn_test, steps=1)
