import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import pandas as pd
tf.reset_default_graph()
tf.logging.set_verbosity(tf.logging.INFO)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
global_step = tf.Variable(0, name="global_step", trainable=False)
def ip_fn():
    a=[1.]
    FLAGS=None
    file_q = tf.train.string_input_producer(["C:\\Users\\Uday\\Downloads\\train.csv"], shuffle=False)
    reader = tf.TextLineReader(skip_header_lines=1)
    key, val = reader.read(file_q)
    record_defaults = [a for i in range(785)]
    rev = tf.decode_csv(
                    val, record_defaults, field_delim=',')
    batch = 100
    min_after_dequeue=500
    capacity = min_after_dequeue + 3 * 100
    feat_id, feat_rev = tf.train.shuffle_batch([rev, rev], batch_size=batch,
                                    capacity = capacity,
                                    min_after_dequeue = min_after_dequeue,
                                    allow_smaller_final_batch=True)


    t = tf.reshape(feat_rev, [batch, 785])
    labels, features = tf.split(t, [1,784], 1)
    labels = tf.reshape(tf.one_hot(tf.cast(labels, tf.int32), 10), [batch, 10])

    return features, labels

def ip_fn_infer():
    a=[1.]
    FLAGS=None
    file_q = tf.train.string_input_producer(["C:\\Users\\Uday\\Downloads\\test.csv"], shuffle=False)
    reader = tf.TextLineReader(skip_header_lines=1)
    key, val = reader.read(file_q)
    record_defaults = [a for i in range(784)]
    rev = tf.decode_csv(
                    val, record_defaults, field_delim=',')
    batch = 100
    min_after_dequeue=500
    capacity = min_after_dequeue + 3 * 100
    feat_id, feat_rev = tf.train.shuffle_batch([rev, rev], batch_size=batch,
                                    capacity = capacity,
                                    min_after_dequeue = min_after_dequeue,
                                    allow_smaller_final_batch=True)


    features = tf.reshape(feat_rev, [batch, 784])
    labels = None
    return features, labels

'''
with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    #t = tf.string_split(rev)
    #t = tf.convert_to_tensor(feat_rev, tf.float32)
    t = tf.reshape(feat_rev, [batch, 785])
    labels, features = tf.split(t, [1,784], 1)
    print(features)
    
    
    coord.request_stop()
    coord.join(threads)
'''

def alexnet_model_fn(features, labels, mode):
    
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
    lss=0.
    loss=tf.convert_to_tensor(lss)
    predictions=None
    eval_metric_ops = None
    global_step = tf.train.get_global_step()
    if (mode == tf.estimator.ModeKeys.EVAL or
             mode == tf.estimator.ModeKeys.TRAIN):
        
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                        labels=labels, logits=dense_nw_logits))    

    if (mode == tf.estimator.ModeKeys.TRAIN):
        train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss,
                                        global_step = global_step)

    if (mode == tf.estimator.ModeKeys.PREDICT or
             mode == tf.estimator.ModeKeys.EVAL):                   
        predictions = {"pred_labels": tf.argmax(dense_nw_logits, axis=0)}

    if (mode == tf.estimator.ModeKeys.EVAL):
        eval_metric_ops = {"accuracy":
                           tf.metrics.accuracy(tf.argmax(labels),
                                                tf.argmax(dense_nw_logits))}

    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions,
                        loss=loss, train_op=train_op,
                                      eval_metric_ops=eval_metric_ops)

def my_ip_fn():
    x ,y = mnist.train.next_batch(1000)
    x1 = tf.convert_to_tensor(x)
    y1 = tf.convert_to_tensor(y)
    return x1 ,y1

def ip_fn_test():
    x ,y = mnist.test.next_batch(1000)
    x1 = tf.convert_to_tensor(x)
    y1 = None #tf.convert_to_tensor(y)
    return x1 ,y1

mnist_classifier = tf.estimator.Estimator(
                            model_fn = alexnet_model_fn,
                            model_dir="C:\\tf_examples\\kagle_mnist_model")
for i in range(10):
    mnist_classifier.train(input_fn = my_ip_fn, steps=1, max_steps=None)
    mnist_classifier.evaluate(input_fn = my_ip_fn, steps=1)

pred = mnist_classifier.predict(input_fn=ip_fn_test)

for i, p in enumerate(pred):
    print(i+1,",", p["pred_labels"])
    
