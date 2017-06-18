import tensorflow as tf
import tflearn
import tflearn.datasets.oxflower17 as oxflower17

tf.reset_default_graph()
tf.logging.set_verbosity(tf.logging.INFO)

X, Y = oxflower17.load_data(one_hot=True)

GLOBAL_STEP = tf.Variable(0, name="global_step", trainable=False)

def vggnet_model_fn(mode, features, labels):
    
    features = tf.reshape(features, [-1, 224, 224, 3])
    
    network = tf.layers.conv2d(inputs=features,
                           filters=64,
                           kernel_size=3,
                           padding='SAME')
    network = tf.layers.conv2d(inputs=features,
                           filters=64,
                           kernel_size=3,
                           padding='SAME')
    network = tf.layers.max_pooling2d(network, pool_size=[2,2], strides=2)
    
    
    network = tf.layers.conv2d(inputs=network,
                           filters=128,
                           kernel_size=3,
                           padding='SAME')
    network = tf.layers.conv2d(inputs=network,
                           filters=128,
                           kernel_size=3,
                           padding='SAME')
    network = tf.layers.max_pooling2d(network, pool_size=[2,2], strides=2)
    
    network = tf.layers.conv2d(inputs=network,
                           filters=256,
                           kernel_size=3,
                           padding='SAME')
    network = tf.layers.conv2d(inputs=network,
                           filters=256, 
                           kernel_size=3,
                           padding='SAME')
    network = tf.layers.conv2d(inputs=network,
                           filters=256, 
                           kernel_size=3,
                           padding='SAME')
    network = tf.layers.max_pooling2d(network, pool_size=[2,2], strides=2)
    
    network = tf.layers.conv2d(inputs=network,
                           filters=512,
                           kernel_size=3,
                           padding='SAME')
    network = tf.layers.conv2d(inputs=network,
                           filters=512,
                           kernel_size=3,
                           padding='SAME')
    network = tf.layers.conv2d(inputs=network,
                           filters=512,
                           kernel_size=3,
                           padding='SAME')
    network = tf.layers.max_pooling2d(network, pool_size=[2,2], strides=2)

    network = tf.layers.conv2d(inputs=network,
                           filters=512,
                           kernel_size=3,
                           padding='SAME')
    network = tf.layers.conv2d(inputs=network,
                           filters=512,
                           kernel_size=3,
                           padding='SAME')
    network = tf.layers.conv2d(inputs=network,
                           filters=512,
                           kernel_size=3,
                           padding='SAME')
    network = tf.layers.max_pooling2d(network, pool_size=[2,2], strides=2)


    network_flat = tf.reshape(network, [-1, 7*7*512]) 

    dense_nw = tf.layers.dense(inputs=network_flat, units=4096)    
    dropout = tf.layers.dropout(inputs=dense_nw, rate=0.5)

    dense_nw = tf.layers.dense(inputs=dropout, units=4096)
    dropout = tf.layers.dropout(inputs=dense_nw, rate=0.5)

    dense_nw_logits = tf.layers.dense(inputs=dropout, units=17,
                                  activation=None)
    
    train_op=None
    GLOBAL_STEP = tf.train.get_global_step()
    if (mode == tf.estimator.ModeKeys.EVAL or
             mode == tf.estimator.ModeKeys.TRAIN):
        
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                        labels=labels, logits=dense_nw_logits))    
    if (mode == tf.estimator.ModeKeys.TRAIN):
        train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss,
                                        global_step = GLOBAL_STEP)
                        
    predictions = {"classes": tf.argmax(
                        input=labels, axis=1),
                        "probabilities": dense_nw_logits}
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions,
                        loss=loss, train_op=train_op)

def my_input_fn():
    x1 = tf.convert_to_tensor(X)
    y1 = tf.convert_to_tensor(Y)
    return x1 ,y1

mnist_classifier = tf.estimator.Estimator(
                            model_fn = vggnet_model_fn,
                            model_dir=None)
for i in range(5):
    mnist_classifier.train(input_fn=my_input_fn, steps=100, max_steps=None)
'''
def my_input_fn_test():
    x1 = tf.convert_to_tensor(X)
    y1 = tf.convert_to_tensor(Y)
    return x1 ,y1

mnist_classifier.evaluate(input_fn=my_input_fn_test, steps=1)
'''
