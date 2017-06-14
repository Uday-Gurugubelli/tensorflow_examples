import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
tf.reset_default_graph()

x=tf.placeholder(tf.float32, [None, 784])
y=tf.placeholder(tf.float32, [None, 10])
    
features = tf.reshape(x, [-1, 28, 28, 1])
    
conv1 = tf.layers.conv2d(inputs=features,
                           filters=16,
                           kernel_size=3,
                           padding='SAME')
conv1_pool = tf.layers.max_pooling2d(conv1, pool_size=[2,2], strides=2)
conv1_pool_norm = tf.nn.local_response_normalization(conv1_pool)

conv2 = tf.layers.conv2d(inputs=conv1_pool_norm,
                           filters=32,
                           kernel_size=3,
                           padding='SAME')
conv2_pool = tf.layers.max_pooling2d(conv2, pool_size=[2,2], strides=2)
conv2_pool_norm = tf.nn.local_response_normalization(conv2_pool)

conv3 = tf.layers.conv2d(inputs=conv2_pool_norm,
                           filters=64,
                           kernel_size=2,
                           padding='SAME')
conv4 = tf.layers.conv2d(inputs=conv3,
                           filters=128, 
                           kernel_size=2,
                           padding='SAME')
conv5 = tf.layers.conv2d(inputs=conv4,
                           filters=256,
                           kernel_size=2,
                           padding='SAME')
conv5_pool = tf.layers.max_pooling2d(conv5, pool_size=[2,2], strides=2)
conv5_pool_norm = tf.nn.local_response_normalization(conv5_pool)

conv5_pool_norm_flat = tf.reshape(conv5_pool_norm, [-1, 3*3*256]) 

dense1 = tf.layers.dense(inputs=conv5_pool_norm_flat, units=1024)
dense1_drop = tf.layers.dropout(inputs=dense1, rate=0.5)

dense2 = tf.layers.dense(inputs=dense1_drop, units=1024)
dense2_drop = tf.layers.dropout(inputs=dense2, rate=0.5)

dense_logits = tf.layers.dense(inputs=dense2_drop, units=10,
                                  activation=None)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                        labels=y, logits=dense_logits))

train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)   
                        
init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    for i in range(100):
        xb, yb = mnist.train.next_batch(100)
        lss, _ = sess.run([loss,train_op], feed_dict={x:xb, y:yb})
        print("step", i, "loss: ", lss)

