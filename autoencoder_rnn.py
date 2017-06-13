import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

tf.reset_default_graph()

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])

batch= 1000

state1=tf.zeros([batch, 784])
state2=tf.zeros([batch, 1000])
state3=tf.zeros([batch, 500])
state4=tf.zeros([batch, 250])
state5=tf.zeros([batch, 30])
state6=tf.zeros([batch, 250])
state7=tf.zeros([batch, 500])
state8=tf.zeros([batch, 1000])

state=[state1, state2, state3, state4, state5, state6, state7, state8]

rbm = tf.contrib.rnn.BasicRNNCell(num_units=784,
                                    activation=tf.nn.sigmoid)
rbm1 = tf.contrib.rnn.BasicRNNCell(num_units=1000,
                                    activation=tf.nn.sigmoid)
rbm2 = tf.contrib.rnn.BasicRNNCell(num_units=500,
                                    activation=tf.nn.sigmoid)
rbm3 = tf.contrib.rnn.BasicRNNCell(num_units=250,
                                    activation=tf.nn.relu)
rbm4 = tf.contrib.rnn.BasicRNNCell(num_units=30,
                                    activation=tf.nn.sigmoid)
rbm5 = tf.contrib.rnn.BasicRNNCell(num_units=2500,
                                    activation=tf.nn.sigmoid)
rbm6 = tf.contrib.rnn.BasicRNNCell(num_units=500,
                                    activation=tf.nn.sigmoid)
rbm7 = tf.contrib.rnn.BasicRNNCell(num_units=1000,
                                    activation=tf.nn.relu)

rnn = tf.contrib.rnn.MultiRNNCell([rbm1, rbm2, rbm3, rbm4, rbm5, rbm6, rbm7, rbm])
op, state = rnn(x, state)

cost = tf.reduce_mean(-x*tf.log(op)-(1-x)*tf.log(1-op))

train_step = tf.train.GradientDescentOptimizer(
                learning_rate=0.001).minimize(cost)

init_op=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    for i in range(30):
        xb, yb = mnist.test.next_batch(batch)
        lss, t1, t2 = sess.run([cost,op,train_step], feed_dict={x:xb})
        print('loss=',lss)
        
