'''
This code implements the G. E Hinton and R. R. Salakhutdinov Science July 2006.

As said, 2-layer RBM are trained firest as the process of pre-training and
then trained together, given that the weights are initilized with
pre-trained weights.

'''
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

tf.reset_default_graph()
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

'''
This file contains 2-layer RBM modules. For the convinence and to redice the number of
variable declerations, few extra definitions are added
'''

class rbm:
    def __init__(self, ip_layer, ip_dim, op_dim):
        self.ip_dim = ip_dim
        self.op_dim = op_dim
        self.ip_layer = ip_layer
        self.weights = tf.Variable(tf.random_normal([ip_dim, op_dim],
                                           mean=0, stddev=0.01))
        self.bias_v = tf.Variable(tf.zeros([ip_dim]))
        self.bias_h = tf.Variable(tf.zeros([op_dim]))
        
        self.op_layer = tf.nn.softmax(tf.nn.sigmoid(
                    tf.matmul(self.ip_layer, self.weights) + self.bias_h))
        ''' Gibbs sampling step for k = 1 '''
        self.conf_ip_layer = tf.nn.softmax(tf.nn.sigmoid(tf.matmul(
                    self.op_layer, tf.transpose(self.weights))+self.bias_v))

        self.conf_op_layer = tf.nn.softmax(tf.nn.sigmoid(
                    tf.matmul(self.conf_ip_layer, self.weights) + self.bias_h))

    #''' to use with out optimizer
    def update(self):
        delta_w = 0.001*tf.subtract(tf.matmul(
                    self.ip_layer, self.op_layer), tf.matmul(
                        self.conf_ip_layer, self.conf_op_layer))
        delta_bh = 0.001*tf.reduce_mean(tf.subtract(
                        self.op_layer, self.conf_op_layer), axis=1)
        delta_bv = 0.001*tf.reduce_mean(tf.subtract(
                        self.ip_layer, self.conf_ip_layer), axis=1)

        self.weights.assign_add(delta_w)
        self.bias_h.assign_add(delta_bh)
        self.bias_v.assign_add(delta_bv)
        return self.weights
     # '''
        
    def get_op_layer(self):
        return self.op_layer
    
    def get_weights(self):
        return self.weights
    
    def get_biases(self):
        return self.bias_v, self.bias_h

    def free_energy(self, sample):
        wxb = tf.matmul(sample, self.weights) + self.bias_h
        b_term = tf.tensordot(sample, self.bias_v, axes=1)
        hid_term = tf.reduce_sum(tf.log(1+tf.exp(wxb)), axis=1)
        return -hid_term - b_term 

    ''' train step is returned to cross check the results '''
    ''' cost is given as difference between the free energy of the input units
        and the free energy of the reconstructed image units
    '''
    def train_step(self):
        cost = tf.reduce_mean(self.free_energy(
                    self.ip_layer) - self.free_energy(self.conf_ip_layer))
        
        train_step = tf.train.GradientDescentOptimizer(
                        learning_rate=0.001).minimize(cost)
        return cost, train_step

x = tf.placeholder(tf.float32, [None, None])

''' 2-layer rbm instances for pre_training'''

rbm1 = rbm(x, 784, 1000)
rbm2 = rbm(rbm1.op_layer, 1000, 500)
rbm3 = rbm(rbm2.op_layer, 500, 250)
rbm4 = rbm(rbm3.op_layer, 250, 30)

''' Pre-training of the RBMs with the option of selecting the GSD or updates'''

iter = 10
iter_out = 5
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    for i in range(iter):
        xb, yb = mnist.train.next_batch(1000)
        loss1, step1 = rbm1.train_step()
        lss, _ = sess.run([loss1, rbm1.update()], feed_dict={x:xb})
        if(i%iter_out == 0):
            print('rmb1: loss=',' %.5f' %lss, '\t step:', '%d' %i)

    for i in range(iter):
        xb, yb = mnist.train.next_batch(1000)
        loss2, step2 = rbm2.train_step()
        lss, _ = sess.run([loss2, rbm2.update()], feed_dict={x:xb})
        if(i%iter_out == 0):
            print('rmb2: loss=',' %.5f' %lss, '\t step:', '%d' %i)

    for i in range(iter):
        xb, yb = mnist.train.next_batch(1000)
        loss3, step3 = rbm3.train_step()
        lss, _ = sess.run([loss3, rbm3.update()], feed_dict={x:xb})
        if(i%iter_out == 0):
            print('rmb3: loss=',' %.5f' %lss, '\t step:', '%d' %i) 
    
    for i in range(iter):
        xb, yb = mnist.train.next_batch(1000)
        loss4, step4 = rbm4.train_step()
        lss, _ = sess.run([loss4, rbm4.update()], feed_dict={x:xb})
        if(i%iter_out == 0):
            print('rmb4: loss=',' %.5f' %lss, '\t step:', '%d' %i) 

layer1 = tf.nn.sigmoid(tf.add(
                    tf.matmul(x, rbm1.weights) , rbm1.bias_h))
layer2 = tf.nn.sigmoid(tf.add(
                    tf.matmul(layer1, rbm2.weights) , rbm2.bias_h))
layer3 = tf.nn.sigmoid(tf.add(
                    tf.matmul(layer2, rbm3.weights) , rbm3.bias_h))
''' layer with 30 units are not logistic as per the paper '''
layer4 = tf.add(tf.matmul(layer3, rbm4.weights) , rbm4.bias_h)
 
sample3 = tf.nn.sigmoid(tf.add(tf.matmul(
                    layer4, tf.transpose(rbm4.weights)),rbm4.bias_v))
sample2 = tf.nn.sigmoid(tf.add(tf.matmul(
                    sample3, tf.transpose(rbm3.weights)),rbm3.bias_v))
sample1 = tf.nn.sigmoid(tf.add(tf.matmul(
                    sample2, tf.transpose(rbm2.weights)),rbm2.bias_v))
x_sample = tf.nn.sigmoid(tf.add(tf.matmul(
                    sample1, tf.transpose(rbm1.weights)),rbm1.bias_v))

'''Diff of the cross entropy of the input image and the
reconstructed image'''

cost = tf.reduce_mean(-x*tf.log(x_sample)-(1-x)*tf.log(1-x_sample))

train_step = tf.train.GradientDescentOptimizer(
                learning_rate=0.01).minimize(cost)

print("\n Autoendocer fine tuning...")

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    for i in range(iter):
        xb, yb = mnist.train.next_batch(1000)
        lss, _ = sess.run([cost, train_step], feed_dict={x:xb})
        if(i%iter_out == 0):
            print('Autoencoder: loss=',' %.5f' %lss, '\t step:', '%d' %i)
    xt, _ = mnist.test.next_batch(1000)
    accuracy = sess.run(cost, feed_dict={x:xt})
    print("Autoencoder Accuracy on test images :",'%f' %accuracy)
    
