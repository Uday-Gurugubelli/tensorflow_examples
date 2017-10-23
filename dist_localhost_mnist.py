import argparse
import sys
import os
import multiprocessing as mp
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
tf.logging.set_verbosity(tf.logging.INFO)
tf.reset_default_graph()
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
FLAGS=None

def nn(features):
    x = tf.layers.dense(inputs=features, units=512, activation=tf.nn.relu)
    x = tf.nn.dropout(x, 0.8)
    x = tf.layers.dense(inputs=x, units=256, activation=tf.nn.relu)
    x = tf.nn.dropout(x, 0.8)
    y = tf.layers.dense(inputs=x, units=10, activation=tf.nn.softmax)
    return y

def task0():
    cluster = tf.train.ClusterSpec({"local": ["localhost:2222", "localhost:2223"]})
    server = tf.train.Server(cluster, job_name="local",
                                     task_index=0)
def task1():
    cluster = tf.train.ClusterSpec({"local": ["localhost:2222", "localhost:2223"]})
    server = tf.train.Server(cluster, job_name="local",
                                    task_index=1)
def main(_):
    #ps_hosts=FLAGS.ps_hosts.split(",")
    #worker_hosts=FLAGS.worker_hosts.split(",")
    #cluster = tf.train.ClusterSpec({"ps":ps_hosts, "worker":worker_hosts})
    mp.get_context('spawn')
                    
    if FLAGS.job_name == "local":
        
        p0=mp.Process(target=task0, args=())
        p0.start()
        p0.join()
        p1=mp.Process(target=task1, args=())
        p1.start()
        p1.join()
        
        with tf.device(tf.train.replica_device_setter(
                    worker_device="/job:local/task:0",
                            cluster=None)):
            feat, labels = mnist.train.next_batch(100)
            feat = tf.reshape(feat, [100, 784])
            labels  = tf.reshape(labels, [100,10])
            y = nn(feat)
            global_step = tf.contrib.framework.get_or_create_global_step()
            loss = tf.losses.softmax_cross_entropy(
                    onehot_labels=labels, logits=y)
            #train_op = tf.train.RMSPropOptimizer(0.01).minimize(
            #                loss, global_step = global_step)

        with tf.device(tf.train.replica_device_setter(
                    worker_device="/job:local/task:1",
                            cluster=None)):
            
            train_op = tf.train.RMSPropOptimizer(0.01).minimize(
                            loss, global_step = global_step)

            hooks =[tf.train.StopAtStepHook(last_step = 10000)]
        with tf.train.MonitoredTrainingSession(master='',#server.target,
                            is_chief=(FLAGS.task_index == 0),
                            checkpoint_dir="/tmp/train_logs",
                            hooks=hooks) as mon_sess:
            while not mon_sess.should_stop():
                mon_sess.run(train_op)

if __name__ == "__main__":
                                               
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  # Flags for defining the tf.train.ClusterSpec
  parser.add_argument(
      "--ps_hosts",
      type=str,
      default="",
      help="Comma-separated list of hostname:port pairs"
  )
  parser.add_argument(
      "--worker_hosts",
      type=str,
      default="",
      help="Comma-separated list of hostname:port pairs"
  )
  parser.add_argument(
      "--job_name",
      type=str,
      default="local",
      help="One of 'ps', 'worker', 'local' "
  )
  # Flags for defining the tf.train.Server
  parser.add_argument(
      "--task_index",
      type=int,
      default=0,
      help="Index of task within the job"
  )

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

                                               
