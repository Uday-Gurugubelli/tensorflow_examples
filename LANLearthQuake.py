# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
#tf.enable_eager_execution()
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import sys
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
def input_fn():
    file = "../input/train.csv"
    record_defaults = [tf.float32]*2
    dataset = tf.data.experimental.CsvDataset(file, record_defaults, header=True)
    dataset = dataset.batch(150000)
    itr  = dataset.make_one_shot_iterator()
    data = itr.get_next()
    tf.print("data:",data)
    return data
    
def nn(x, mode):
    x = tf.reshape(x,(-1,150000))
    x = tf.layers.dense(inputs=x, units=1000, activation=tf.nn.tanh,
				#kernel_initializer=None, #tf.random_normal_initializer(0, 1),
				#bias_initializer=tf.zeros_initializer(),
		                #kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0005),
		                #bias_regularizer=None
				)#tf.contrib.layers.l2_regularizer(0.001))
    x = tf.layers.batch_normalization(x, training=mode==tf.estimator.ModeKeys.TRAIN)
    x = tf.layers.dropout(x, 0.70)
    x = tf.layers.dense(inputs=x, units=1000, activation=tf.nn.relu,
				#kernel_initializer=None, #tf.random_normal_initializer(0, 1),
				#bias_initializer=tf.zeros_initializer(),
		                #kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0005),
		                #bias_regularizer=None
				)#tf.contrib.layers.l2_regularizer(0.001))
    x = tf.layers.batch_normalization(x, training=mode==tf.estimator.ModeKeys.TRAIN)
    x = tf.layers.dropout(x, 0.70)
    x = tf.layers.dense(inputs=x, units=1000, activation=tf.nn.relu,
				#kernel_initializer=None, #tf.random_normal_initializer(0, 1),
				#bias_initializer=tf.zeros_initializer(),
		                #kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0005),
		                #bias_regularizer=None
				)#tf.contrib.layers.l2_regularizer(0.001))
    x = tf.layers.batch_normalization(x, training=mode==tf.estimator.ModeKeys.TRAIN)
    x = tf.layers.dense(inputs=x, units=1, activation=None)#tf.nn.softmax)
    return x

def model_fn(features, labels, mode):
    
    y=nn(features,mode)
    train_op=None
    loss=tf.convert_to_tensor(0.)
    predictions=None
    eval_metric_ops=None
    global_step=tf.train.get_global_step()
    print("global_step:", global_step)
    if(mode == tf.estimator.ModeKeys.EVAL or
        mode == tf.estimator.ModeKeys.TRAIN):
            labels = tf.reshape(labels, (150000,1))
            loss = tf.losses.absolute_difference(labels=labels, predictions=y) + tf.losses.get_regularization_loss()
	        
    if(mode == tf.estimator.ModeKeys.TRAIN):
        lr = tf.train.exponential_decay(0.01, global_step, 1000, 0.98000, staircase=False)
        train_op = tf.train.AdamOptimizer(lr).minimize(loss, global_step = global_step)
	
    if(mode == tf.estimator.ModeKeys.PREDICT):
        predictions = {"predictions": y}
    if(mode == tf.estimator.ModeKeys.EVAL):         
        eval_metric_ops = {"mean_absolute_error": tf.metrics.mean_absolute_error(labels, y)}
   	    
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, 
            train_op=train_op, predictions=predictions, eval_metric_ops = eval_metric_ops)

config = tf.estimator.RunConfig(keep_checkpoint_max=1)
regressor = tf.estimator.Estimator(model_fn = model_fn, model_dir="/var/tmp/model_dnn", config=config)

#for _ in range(10):                        
regressor.train(input_fn = input_fn, steps= 4000)
rslts=regressor.evaluate(input_fn = input_fn, steps= 10)
print(rslts)
submission = pd.read_csv('../input/sample_submission.csv', index_col='seg_id', dtype={"time_to_failure": np.float32})

# Load each test data, create the feature matrix, get numeric prediction

def pred_input_fn():
    test_files = "../input/test/"+submission.index.values+".csv"
    #print(test_files)
    record_defaults = [tf.float32]
    dataset = tf.data.experimental.CsvDataset(test_files, record_defaults, header=True)
    dataset = dataset.batch(150000)
    itr = dataset.make_one_shot_iterator()
    data = itr.get_next()
    return data, None

pred = regressor.predict(input_fn=pred_input_fn)

#print(next(pred))
for i,p in enumerate(pred):
    [pp]=p["predictions"]
    print(p,pp)
    submission.time_to_failure[i] = pp
    
submission.head()
# Save
submission.to_csv('submission.csv')


