# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import gc
import os
print(os.listdir("../input"))
import tensorflow as tf

# Any results you write to the current directory are saved as output.
# Import data
sales = pd.read_csv('../input/sales_train.csv', parse_dates=['date'], infer_datetime_format=True, dayfirst=True)
shops = pd.read_csv('../input/shops.csv')
items = pd.read_csv('../input/items.csv')
cats = pd.read_csv('../input/item_categories.csv')
val = pd.read_csv('../input/test.csv')

# Rearrange the raw data to be monthly sales by item-shop
df = sales.groupby([sales.date.apply(lambda x: x.strftime('%Y-%m')),'item_id','shop_id']).sum().reset_index()
df = df[['date','item_id','shop_id','item_cnt_day']]
print(df)
df["item_cnt_day"].clip(0.,20.,inplace=True)
df = df.pivot_table(index=['item_id','shop_id'], columns='date',values='item_cnt_day',fill_value=0).reset_index()
#print("df:", df)
# Merge data from monthly sales to specific item-shops in test data
test = pd.merge(val,df,on=['item_id','shop_id'], how='left').fillna(0)

# Strip categorical data so keras only sees raw timeseries
test = test.drop(labels=['ID','item_id','shop_id'],axis=1)

# Rearrange the raw data to be monthly average price by item-shop
# Scale Price
scaler = MinMaxScaler(feature_range=(0, 1))
sales["item_price"] = scaler.fit_transform(sales["item_price"].values.reshape(-1,1))
df2 = sales.groupby([sales.date.apply(lambda x: x.strftime('%Y-%m')),'item_id','shop_id']).mean().reset_index()
df2 = df2[['date','item_id','shop_id','item_price']].pivot_table(index=['item_id','shop_id'], columns='date',values='item_price',fill_value=0).reset_index()

# Merge data from average prices to specific item-shops in test data
price = pd.merge(val,df2,on=['item_id','shop_id'], how='left').fillna(0)
price = price.drop(labels=['ID','item_id','shop_id'],axis=1)

# Create x and y training sets from oldest data points
y_train = test['2015-10']
x_sales = test #.drop(labels=['2015-10'],axis=1)
x_sales = x_sales.values.reshape((x_sales.shape[0], x_sales.shape[1], 1))
x_prices = price #.drop(labels=['2015-10'],axis=1)
x_prices= x_prices.values.reshape((x_prices.shape[0], x_prices.shape[1], 1))
X = np.append(x_sales,x_prices,axis=2)

Y = y_train.values.reshape((214200, 1))
print("Training Predictor Shape: ",X.shape) ##214200, 33,2
print("Training Predictee Shape: ",Y.shape) ## 214200, 1

ffss=list()
llbbs=list()
test=list()
for j in range(214200):
    fs=list()
    lbs=list()
    test_fs=list()
    xx = X[j,:,:]
    for i in range(28):
        fs.append(xx[i:i+5,:])
        lbs.append(xx[6+i,0])
        test_fs.append(xx[i+1:i+6,:])
    ffss.append(fs)
    llbbs.append(lbs)
    test.append(test_fs)
XX = np.asarray(ffss)
YY = np.asarray(llbbs)
TEST = np.asarray(test)
print(XX.shape, YY.shape, TEST.shape)
import tensorflow as tf
#x = tf.placeholder(tf.float32, [None, 2])
#y = tf.placeholder(tf.float32, [None, 1])
#X = np.reshape(X, (33,214200,2))
def rnn(x):
    lstm = tf.nn.rnn_cell.LSTMCell(10, state_is_tuple=True, reuse=tf.AUTO_REUSE, activation=tf.nn.relu)
    #lstm1 = tf.nn.rnn_cell.LSTMCell(2,state_is_tuple=True, reuse=tf.AUTO_REUSE)
    stacked_lstm = tf.nn.rnn_cell.MultiRNNCell([lstm, lstm, lstm])#, lstm, lstm])
    initial_state = state = stacked_lstm.zero_state(1, tf.float32)
    x= tf.reshape(x, [1,28,10])
    #for i in range(28):
        #print(x.shape)
    #    x1 = tf.reshape(x[i,:], [1,5,2])
    output, state = tf.nn.dynamic_rnn(cell=stacked_lstm, inputs=x, initial_state=initial_state, sequence_length=[5])
    f_state = state
    print("output:", output)
    output = tf.reshape(output, (-1,280))
    dense_out = tf.layers.dense(inputs=output, units=28, activation=None ) #tf.nn.relu)
    return dense_out
    
def model_fn(features, labels, mode):
    y = rnn(features)
    y = tf.reshape(y, [1,28])
    train_op = None
    loss = tf.convert_to_tensor(0.)
    predictions = None
    eval_metric_ops = None
    global_step = tf.train.get_global_step()
    if(mode == tf.estimator.ModeKeys.EVAL or
            mode == tf.estimator.ModeKeys.TRAIN):
        labels=  tf.reshape(labels, [1,28])
        loss = tf.losses.absolute_difference(labels, y) + tf.losses.get_regularization_loss()
    if(mode == tf.estimator.ModeKeys.TRAIN):
        lr = tf.train.exponential_decay(0.01, global_step, 1000, 0.10000, staircase=False)
        train_op = tf.train.AdamOptimizer(lr).minimize(loss, global_step = global_step)
    if(mode == tf.estimator.ModeKeys.PREDICT):
    	predictions = {"predictions": y[27]}
    if(mode == tf.estimator.ModeKeys.EVAL):
        eval_metric_ops = {"mae error": tf.metrics.mean_absolute_error(labels,y)}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, 
    				train_op=train_op, predictions=predictions, eval_metric_ops = eval_metric_ops)

est = tf.estimator.Estimator(model_fn=model_fn, model_dir="/var/tmp/model_dir")					
for _ in range(1):
    est.train(input_fn=tf.estimator.inputs.numpy_input_fn(
        #dict({"features":X}), Y,
        XX.astype(np.float32), YY.astype(np.float32),
        batch_size=1, num_epochs=10, shuffle=True), steps= 214200)
    est.evaluate(input_fn=tf.estimator.inputs.numpy_input_fn(
        XX.astype(np.float32), YY.astype(np.float32),
        batch_size=1, num_epochs=1, shuffle=True), steps= 10)

# Transform test set into numpy matrix
#test = test.drop(labels=['2013-01'],axis=1)
x_test_sales = test.values.reshape((test.shape[0], test.shape[1], 1))
x_test_prices = price #.drop(labels=['2013-01'],axis=1)
x_test_prices = x_test_prices.values.reshape((x_test_prices.shape[0], x_test_prices.shape[1], 1))
print(test.shape)
# Combine Price and Sales Df
test = np.append(x_test_sales,x_test_prices,axis=2)
print(test.shape)
pred=est.predict(input_fn=tf.estimator.inputs.numpy_input_fn(
        TEST.astype(np.float32),
        batch_size=1, num_epochs=1, shuffle=False))
predictions=list()
for i,p in enumerate(pred):
    [pp] = p["predictions"]
    print(p,pp)
    predictions.append(pp)
print(predictions)
submission = pd.DataFrame(predictions,columns=['item_cnt_month'])
submission.to_csv('submission.csv',index_label='ID')
print(submission.head())

