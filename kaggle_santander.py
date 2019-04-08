# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import argparse
import numpy as np
import pandas as pd
#import tensorflow as tf
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.model_selection import train_test_split,GridSearchCV, StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score, auc
import keras
from keras.utils import to_categorical
import keras.backend as K
from keras.models import Sequential, Model, model_from_json
from keras.layers import Add, Subtract, Input, Dense, Activation, Dropout, Flatten, BatchNormalization, Concatenate, Average, Lambda
from keras.layers import Conv2D, MaxPooling2D, Reshape
from sklearn.mixture import GaussianMixture
#tf.enable_eager_execution()
#tf.logging.set_verbosity(tf.logging.INFO)

import pandas as pd
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
print(train_df.columns)
print(test_df.columns)
#ss0=train_df[train_df['target']==0].sample(n=50000)
#ss1=train_df[train_df['target']==1].sample(n=150000, replace=True)
#train_df.append([ss0, ss1], ignore_index=True)
#sample_weights = ((0.9/20000)*train_df.target.values + (0.1/180000)*(1-train_df.target.values))
##train_df = pd.DataFrame(train_df.sample(frac=2,weights=weights, replace=True).values, columns=train_df.columns)
#train_df.set_index(train_df.index, inplace=True)
#train_df = train_df.reindex()
#train_df['target'] = 1-train_df['target']
#print(train_df)
train_df.drop("ID_code", axis=1, inplace=True)
#y_train = pd.get_dummies(train_df, columns=['target'],dtype=np.int32)
#y_train = np.asarray(y_train.values)

x_ones = train_df[train_df['target']==1].iloc[:-98,:]
x_zeros = train_df[train_df['target']==0].append(train_df[train_df['target']==1].iloc[-98:,:])
x_ones.drop("target", axis=1, inplace=True)
x_zeros.drop("target", axis=1, inplace=True)
'''
if(os.path.isfile('data.csv')):
    train_df = pd.read_csv('data.csv')
else:
    gmm = GaussianMixture(n_components=200, covariance_type='tied')
    gmm.fit(np.asarray(x_ones))
    f_x_ones=gmm.sample(100000)[0]
    f_x_ones = pd.DataFrame(f_x_ones, columns=train_df.columns[1:])
    f_x_ones['target'] =1
    train_df = train_df.append(f_x_ones, ignore_index=True)

    train_df.to_csv("data.csv")
'''

y_train = train_df.target.values
train_df.drop("target", axis=1, inplace=True)
train_df.drop("var_45", axis=1, inplace=True)
train_df.drop("var_68", axis=1, inplace=True)
tr_df = train_df.apply(lambda x: [y if y < 55.0 else 55.0 for y in x])
train_df = tr_df.apply(lambda x: [y if y > -45.0 else -45.0 for y in x])
#train_df = train_df/(train_df.std()*train_df.std())
df = train_df.append(train_df.std(), ignore_index=True)
df_sorted = df.T.sort_values(200000, axis=0)
train_df = df_sorted.T
train_df.drop(index=200000, inplace=True)
print(train_df.columns)

test_id = test_df.ID_code
test_df.drop("ID_code", axis=1, inplace=True)
test_df.drop("var_45", axis=1, inplace=True)
test_df.drop("var_68", axis=1, inplace=True)

ts_df = test_df.apply(lambda x: [y if y < 60.0 else 60.0 for y in x])
test_df = ts_df.apply(lambda x: [y if y > -50.0 else -50.0 for y in x])
#test_df = test_df/(test_df.std()*test_df.std())
df = test_df.append(test_df.std(), ignore_index=True)
df_sorted = df.T.sort_values(200000, axis=0)
test_df = df_sorted.T
test_df.drop(index=200000, inplace=True)
print(test_df.columns)

NUM_FEAT = 198
NUM_FEAT1 = 66 #35
NUM_FEAT2 = 66 #69
NUM_FEAT3 = 66
NUM_FEAT4 = 24

#scaler = StandardScaler()
#train_df = pd.DataFrame(scaler.fit_transform(train_df))
#test_df = pd.DataFrame(scaler.fit_transform(test_df))

#print(train_df.mean(), train_df.std())
x_train = train_df.values
x_test = test_df.values

xx_train = x_train
yy_train = y_train

#x_train = Lambda((lambda x:x.reshape(198,1)*x.reshape(1,198)))(x_train)
#x_test = Lambda((lambda x:x.reshape(198,1)*x.reshape(1,198)))(x_test)

x_train = (x_train - x_train.min())/(x_train.max()-x_train.min())
x_test = (x_test - x_test.min())/(x_test.max()-x_test.min())

#print(x_train[1].reshape(198,1)*x_train[3].reshape(1,198))

#x_train = np.reshape(x_train, (-1, NUM_FEAT,1))
#y_train = np.reshape(y_train.values, (2000, 100))
#x_test = np.reshape(x_test, (-1,NUM_FEAT,1))
#x_test = x_test.values
#x_train = x_train.values

ip = Input(shape=(198,))
def net(ipp):
    m1 = Lambda(lambda x:K.reshape(x, (-1,1,198,1))*K.transpose(K.reshape(x, (1,1,198,-1))))(ipp)
    #m1 = Lambda(lambda x:((x-K.min(x))/(K.max(x)-K.min(x))))(m1)
    #m1 = Reshape((198,198,1))
    print(m1)
    conv1 = Conv2D(16, 3)(m1)
    pool1 = MaxPooling2D(2)(conv1)
    #bn1 = BatchNormalization()(pool1)
    conv2 = Conv2D(32, 3)(pool1)
    pool2 = MaxPooling2D(2)(conv2)
    #conv3 = Conv2D(64, 3)(pool2)
    #pool3 = MaxPooling2D()(conv3)
    #bn2 = BatchNormalization()(pool2)
    fltn1 = Flatten()(pool2)
    l1 = Dense(256, activation='elu')(fltn1)
    d1 = Dropout(0.25)(l1)
    l2 = Dense(512, activation='elu')(d1)
    return l2
    
A = net(ip)
B = net(ip)
dA = Dropout(0.5)(A)
dB = Dropout(0.25)(B)

summ = Add()([dA,dB])
subb = Subtract()([dA,dB])
mg = Concatenate()([dA,dB])
merge = Concatenate()([summ, subb, mg])
#d = Dropout(0.25)(merge)
ll1 = Dense(512, activation='elu')(merge)
dd1 = Dropout(0.25)(ll1)
ll2 = Dense(512, activation='elu')(dd1)
dd2 = Dropout(0.25)(ll2)
ll3 = Dense(1024, activation='elu')(dd2)
out = Dense(2, activation='relu')(ll3)

model = Model(ip, out)

import tensorflow as tf
def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc
def auroc(y_true, y_pred):
    return tf.py_function(roc_auc_score, (y_true, y_pred), tf.double)

#from keras.utils import to_categorical
#y_train = to_categorical(y_train, num_classes=None)
class_weights = None #{'cat_out':[0.9,0.1]}
#sample_weights = ((0.9/20000)*y_train + (0.1/180000)*(1-y_train))  
sample_weights = ((0.9)*y_train + (0.1)*(1-y_train))

class_weights = {0:0.5, 1:0.5}
#for i in range(2):
#    batch=int(200/(2*(i+1)))

yy_train = y_train
y_train = to_categorical(y_train, num_classes=None)
x_train = np.reshape(x_train, (-1, NUM_FEAT))
opt = keras.optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(optimizer=opt, loss='mean_squared_error',  metrics=['accuracy'])
folds = StratifiedShuffleSplit(n_splits=5, train_size=0.9, test_size=0.1, random_state=31415)
for train_index, test_index in folds.split(x_train, y_train):
    sample_weights = (0.90)*yy_train[train_index] + (0.10)*(1-yy_train[train_index])
    model.fit(x_train[train_index], y_train[train_index],
            validation_data=(x_train[test_index], y_train[test_index]),
                                   batch_size=100, epochs=5, verbose=2, sample_weight = sample_weights, class_weight=class_weights)
    #model.evaluate([x_train[test_index], x_train[test_index]], y_train[test_index], batch_size=100)
y_pred = model.predict(x_train)
#y_train=label_binarize(list(y_train), [0,1])
roc_val = roc_auc_score(y_train, y_pred)    
#auc = auc(y_train[test_index], y_pred)
print("ROC SCORE:",roc_val)#, "AUC:",auc)
'''
    model.fit(x_train[x_ones.index], y_train[x_ones.index],
            validation_data=(x_train[test_index], y_train[test_index]),
                                   batch_size=100, epochs=5, verbose=2) #sample_weight = sample_weights, class_weight=class_weights)
    #model.evaluate([x_train[test_index], x_train[test_index]], y_train[test_index], batch_size=100)
    y_pred = model.predict(x_train)
    #y_train=label_binarize(list(y_train), [0,1])
    roc_val = roc_auc_score(y_train, y_pred)    
    #auc = auc(y_train[test_index], y_pred)
    print("ROC SCORE:",roc_val)#, "AUC:",auc)
'''    
x_test = np.reshape(x_test, (-1,NUM_FEAT))
y_pred_t = model.predict(x_test, batch_size=100)
submission = pd.read_csv('../input/sample_submission.csv', index_col='ID_code', dtype={"target": np.float32})
submission['target'] = y_pred_t
submission.to_csv('submission.csv')

from sklearn.metrics import confusion_matrix
pred = model.predict(x_train, batch_size=100)
print(confusion_matrix(y_train[:,0],np.round(pred[:,0])))

del model
