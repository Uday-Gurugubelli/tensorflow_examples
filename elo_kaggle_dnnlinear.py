import tensorflow as tf
import numpy as np
import pandas as pd
from numpy import linalg as la
import sklearn.preprocessing as preprocess
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC
data = pd.read_csv("train.csv")
#print(data.shape)
#print(data.dtypes)
drop_cols = ["first_active_month", "card_id"]
data.drop(drop_cols, axis=1, inplace=True)
#print(data.columns)
#print(data)
x_train = data.drop(['target'], axis=1).astype(np.float32)
y_train = data.target
#print(y_train)
test_data = pd.read_csv("test.csv")
test_id = test_data.card_id
x_test = test_data.drop(drop_cols, axis=1).astype(np.float32)

#norm = preprocess.StandardScaler().fit(x_train)
#x_train = norm.transform(x_train)
#x_train = x_train + x_train
#norm = preprocess.StandardScaler().fit(x_test)
#x_test = norm.transform(x_test)

#x_train = pd.get_dummies(x_train, columns = ["feature_1", "feature_2", "feature_3"])
#x_test = pd.get_dummies(x_test,  columns = ["feature_1", "feature_2", "feature_3"])
#print(x_train)
#x_train = np.concatenate([x_train, x_train, x_train, x_train, x_train,
#               x_train, x_train, x_train, x_train, x_train], axis=0)
#y_train = np.concatenate([y_train, y_train, y_train, y_train, y_train,
#               y_train, y_train, y_train, y_train, y_train], axis=0)
#print(x_train.shape)
#print(x_test)
#rgrsr = XGBRegressor()
feature_1 = tf.feature_column.categorical_column_with_identity(
                                  'feature_1', 6)
feature_2 = tf.feature_column.categorical_column_with_identity(
                                  'feature_2', 4)
feature_3 = tf.feature_column.categorical_column_with_identity(
                                  'feature_3', 2)

feat_1 = tf.feature_column.indicator_column(
                tf.feature_column.categorical_column_with_vocabulary_list(
                                    'feature_1', ['1', '2', '3', '4', '5']))
feat_2 = tf.feature_column.indicator_column(
                tf.feature_column.categorical_column_with_vocabulary_list(
                                    'feature_2', ['1', '2', '3']))
feat_3 = tf.feature_column.indicator_column(
                tf.feature_column.categorical_column_with_vocabulary_list(
                                    'feature_3', ['0', '1']))

#feature_1 = tf.feature_column.bucketized_column(
#    tf.feature_column.numeric_column(key = 'feature_1'), [0, 6])
#feature_2 = tf.feature_column.bucketized_column(
#    tf.feature_column.numeric_column(key = 'feature_2'), [0, 4])
#feature_3 = tf.feature_column.bucketized_column(
#    tf.feature_column.numeric_column(key = 'feature_3'), [0, 2])


feature_1x2 = tf.feature_column.crossed_column([feature_1, feature_2], 25000)
feature_1x3 = tf.feature_column.crossed_column([feature_1, feature_3], 25000)
feature_2x3 = tf.feature_column.crossed_column([feature_2, feature_3], 25000)
feature_1x2x3 = tf.feature_column.crossed_column([feature_1, feature_2, feature_3], 25000)

feature_1_emb = tf.feature_column.embedding_column(feature_1, 5)
feature_2_emb = tf.feature_column.embedding_column(feature_2, 5)
feature_3_emb = tf.feature_column.embedding_column(feature_3, 5)
feature_1x2_emb = tf.feature_column.embedding_column(feature_1x2, 5)
feature_1x3_emb = tf.feature_column.embedding_column(feature_1x3, 5)
feature_2x3_emb = tf.feature_column.embedding_column(feature_2x3, 5)
feature_1x2x3_emb = tf.feature_column.embedding_column(feature_1x2x3, 5)

'''
feature_1_1 = tf.feature_column.bucketized_column(
    tf.feature_column.numeric_column(key = 'feature_1_1'), (0, 1))
feature_1_2 = tf.feature_column.bucketized_column(
    tf.feature_column.numeric_column(key = 'feature_1_2'), (0, 1))
feature_1_3 = tf.feature_column.bucketized_column(
    tf.feature_column.numeric_column(key = 'feature_1_3'), (0, 1))
feature_1_4 = tf.feature_column.bucketized_column(
    tf.feature_column.numeric_column(key = 'feature_1_4'), (0, 1))
feature_1_5 = tf.feature_column.bucketized_column(
    tf.feature_column.numeric_column(key = 'feature_1_5'), (0, 1))
feature_2_1 = tf.feature_column.bucketized_column(
    tf.feature_column.numeric_column(key = 'feature_2_1'), (0, 1))
feature_2_2 = tf.feature_column.bucketized_column(
    tf.feature_column.numeric_column(key = 'feature_2_2'), (0, 1))
feature_2_3 = tf.feature_column.bucketized_column(
    tf.feature_column.numeric_column(key = 'feature_2_3'), (0, 1))
feature_3_0 = tf.feature_column.bucketized_column(
    tf.feature_column.numeric_column(key = 'feature_3_0'), (0, 1))
feature_3_1 = tf.feature_column.bucketized_column(
    tf.feature_column.numeric_column(key = 'feature_3_1'), (0, 1))

feature_columns = [feature_1_1,feature_1_2,feature_1_3,feature_1_4,feature_1_5,
                   feature_2_1, feature_2_2, feature_2_3, feature_3_0, feature_3_1]
dnn_feature_columns = [feature_1_emb, feature_2_emb, feature_3_emb, feature_1x2_emb,
                        feature_1x3_emb, feature_2x3_emb, feature_1x2x3_emb]

'''
feature_columns = [feature_1, feature_2, feature_3, feature_1x2, feature_1x3, feature_2x3, feature_1x2x3]


classifier = tf.estimator.LinearRegressor(
                feature_columns = feature_columns,
                model_dir = "./model_dnn",
                optimizer =     #tf.train.RMSPropOptimizer(   #0.0001))
                        lambda: tf.train.AdamOptimizer(
                            learning_rate=tf.train.exponential_decay(
                            learning_rate=0.1,
                            global_step=tf.train.get_global_step(),
                            decay_steps=20000,
                            decay_rate=0.96)))

'''
classifier = tf.estimator.DNNRegressor(
                feature_columns = dnn_feature_columns,
                hidden_units = [512, 512, 512, 512],
                model_dir = "./model_dnn1",
                #n_classes = 1,
                #activation_fn = tf.nn.elu,
                dropout=0.7,
                #batch_norm = True,
                optimizer =  #tf.train.GradientDescentOptimizer(),
                    lambda: tf.train.AdamOptimizer(
                        learning_rate=tf.train.exponential_decay(
                        learning_rate=0.1,
                        global_step=tf.train.get_global_step(),
                        decay_steps=20000,
                        decay_rate=0.96))) 
                        
classifier = tf.estimator.BoostedTreesRegressor(
                feature_columns = feature_columns,
                model_dir = "./model_dnn",
                n_batches_per_layer = 10,
                n_trees = 100,
                max_depth=100)
                #tree_complexity = 0.2,
                #l2_regularization=0.01)

'''

classifier.train(input_fn = tf.estimator.inputs.pandas_input_fn(
            x_train.astype(np.int32), y_train,
            batch_size=50, num_epochs=50, shuffle=True), steps= 200000)
        
classifier.evaluate(input_fn = tf.estimator.inputs.pandas_input_fn(
            x_train.astype(np.int32), y_train, shuffle=True), steps= 1000)

pred = classifier.predict(input_fn = tf.estimator.inputs.pandas_input_fn(
        x_test.astype(np.int32), batch_size=10, shuffle=False))

wf = open('./elo_kaggle_submit_file.csv', "w")
wf.write("card_id,target\n")
for i, prd in enumerate(pred):
        [opp] = prd['predictions']
        #print(opp)
        ll = str(test_id.iloc[i]) + "," + str(opp) + "\n"                        
        wf.write(ll)
wf.close()
