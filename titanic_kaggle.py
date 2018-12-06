import argparse
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

tf.logging.set_verbosity(tf.logging.INFO)
tf.reset_default_graph()

train_df = pd.read_csv("./train.csv")
test_df = pd.read_csv("./test.csv")
train_df.Pclass = train_df.Pclass.astype('str')

#cat_list = ["Pclass","Name","Sex","Ticket","Cabin","Embarked"]
drop_cols = ["Name","Ticket", "Cabin"]  # due to 270 values only 
train_df.drop(drop_cols, axis=1, inplace=True)
test_df.drop(drop_cols, axis=1, inplace=True)

#print(train_df)
#print(train_df.columns)

train_y = train_df.Survived
test_ID = test_df.PassengerId

train_X = train_df.drop(["PassengerId"], axis=1)
test_X = test_df.drop(["PassengerId"], axis=1)
train_X = train_X.drop(["Survived"], axis=1)

num_cols = ["Age","SibSp","Fare", "Parch"]
num_imputer = SimpleImputer()
train_X[num_cols] = num_imputer.fit_transform(train_X[num_cols])
test_X[num_cols] = num_imputer.fit_transform(test_X[num_cols])

num_cols = ["SibSp", "Parch"]
num_imputer = SimpleImputer(missing_values = 0, 
                       strategy = "constant", fill_value = random.randint(1,4))
train_X[num_cols] = num_imputer.fit_transform(train_X[num_cols])
test_X[num_cols] = num_imputer.fit_transform(test_X[num_cols])

train_X = pd.get_dummies(train_X, columns = ["Sex", "Pclass", "Embarked"])
test_X = pd.get_dummies(test_X, columns = ["Sex", "Pclass", "Embarked"])

#print(train_X.head())
#print(test_X)

for i in range(891):  
  if train_X.iloc[i,0] > 60 : train_X.iloc[i,0] = 60
  if train_X.iloc[i,3] > 100 : train_X.iloc[i,3] = 100
  
for i in range(418):
  if test_X.iloc[i,0] > 60 : test_X.iloc[i,0] = 60 
  if test_X.iloc[i,3] > 100 : test_X.iloc[i,3] = 100  
    

#print(train_X.head())

column_to_normalize = ['Age','Fare','SibSp','Parch']
train_X[column_to_normalize] = train_X[column_to_normalize].apply(
                lambda x: x / x.max())
test_X[column_to_normalize] = test_X[column_to_normalize].apply(
                lambda x: x / x.max())   
    
#print(test_X)


age = tf.feature_column.bucketized_column(
    tf.feature_column.numeric_column(key = 'Age'), (0, 1))
sibsp = tf.feature_column.bucketized_column(
    tf.feature_column.numeric_column(key = 'SibSp'), (0,1))
parch = tf.feature_column.bucketized_column(
    tf.feature_column.numeric_column(key = 'Parch'), (0, 1,))
fare = tf.feature_column.bucketized_column(
    tf.feature_column.numeric_column(key = 'Fare'), (0, 1))

sex_male = tf.feature_column.bucketized_column(
    tf.feature_column.numeric_column(key = 'Sex_male'), (0,1))
sex_female = tf.feature_column.bucketized_column(
    tf.feature_column.numeric_column(key = 'Sex_female'), (0,1))
embarked_s = tf.feature_column.bucketized_column(
    tf.feature_column.numeric_column(key = 'Embarked_S'), (0,1))
embarked_q = tf.feature_column.bucketized_column(
    tf.feature_column.numeric_column(key = 'Embarked_Q'), (0,1))
embarked_c = tf.feature_column.bucketized_column(
    tf.feature_column.numeric_column(key = 'Embarked_C'), (0,1))
pclass_1 = tf.feature_column.bucketized_column(
    tf.feature_column.numeric_column(key = 'Pclass_1'), (0, 1))
pclass_2 = tf.feature_column.bucketized_column(
    tf.feature_column.numeric_column(key = 'Pclass_2'), (0, 1))
pclass_3 = tf.feature_column.bucketized_column(
    tf.feature_column.numeric_column(key = 'Pclass_3'), (0, 1))

feature_columns = [age, sibsp, fare, sex_male, sex_female, parch,
                   embarked_c, embarked_q, embarked_s, pclass_1, pclass_2, pclass_3]

classifier = tf.estimator.LinearClassifier(
                          feature_columns, "./model_lin",
                          optimizer = tf.train.AdamOptimizer(0.01))
                                          #tf.train.exponential_decay(
                                          #learning_rate=0.1,
                                          #global_step=tf.train.get_global_step(),
                                          #decay_steps=1000,
                                          #decay_rate=0.9)))
                          
''' got 0.7799 the high 6dec'''
#classifier = tf.estimator.BoostedTreesClassifier(
#                          feature_columns, 50, "./model_tree",
#                          learning_rate = 0.01 ,#tf.train.exponential_decay(
                                          #learning_rate=0.1,
                                          #global_step=tf.train.get_global_step(),
                                          #decay_steps=10000,
                                          #decay_rate=0.96))
#                          l1_regularization=0.05,
#                          l2_regularization=0.05)
                          
'''
classifier = tf.estimator.DNNLinearCombinedClassifier(
                linear_feature_columns = feature_columns,
                linear_optimizer = tf.train.AdamOptimizer(tf.train.exponential_decay(
                                          learning_rate=0.1,
                                          global_step=g_step,
                                          decay_steps=1000,
                                          decay_rate=0.9)),
                dnn_feature_columns = feature_columns,
                dnn_optimizer = tf.train.AdamOptimizer(tf.train.exponential_decay(
                                          learning_rate=0.1,
                                          global_step=g_step,
                                          decay_steps=1000,
                                          decay_rate=0.9)),
                dnn_hidden_units = [24, 16, 8],
                model_dir = "./model_dnn",
                n_classes = 2,
                dnn_activation_fn = tf.nn.elu,
                #dnn_dropout=0.2,
                batch_norm = True)
'''

def main(_):
    #file = open('./test.csv', "r")
    #fl = file.readline()
    wf = open('./titanic_kaggle_submit_file.txt', "w")
    wf.write("PassengerId,Survived\n")
    for i in range(5):
      classifier.train(input_fn = tf.estimator.inputs.pandas_input_fn(
            train_X, train_y,
            batch_size=10, num_epochs=25, shuffle=True), steps= 500000)
        
      classifier.evaluate(input_fn = tf.estimator.inputs.pandas_input_fn(
            train_X, train_y, batch_size=10, shuffle=True), steps= 90)
    
    pred = classifier.predict(input_fn = tf.estimator.inputs.pandas_input_fn(
        test_X, batch_size=10, shuffle=False))
    for i, prd in enumerate(pred):
        [opp] = prd['classes'].astype(int)
        #print(opp)
        ll = str(i+892) + "," + str(opp) + "\n"                        
        wf.write(ll)
    #file.close()
    wf.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', type=bool,
                        default=False,
                        help='path to model dir')
    parser.add_argument('--test', type=bool,
                        default=False,
                        help='path to model dir')
    parser.add_argument('--phase', type=bool,
                        default=True,
                        help='path to model dir')
    FLAGS, unparsed = parser.parse_known_args()

tf.app.run()
