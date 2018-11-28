import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import accuracy_score

tf.logging.set_verbosity(tf.logging.INFO)
tf.reset_default_graph()

train_df = pd.read_csv("./train.csv")
test_df = pd.read_csv("./test.csv")
train_df.Pclass = train_df.Pclass.astype('str')

cat_list = ["Pclass","Name","Sex","Ticket","Cabin","Embarked"]
drop_cols = ["Name","Ticket"]
train_df.drop(drop_cols, axis=1, inplace=True)
test_df.drop(drop_cols, axis=1, inplace=True)

#print(train_df)
#print(train_df.columns)

train_y = train_df.Survived
test_ID = test_df.PassengerId

train_X = train_df.drop(["PassengerId"], axis=1)
test_X = test_df.drop(["PassengerId"], axis=1)
train_X = train_X.drop(["Survived"], axis=1)

num_cols = ["Age","SibSp","Parch","Fare"]
cat_cols = ["Sex","Cabin","Embarked","Pclass"]

num_imputer = SimpleImputer()
train_X[num_cols] = num_imputer.fit_transform(train_X[num_cols])
test_X[num_cols] = num_imputer.fit_transform(test_X[num_cols])

for col in cat_cols:
    cat = LabelEncoder()
    cat.fit(list(train_X[col].values.astype('str')) + list(test_X[col].values.astype('str')))
    train_X[col] = cat.transform(list(train_X[col].values.astype('str')))
    test_X[col] = cat.transform(list(test_X[col].values.astype('str')))

print(train_X)
print(test_X)

train_X = train_X/train_X.max().astype(np.float32)
test_X = test_X/test_X.max().astype(np.float32)

print(train_X)
print(test_X)


age = tf.feature_column.bucketized_column(
    tf.feature_column.numeric_column(key = 'Age'), (0, 1))
sibsp = tf.feature_column.bucketized_column(
    tf.feature_column.numeric_column(key = 'SibSp'), (0,1))
parch = tf.feature_column.bucketized_column(
    tf.feature_column.numeric_column(key = 'Parch'), (0, 1,))
fare = tf.feature_column.bucketized_column(
    tf.feature_column.numeric_column(key = 'Fare'), (0, 1))

sex = tf.feature_column.bucketized_column(
    tf.feature_column.numeric_column(key = 'Sex'), (0,1))
cabin = tf.feature_column.bucketized_column(
    tf.feature_column.numeric_column(key = 'Cabin'), (0,1))
embarked = tf.feature_column.bucketized_column(
    tf.feature_column.numeric_column(key = 'Embarked'), (0,1))
pclass = tf.feature_column.bucketized_column(
    tf.feature_column.numeric_column(key = 'Pclass'), (0, 1))

feature_columns = [age, sibsp, parch, fare, sex, cabin, embarked, pclass]

classifier = tf.estimator.BoostedTreesClassifier(
                feature_columns = feature_columns,
                model_dir = "./model_dnn",
                n_batches_per_layer = 50,
                n_trees = 100,
                l2_regularization=0.1)

def main(_):
    #file = open('./test.csv', "r")
    #fl = file.readline()
    wf = open('./titanic_kaggle_submit_file.txt', "w")
    wf.write("PassengerId,Survived\n")

    classifier.train(input_fn = tf.estimator.inputs.pandas_input_fn(
        train_X, train_y,
        batch_size=16, num_epochs=100, shuffle=True), steps= 10000)
        
    classifier.evaluate(input_fn = tf.estimator.inputs.pandas_input_fn(
        train_X, train_y, shuffle=True), steps= 1)
    
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
