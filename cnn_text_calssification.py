from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble

import pandas, xgboost, numpy, textblob, string
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers 

import pandas as pd

# read data
train_df = pd.read_csv("./Dataset/Train.csv")
test_df = pd.read_csv("./Dataset/Test.csv")
print(train_df.columns)
print(train_df['user'].head())

drop_list = ['ID', 'date', 'user', 'topic']
train_df.drop(drop_list, axis=1, inplace=True)
test_df.drop(drop_list, axis=1, inplace=True)

train_df["full_comment"] = train_df["comment"] +train_df["parent_comment"]
test_df["full_comment"] = test_df["comment"] +test_df["parent_comment"]

import nltk
nltk.download()

from nltk.corpus import wordnet

def get_wordnet_pos(pos_tag):
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
    
import string
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer

def clean_text(text):
    # lower text
    text = text.lower()
    # tokenize text and remove puncutation
    text = [word.strip(string.punctuation) for word in text.split(" ")]
    # remove words that contain numbers
    text = [word for word in text if not any(c.isdigit() for c in word)]
    # remove stop words
    stop = stopwords.words('english')
    text = [x for x in text if x not in stop]
    # remove empty tokens
    text = [t for t in text if len(t) > 0]
    # pos tag text
    pos_tags = pos_tag(text)
    # lemmatize text
    text = [WordNetLemmatizer().lemmatize(t[0], get_wordnet_pos(t[1])) for t in pos_tags]
    # remove words with only one letter
    text = [t for t in text if len(t) > 1]
    # join all
    text = " ".join(text)
    return(text)

# clean text data
train_df["full_comment_clean"] = train_df["full_comment"].apply(lambda x: clean_text(x))
test_df["full_comment_clean"] = test_df["full_comment"].apply(lambda x: clean_text(x))
print(train_df.head())

df = train_df["full_comment_clean"] + test_df["full_comment_clean"]
df = df.fillna(" ")

tokenizer = text.Tokenizer()
tokenizer.fit_on_texts(df)
#df = tokenizer.texts_to_matrix(df, mode='count')
train_x = tokenizer.texts_to_sequences(train_df["full_comment_clean"])
test_x = tokenizer.texts_to_sequences(test_df["full_comment_clean"])
#print(tokenizer.word_index)
#print(df.shape)
#print(df)
#print([len(x) for x in df])
train_x = sequence.pad_sequences(train_x, maxlen=100)
test_x = sequence.pad_sequences(test_x, maxlen=100)
train_y = train_df.label.values

# split the data into train and test
from sklearn.model_selection import train_test_split

x_train, x_valid, y_train, y_valid = train_test_split(train_x, train_y, test_size = 0.20, random_state = 42)

from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.datasets import imdb

# set parameters:
max_features = 50000
maxlen = 100
batch_size = 32
embedding_dims = 50
filters = 50
kernel_size = 3
hidden_dims = 100
epochs = 2

print('Build model...')
model = Sequential()

# we start off with an efficient embedding layer which maps
# our vocab indices into embedding_dims dimensions
model.add(Embedding(max_features,
                    embedding_dims,
                    input_length=maxlen))
model.add(Dropout(0.2))

# we add a Convolution1D, which will learn filters
# word group filters of size filter_length:
model.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1))
# we use max pooling:
model.add(GlobalMaxPooling1D())

# We add a vanilla hidden layer:
model.add(Dense(hidden_dims))
model.add(Dropout(0.2))
model.add(Activation('relu'))

# We project onto a single unit output layer, and squash it with a sigmoid:
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
              
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_valid, y_valid))
          
import numpy as np

y_test_pred = np.round(model.predict(test_x)).flatten()

test_df = pd.read_csv("./Dataset/Test.csv")
print(test_df.ID.values)
result_df = pd.DataFrame({'ID':test_df.ID.values, 'label':y_test_pred})
result_df.to_csv("icertis_submission.csv", index=False)

from sklearn.metrics import f1_score, accuracy_score
import numpy as np
y_pred = np.round(model.predict(x_valid))
print(f1_score(y_valid, y_pred, average='weighted'))
print(accuracy_score(y_valid, y_pred))

