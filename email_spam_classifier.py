#encoding: utf-8
import numpy as np
import data_utils as du
import codecs
import glob as glob
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
tf.logging.set_verbosity(tf.logging.INFO)
tf.reset_default_graph()
mnist = input_data.read_data_sets("MNIST_data/")

class mail_classifier_model():
    def __init__(self):
        self.batch=100
    def nn(self,x):
        x = tf.layers.dense(inputs=x, units=1024, activation=tf.nn.relu)
        x = tf.nn.dropout(x, 0.5)
        x = tf.layers.dense(inputs=x, units=512, activation=tf.nn.relu)
        x = tf.nn.dropout(x, 0.5)
        x = tf.layers.dense(inputs=x, units=256, activation=tf.nn.relu)
        x = tf.nn.dropout(x, 0.5)
        x = tf.layers.dense(inputs=x, units=1, activation=tf.nn.sigmoid)
        return x
   def estimator(self):
        return tf.estimator.Estimator(model_fn = self.model_fn,
                             model_dir="/home/udaygurugubelli/mail_classifier/model_dir/")
    def train(self):
        est = self.estimator()
        #for i in range(500):
        est.train(input_fn = self.input_fn, steps=10)
            #if(i % 10 == 0):
        est.evaluate(input_fn=self.input_fn, steps=1)
    def predict(self):
        est = self.estimator()
        pred = est.predict(input_fn=self.input_fn)
    def input_fn(self):
        t1, t2 = mnist.train.next_batch(100)
        #t1 = np.random.uniform(0., 1., [100,784])
        t2 = tf.reshape(t2, [100, 1])
        return tf.convert_to_tensor(t1), t2

    def model_fn(self, features, labels, mode):
        y = self.nn(features)
        train_op = None
        loss = tf.convert_to_tensor(0.0)
        predictions = None
        eval_metric_ops = None
        global_step = tf.train.get_global_step()
        if(mode == tf.estimator.ModeKeys.EVAL or
                mode == tf.estimator.ModeKeys.TRAIN):
            loss = tf.losses.mean_squared_error(labels, y)
        if(mode == tf.estimator.ModeKeys.TRAIN):
            train_op = tf.train.GradientDescentOptimizer(0.01).minimize(
                loss, global_step = global_step)
        if(mode == tf.estimator.ModeKeys.PREDICT):
            predictions = {"pred_labels ": y}
        if (mode == tf.estimator.ModeKeys.EVAL):
            eval_metric_ops = {"accuracy":
                    tf.metrics.accuracy(labels, y)}
        ret = tf.estimator.EstimatorSpec(mode=mode, loss=loss,
                        train_op=train_op, predictions=predictions,
                            eval_metric_ops = eval_metric_ops)
        return ret
        
# Load Conceptnet Numberbatch's (CN) embeddings, similar to GloVe, but probably better 
# (https://github.com/commonsense/conceptnet-numberbatch)
class embeddings(object):
    def __init__(self, word_dict):
        self.word_dict = word_dict
        self.embeddings_index = self.load_embedds()
    def load_embedds(self):
        embeddings_index = {}
        with codecs.open('./numberbatch-en-17.06.txt', 'r', encoding='utf-8') as f_handler:
            for line in f_handler:
                values = line.split(' ')
                word = values[0]
                embedding = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = embedding
        print('length of Word embeddings:', len(embeddings_index))
        return embeddings_index
    
    def create_embed_matrix(self):
        self.embedding_dim = 300
        self.nb_words = len(vocab_to_int)
        # Create matrix with default values of zero
        self.word_embedding_matrix = np.zeros((self.nb_words, self.embedding_dim), dtype=np.float32)
        for word, i in vocab_to_int.items():
            if word in embeddings_index:
                word_embedding_matrix[i] = self.embeddings_index[word]
            else:
                # If word not in CN, create a random embedding for it
                new_embedding = np.array(np.random.uniform(-1.0, 1.0, self.embedding_dim))
                self.embeddings_index[word] = new_embedding
                self.word_embedding_matrix[i] = new_embedding
        # Check if value matches len(vocab_to_int)
        print(len(word_embedding_matrix))
        
class pre_process_text():
    def __init__(self):
        pass
    def clean_text(self, text, remove_stopwords = True):
        '''Remove unwanted characters, stopwords, and format the text to create fewer nulls word embeddings'''
        # Convert words to lower case
        text = text.lower()
        # Replace contractions with their longer forms
        if True:
            text = text.split()
            new_text = []
            for word in text:
                if word in du.contractions:
                    new_text.append(du.contractions[word])
                else:
                    new_text.append(word)
        text = " ".join(new_text)
        # Format words and remove unwanted characters
        text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
        text = re.sub(r'\<a href', ' ', text)
        text = re.sub(r'&amp;', '', text)
        text = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', text)
        text = re.sub(r'<br />', ' ', text)
        text = re.sub(r'\'', ' ', text)

        # Optionally, remove stop words
        if remove_stopwords:
            text = text.split()
            stops = set(stopwords.words("english"))
            text = [w for w in text if not w in stops]
            text = " ".join(text)
        return text
    def make_vocabulary(self, text):
        tokens = word_tokenize(text)
        words = [w.lower for w in tokens]
        self.word_tokens = set(tokens)
        self.word_dict = {w:i for i, w in enumerate(self.word_tokens)}
        list_to_remove = self.word_dict.keys()
        for item in list_to_remove:
            if item.isalpha() == False:
                del self.word_dict[item]
            elif (len(item) == 1):
                del self.word_dict[item]
        self.inv_word_dict = dict([[v,k] for k,v in self.word_dict.iteritems()])
        #print(len(self.word_dict))
        #for i in range(100,110):
        #   print(self.inv_word_dict[i])

    #dictionary to convert words to integers
    def text_to_int(self, text):
        text_to_int = ''
        self.value = 0
        words = text.split(' ')
        for w in words:
        #if word in self.embeddings_index:
            try:
                text_to_int += ' ' + str(self.word_dict[w])
            except KeyError:
                pass
        return text_to_int
        
    # Dictionary to convert integers to words    
    def int_to_text(self, text):
        int_to_text = ' '
        words = text.split(' ')
        #print(words)
        for w in words:
            try:
                if w != '':
                    int_to_text += ' ' + self.inv_word_dict[int(w)]
                    #print(self.inv_word_dict[int(w)])
            except KeyError:
                pass
        return int_to_text
f = glob.glob("./data/enron1/spam_test/*.txt")
clean_text = ''
for ff in f:
    #print(ff)  
    h = open(ff)
    txt = h.read()
    pre_proc_text = pre_process_text()
    clean_text += pre_proc_text.clean_text(txt)
    #print(len(clean_text))
    h.close()
pre_proc_text.make_vocabulary(clean_text)
for ff in f:
    h = open(ff)
    txt = h.read()
    txt = pre_proc_text.clean_text(txt)
    ints = pre_proc_text.text_to_int(txt)
    #print(ints)
    txt=pre_proc_text.int_to_text(ints)
    print(txt)
embds = embeddings()                                
model = mail_classifier_model()
model.train()
                            
