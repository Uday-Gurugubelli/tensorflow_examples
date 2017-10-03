#encoding: utf-8
import numpy as np
import data_utils as du
import codecs
import glob as glob
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)
tf.reset_default_graph()

class mail_classifier_model():
    def __init__(self):
        self.batch=100
    def rnn(self, ips):
        with tf.variable_scope("rnn"):
            ips = tf.nn.embedding_lookup(self.word_embedd_matrix, ips)
            ips = tf.reshape(ips, [75, 100, 300])
            cell = tf.nn.rnn_cell.BasicRNNCell(64, activation=tf.nn.sigmoid)
            rnn, _ = tf.nn.dynamic_rnn(cell, ips, dtype=tf.float32)
            output = tf.transpose(rnn, [1, 0, 2])
            last_layer = tf.gather(output, int(output.get_shape()[0]) - 1)
            logits = tf.layers.dense(last_layer, 1, activation=None)
            return logits
    def cnn(self, ips):
        ips = tf.nn.embedding_lookup(self.word_embedd_matrix, ips)
        ips = tf.reshape(ips, [-1, 100, 300, 1])
        network = tf.layers.conv2d(inputs=ips,
                           filters=16,
                           kernel_size=3,
                           padding='SAME')
        network = tf.layers.max_pooling2d(network, pool_size=[2,2], strides=2)
        network_flat = tf.reshape(network, [-1, 50*150*16])
        dense_nw = tf.layers.dense(inputs=network_flat, units=1024)
        dropout = tf.layers.dropout(inputs=dense_nw, rate=0.5)
        logits = tf.layers.dense(dropout, 1)
        return logits
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
        self.word_embedd_matrix, txt_ints, trgts = enron_data.enron_data()
        t1 = txt_ints
        t2 = trgts
        lengths = []
        for i in range(100):
            lengths.append(100)
        #print(len(lengths))
        self.text_lengths =lengths
        self.seq_len = lengths
        t1 = tf.reshape(t1, [75, 100])
        t2 = np.ones(75).astype(np.float32)
        t2 = tf.reshape(t2, [75, 1])
        return t1, t2

    def model_fn(self, features, labels, mode):
        y = self.cnn(features)
        train_op = None
        loss = tf.convert_to_tensor(0.0)
        predictions = None
        eval_metric_ops = None
        global_step = tf.train.get_global_step()
        if(mode == tf.estimator.ModeKeys.EVAL or
                mode == tf.estimator.ModeKeys.TRAIN):
            loss = tf.losses.mean_squared_error(labels, y)
        if(mode == tf.estimator.ModeKeys.TRAIN):
            train_op = tf.train.GradientDescentOptimizer(0.001).minimize(
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
    
    def create_embedd_matrix(self):
        self.embedding_dim = 300
        self.nb_words = len(self.word_dict)
        print("dict len: ", self.nb_words)
        # Create matrix with default values of zero
        self.word_embedding_matrix = np.zeros((self.nb_words, self.embedding_dim), dtype=np.float32)
        for word, i in self.word_dict.items():
            try:
                if word in self.embeddings_index and i < len(self.word_dict):
                    self.word_embedding_matrix[i] = self.embeddings_index[word]
                else:
                    new_embedding = np.array(np.random.uniform(-1.0, 1.0, self.embedding_dim))
                    self.embeddings_index[word]=new_embedding
                    self.word_embedding_matrix[i] = new_embedding
            except IndexError:
                pass
        print("embedd matrix len:" , len(self.word_embedding_matrix))
        
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
        
    #dictionary to convert words to integers
    def text_to_int(self, text):
        text_to_int = []
        self.value = 0
        words = text.split(' ')
        for w in words:
        #if word in self.embeddings_index:
            try:
                text_to_int.append(int(self.word_dict[w]))
            except KeyError:
                pass
        return text_to_int
        
    # Dictionary to convert integers to words    
    def int_to_text(self, text):
        int_to_text = []
        words = text.split(' ')
        #print(words)
        for w in words:
            try:
                #if w != '':
                int_to_text.append(str(self.inv_word_dict[int(w)]))
                #print(self.inv_word_dict[int(w)])
            except KeyError:
                pass
        return int_to_text
    
def enron_data():
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
    f = glob.glob("./data/enron1/spam_test/*.txt")
    text_in_ints = []
    for ff in f:
        h = open(ff)
        txt = h.read()
        h.close()
        txt = pre_proc_text.clean_text(txt)
        ints_list = pre_proc_text.text_to_int(txt)
        i_list = []
        il = ints_list
        if(len(il) > 150):
            cnt = 0
            for i,e in enumerate(il):
                if (i < 120 and int(e) < 15120 and cnt < 100):
                    i_list.append(int(e))
                    cnt = cnt + 1
            text_in_ints.append(i_list)
    print(len(text_in_ints))
    lengths = [len(l) for l in text_in_ints]
    print(lengths)
    embedds = embeddings(pre_proc_text.word_dict)
    embedds.create_embedd_matrix()
    targets = np.ones(75)
    return embedds.word_embedding_matrix, text_in_ints, targets

model = mail_classifier_model()
model.train()


                            
