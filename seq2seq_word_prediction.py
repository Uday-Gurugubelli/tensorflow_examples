import tensorflow as tf
import numpy as np
tf.reset_default_graph()
tf.logging.set_verbosity(tf.logging.INFO)

file = "/home/udaygurugubelli/nlp/text_lines_input_1k.txt"

file_q= tf.train.string_input_producer([file], shuffle=False)
reader = tf.TextLineReader(skip_header_lines=0)
line_key, line_value  = reader.read(file_q)
min_after_dequeue=500
capacity = min_after_dequeue + 3 * 100

line_k, line = tf.train.shuffle_batch([line_key, line_value], batch_size=2,
                                capacity = capacity,
                                min_after_dequeue = min_after_dequeue,
                                allow_smaller_final_batch=False)

line_join=tf.reduce_join(line)
#tokenizer = tf.contrib.learn.preprocessing.tokenizer(line)
init = tf.global_variables_initializer()            
sess=tf.InteractiveSession()
sess.run(init)
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

#word_list=tf.contrib.keras.preprocessing.text.text_to_word_sequence(
#		                                		line_join.eval().decode())
#tokenizer.fit_on_texts(word_list)
#print(word_lsit)

word_list = tf.string_split(line)
wordlist = word_list.values.eval()
word_set = set(wordlist)
dict = {w:i for i, w in enumerate(word_set)}

VOCAB_LEN = len(dict)
MAX_LEN = 3

next_word = []
seq = []
for i in range(VOCAB_LEN - MAX_LEN - 1):
    seq.append(wordlist[i:i+MAX_LEN])
    next_word.append(wordlist[i+MAX_LEN])    
#print (seq)
SEQ_LEN = len(seq)

X = np.zeros([SEQ_LEN, MAX_LEN, VOCAB_LEN], dtype=np.bool)
Y = np.zeros([SEQ_LEN, VOCAB_LEN], dtype=np.bool)

for i, s in enumerate(seq):
    for j, t in enumerate(s):
        X[i, j, dict[t]] = True
    Y[i, dict[next_word[i]]] = True

X=tf.convert_to_tensor(tf.cast(X, tf.float32))
Y=tf.convert_to_tensor(tf.cast(Y, tf.float32))
batch = 10
#X = tf.reshape(X, [SEQ_LEN, MAX_LEN, VOCAB_LEN])
X = tf.reshape(X, [MAX_LEN, SEQ_LEN, VOCAB_LEN])
X = tf.reshape(X, [SEQ_LEN, VOCAB_LEN, MAX_LEN])

#print(X.shape)
input  = tf.placeholder(tf.float32, [SEQ_LEN, MAX_LEN, VOCAB_LEN])
labels = tf.placeholder(tf.float32, [None, VOCAB_LEN])

#input1 = tf.slice(inputs, [0,0,VOCAB_LEN], [SEQ_LEN,1,VOCAB_LEN])
#input1  =tf.reshape(input1, [SEQ_LEN, VOCAB_LEN])

#input2 = tf.slice(inputs, [0,1,VOCAB_LEN], [SEQ_LEN,1,VOCAB_LEN])
#input2  =tf.reshape(input2, [SEQ_LEN, VOCAB_LEN])

#input3 = tf.slice(inputs, [0,2,VOCAB_LEN], [SEQ_LEN,1,VOCAB_LEN])
#input3  =tf.reshape(input3, [SEQ_LEN, VOCAB_LEN])

cell = tf.contrib.rnn.BasicLSTMCell(512)

state = tf.zeros([SEQ_LEN, 512])

print(X.eval(), X.shape)
print(X.shape)
y_ = tf.contrib.rnn.static_rnn(cell=cell, inputs=X, sequence_length=3 )

loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y_, labels))

train_op = tf.train.AdamOptimizer(0.01).minimize(loss)

lss, _ = sess.run([loss, train_op], feed_dict={input: X, labels: Y})
print("Loss: ", lss)

coord.request_stop()
coord.join(threads)	


'''**********'''
'''
batch= 100


vocab_size=7500
nce_biases = tf.Variable(tf.zeros([vocab_size]))
nce_weights = tf.Variable(tf.random_normal([vocab_size, 128],
                        mean=0.,
                        stddev=1.0/tf.sqrt(tf.cast(vocab_size, tf.float32))))

loss = tf.reduce_mean(
                tf.nn.nce_loss(weights=nce_weights,
                               biases = nce_biases,
                               inputs = embed,
                               labels = train_labels,
                               num_sampled=100,
                               num_classes=vocab_size))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0).minimize(loss)
'''
'''**************'''
'''
max_len = 5
init_op=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
    #word_list=tf.contrib.keras.preprocessing.text.text_to_word_sequence(
    #                            line_join.eval().decode())
    #tokenizer.fit_on_texts(word_list)
    print(line)
    word_list = tf.string_split(line)
    #word_ints = tokenizer.texts_to_sequences(word_list)
    w=word_list.values.eval()
    voc_vec = tf.zeros([len(w)])
    next_wrd = []
    seq = []
    for i in range(len(w)):
    	seq.append(w[i:i+max_len])
    	next_wrd(w[i+max_len])
    
    print (seq)
    
	#print("***************")
    
#    for i in range(500):
#        embd, lss = sess.run([embed, loss], feed_dict={train_inputs: t1.eval(),
#                                              train_labels: t2.eval()})
#        print(lss)
    
    coord.request_stop()
    coord.join(threads)
'''