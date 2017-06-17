import tensorflow as tf

tf.reset_default_graph()
tf.logging.set_verbosity(tf.logging.INFO)

init_op=tf.global_variables_initializer()

file = "C:\\tf_examples\\text_lines_input_1k.txt"

def find_bigrams(input_list):
    bigram_list = []
    for i in range(len(input_list)-1):
        bigram_list.append((input_list[i], input_list[i+1]))
    return bigram_list

file_q= tf.train.string_input_producer([file], shuffle=False)
reader = tf.TextLineReader(skip_header_lines=0)
line_key, line_value  = reader.read(file_q)
min_after_dequeue=500
capacity = min_after_dequeue + 3 * 100
word_q = tf.RandomShuffleQueue(capacity=capacity,
                               min_after_dequeue=min_after_dequeue,
                               dtypes=tf.int32, shapes=[1,2])

line_k, line = tf.train.shuffle_batch([line_key, line_value], batch_size=1000,
                                capacity = capacity,
                                min_after_dequeue = min_after_dequeue,
                                allow_smaller_final_batch=False)

line_join=tf.reduce_join(line)
tokenizer = tf.contrib.keras.preprocessing.text.Tokenizer(
                                    num_words=7500)

'''**********'''
vocab_size=7500
embeddings = tf.Variable(tf.random_uniform([vocab_size, 128], -1, 1))
weights = tf.Variable(tf.truncated_normal([vocab_size, 128],
                        stddev=1.0/tf.sqrt(tf.cast(vocab_size, tf.float32))))
biases = tf.Variable(tf.zeros([vocab_size]))

train_inputs  = tf.placeholder(tf.int32, [1000])
train_labels = tf.placeholder(tf.int32, [1000, 1])

embed = tf.nn.embedding_lookup(embeddings, train_inputs)

loss = tf.reduce_mean(
                tf.nn.nce_loss(weights=weights,
                               biases = biases,
                               inputs = embed,
                               labels = train_labels,
                               num_sampled=100,
                               num_classes=7500))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0).minimize(loss)
'''**************'''
with tf.Session() as sess:
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    word_list=tf.contrib.keras.preprocessing.text.text_to_word_sequence(
                                line_join.eval().decode())
    tokenizer.fit_on_texts(word_list)
    word_ints = tokenizer.texts_to_sequences(word_list)

    bigrams = find_bigrams(word_ints)
    rev_bigrams = find_bigrams(word_ints[::-1])
    
    tens_bi = tf.convert_to_tensor(bigrams)
    tens_rev_bi = tf.convert_to_tensor(rev_bigrams)
    tens_bi = tf.reshape(tens_bi, [-1,1,2])
    tens_rev_bi = tf.reshape(tens_rev_bi, [-1,1,2])
    word_enq1 = word_q.enqueue_many(tens_rev_bi)
    word_enq2 = word_q.enqueue_many(tens_bi)
    words_batch=[]
    for i in range(1000):    
        words_batch.append(word_q.dequeue())
    qr = tf.train.QueueRunner(word_q, [word_enq1, word_enq2])
    q_threads = qr.create_threads(sess, coord=coord, start=True)
    print("***************")
    
    t1, t2 = tf.unstack(words_batch, axis=-1)
    x=[]
    y=[]
    for i in range(1000):
        x = t1[i].eval()
        y = t2[i].eval()
    lss = sess.run(loss, feed_dict={train_inputs: x,
                              train_labels: y})
    print(lss)
    
    coord.request_stop()
    coord.join(q_threads)
    coord.join(threads)
