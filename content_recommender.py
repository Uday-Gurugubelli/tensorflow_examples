import tensorflow as tf
import argparse
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

tf.reset_default_graph()
FLAGS=None

'''data format:  #id, description of the product'''
def main(_):
    file_q = tf.train.string_input_producer([FLAGS.file_path], shuffle=False)

    reader = tf.TextLineReader(skip_header_lines=1)
    key, value = reader.read(file_q)
    record_defaults = [[1], ["Description"]]
    rev_id, rev = tf.decode_csv(
                    value, record_defaults, field_delim=',')

    min_after_dequeue=500
    capacity = min_after_dequeue + 3 * 100
    feat_id, feat_rev = tf.train.shuffle_batch([rev_id, rev], batch_size=500,
                                    capacity = capacity,
                                    min_after_dequeue = min_after_dequeue,
                                    allow_smaller_final_batch=True)

    tf_idf = TfidfVectorizer(analyzer='word',
                             ngram_range=(1,3),min_df=0,
                             stop_words='english')

    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        feat_rev_list = feat_rev.eval()
        tf_idf_matrix = tf_idf.fit_transform(feat_rev_list)
        cosine_similarities = linear_kernel(tf_idf_matrix, tf_idf_matrix)
        val, indc  = tf.nn.top_k(cosine_similarities,k=10)
        val_l = val.eval()
        indc_l = indc.eval()
        ''' Evaluating the recommendations for all the rows.
            score 1.0 corrosponding to the id for which we need predictions'''
        for i in range(len(val_l)):
            print([(val_l[i][j], indc_l[i][j]) for j in range(10)])
    
        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str,
                        default='C:\\Busigence\\content_rec_data.csv',
                        help='path to access the file')
    FLAGS, unparsed = parser.parse_known_args()
    main(sys.argv[0])

