from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import glob
import threading
from skimage.io import imread

import tensorflow as tf

FLAGS = None

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
   
def img_to_record(coord):
  while not coord.should_stop():
    imgs = glob.glob(FLAGS.in_dir + "/*")
    for im in imgs:
      img = imread(im)
      rows = img.shape[0]
      cols = img.shape[1]
      depth = img.shape[2]

      filename = os.path.join(FLAGS.out_dir, im.lstrip(FLAGS.in_dir) + '.tfrecords')
      writer = tf.python_io.TFRecordWriter(filename)
      image_raw = img.tostring()
      example = tf.train.Example(features=tf.train.Features(feature={
          'height': _int64_feature(rows),
          'width': _int64_feature(cols),
          'depth': _int64_feature(depth), 
          #'label': _int64_feature(int(labels[index])),
          'image_raw': _bytes_feature(image_raw)}))
      writer.write(example.SerializeToString())
      writer.close()


def main(unused_argv):
  init_gv = tf.global_variables_initializer()
  init_lv=tf.local_variables_initializer()
  with tf.Session() as sess:
    sess.run(init_gv)
    sess.run(init_lv)
    coord=tf.train.Coordinator()
    threads = threading.Thread(target=img_to_record(coord), args=(coord,))

    threads.start()
    
    #coord.request_stop()
    #coord.join(threads)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--in_dir',
      type=str,
      default='/tmp/data',
      help='path to get the image files to convert to TFrecords'
  )
  parser.add_argument(
      '--out_dir',
      type=str,
      default='/tmp/data',
      help="""\
      path to store the TFreocrds.\
      """
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
