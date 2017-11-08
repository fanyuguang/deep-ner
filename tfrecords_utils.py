#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
import os
import tensorflow as tf
import data_utils

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('tfrecords_path', 'data/tfrecords/', 'tfrecords directory')
tf.app.flags.DEFINE_integer('batch_size', 1024, 'words batch size')
tf.app.flags.DEFINE_integer('min_after_dequeue', 10000, 'min after dequeue')
tf.app.flags.DEFINE_integer('num_threads', 1, 'read batch num threads')
tf.app.flags.DEFINE_integer('num_steps', 200, 'num steps, equals the length of words')


def create_record(words_list, labels_list, tfrecords_filename):
  print 'Create record to ' + tfrecords_filename
  writer = tf.python_io.TFRecordWriter(tfrecords_filename)
  assert len(words_list) == len(labels_list)
  for (word_ids, label_ids) in zip(words_list, labels_list):
    word_list = [int(word) for word in word_ids.strip().split()]
    label_list = [int(label) for label in label_ids.strip().split()]
    assert len(word_list) == len(label_list)
    example = tf.train.Example(features=tf.train.Features(feature={
      'words': tf.train.Feature(int64_list=tf.train.Int64List(value=word_list)),
      'labels': tf.train.Feature(int64_list=tf.train.Int64List(value=label_list)),
    }))
    writer.write(example.SerializeToString())
  writer.close()


def read_and_decode(tfrecords_filename):
  print 'Read record from ' + tfrecords_filename
  num_steps = FLAGS.num_steps
  filename_queue = tf.train.string_input_producer([tfrecords_filename], num_epochs=None)
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  feature_configs = {
    # 'words': tf.FixedLenFeature(shape=[num_steps], dtype=tf.int64, default_value=0),
    'words': tf.VarLenFeature(dtype=tf.int64),
    'labels': tf.VarLenFeature(dtype=tf.int64),
  }
  features = tf.parse_single_example(serialized_example, features=feature_configs)
  words = features['words']
  words_len = words.dense_shape[0]
  words_len = tf.minimum(words_len, tf.constant(num_steps, tf.int64))
  words = tf.sparse_to_dense(sparse_indices=words.indices[:num_steps], output_shape=[num_steps],
                             sparse_values=words.values[:num_steps], default_value=0)
  labels = features['labels']
  labels = tf.sparse_to_dense(sparse_indices=labels.indices[:num_steps], output_shape=[num_steps],
                              sparse_values=labels.values[:num_steps], default_value=0)
  batch_size = FLAGS.batch_size
  min_after_dequeue = FLAGS.min_after_dequeue
  capacity = min_after_dequeue + 3 * batch_size
  num_threads = FLAGS.num_threads
  words_batch, labels_batch, words_len_batch = tf.train.shuffle_batch([words, labels, words_len], batch_size=batch_size, capacity=capacity,
                                                     min_after_dequeue=min_after_dequeue, num_threads=num_threads)
  return words_batch, labels_batch, words_len_batch


def print_all(tfrecords_filename):
  number = 1
  for serialized_example in tf.python_io.tf_record_iterator(tfrecords_filename):
    example = tf.train.Example()
    example.ParseFromString(serialized_example)
    words = example.features.feature['words'].int64_list.value
    labels = example.features.feature['labels'].int64_list.value
    word_list = [word for word in words]
    label_list = [label for label in labels]
    print('Number:{}, labels: {}, features: {}'.format(number, label_list, word_list))
    number += 1


def print_shuffle(tfrecords_filename):
  words_batch, labels_batch, words_len_batch = read_and_decode(tfrecords_filename)
  with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    try:
      while not coord.should_stop():
        batch_words_r, batch_labels_r, batch_words_len_r = sess.run([words_batch, labels_batch, words_len_batch])
        print 'batch_words_r : ', batch_words_r.shape
        print batch_words_r
        print 'batch_labels_r : ', batch_labels_r.shape
        print batch_labels_r
        print 'batch_words_len_r : ', batch_words_len_r.shape
        print batch_words_len_r
    except tf.errors.OutOfRangeError:
      print 'Done reading'
    finally:
      coord.request_stop()
    coord.join(threads)


def main(_):
  datasets_path = FLAGS.datasets_path
  vocab_path = FLAGS.vocab_path
  tfrecords_path = FLAGS.tfrecords_path

  words_vocab, labels_vocab = data_utils.initialize_vocabulary(vocab_path)

  train_word_ids_list, train_label_ids_list = data_utils.data_to_token_ids(
    os.path.join(datasets_path, 'train.txt'), words_vocab, labels_vocab)
  validation_word_ids_list, validation_label_ids_list = data_utils.data_to_token_ids(
    os.path.join(datasets_path, 'validation.txt'), words_vocab, labels_vocab)
  test_word_ids_list, test_label_ids_list = data_utils.data_to_token_ids(
    os.path.join(datasets_path, 'test.txt'), words_vocab, labels_vocab)

  create_record(train_word_ids_list, train_label_ids_list, os.path.join(tfrecords_path, 'train.tfrecords'))
  create_record(validation_word_ids_list, validation_label_ids_list, os.path.join(tfrecords_path, 'validate.tfrecords'))
  create_record(test_word_ids_list, test_label_ids_list, os.path.join(tfrecords_path, 'test.tfrecords'))

  print_all(os.path.join(tfrecords_path, 'train.tfrecords'))
  # print_shuffle(os.path.join(tfrecords_path, 'test.tfrecords'))


if __name__ == '__main__':
  tf.app.run()
