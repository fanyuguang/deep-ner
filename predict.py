#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
from itertools import izip
import numpy as np
import tensorflow as tf
from tensorflow.contrib import lookup
import tensorflow.contrib.crf as crf
import crf_utils
import data_utils
import tfrecords_utils
import model
import train

reload(sys)
sys.setdefaultencoding('utf8')

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('prop_limit', 0.9, 'limit predict prop')


"""
predict labels, the operation of transfer words to id token is processed by numpy
"""
def words_predict(words_list):
  num_classes = FLAGS.num_classes
  num_layers = FLAGS.num_layers
  num_steps = FLAGS.num_steps
  embedding_size = FLAGS.embedding_size
  hidden_size = FLAGS.hidden_size
  keep_prob = FLAGS.keep_prob
  vocab_size = FLAGS.vocab_size
  vocab_path = FLAGS.vocab_path
  prop_limit = FLAGS.prop_limit
  checkpoint_path = FLAGS.checkpoint_path

  words_vocab, labels_vocab = data_utils.initialize_vocabulary(vocab_path)
  rev_labels_vocab = dict(izip(labels_vocab.itervalues(), labels_vocab.iterkeys()))

  inputs_data_list = []
  words_len = []
  for words in words_list:
    word_list = words.split()
    words_len.append(len(word_list))
    word_ids = data_utils.words_to_token_ids(word_list, words_vocab)
    word_ids_str = ' '.join([str(tok) for tok in word_ids])
    word_ids_padding = data_utils.align_word(word_ids_str, num_steps)
    inputs_data = np.array([int(tok) for tok in word_ids_padding.strip().split() if tok])
    inputs_data_list.append(inputs_data)

  predict_batch_size = len(inputs_data_list)
  inputs_placeholder = tf.placeholder(dtype=tf.int64, shape=(None, num_steps))
  with tf.variable_scope('model', reuse=None):
    logits, final_state = model.inference(inputs_placeholder, predict_batch_size, num_steps, vocab_size, embedding_size,
                                          hidden_size, keep_prob, num_layers, num_classes, is_training=False)
  # using softmax
  # props = tf.nn.softmax(logits)
  # max_prop_values, max_prop_indices = tf.nn.top_k(props, k=1)
  # predict_props = tf.reshape(max_prop_values, shape=[predict_batch_size, num_steps])
  # predict_indices = tf.reshape(max_prop_indices, shape=[predict_batch_size, num_steps])

  # using crf
  logits = tf.reshape(logits, shape=[predict_batch_size, num_steps, num_classes])
  transition_params = tf.get_variable("transitions", [num_classes, num_classes])
  sequence_length = tf.constant([num_steps] * predict_batch_size, dtype=tf.int64)
  predict_indices, _ = crf_utils.crf_decode(logits, transition_params, sequence_length)
  predict_props = tf.constant(1.0, shape=predict_indices.get_shape(), dtype=tf.float32)

  saver = tf.train.Saver()
  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(checkpoint_path)
    if ckpt and ckpt.model_checkpoint_path:
      print('read model from {}'.format(ckpt.model_checkpoint_path))
      saver.restore(sess, ckpt.model_checkpoint_path)
    else:
      print('No checkpoint file found at %s' % checkpoint_path)
      return

    predict_indices_list, predict_props_list = sess.run([predict_indices, predict_props], feed_dict={inputs_placeholder: inputs_data_list})

    # crf numpy
    # logits_val, transition = sess.run([logits, transition_params], feed_dict={inputs_placeholder: inputs_data_list})
    # predict_indices, _ = crf.viterbi_decode(logits_val, transition)
    # predict_indices_list = np.reshape(predict_indices, newshape=[predict_batch_size, num_steps])
    # predict_props_list = np.ones((predict_batch_size, num_steps))

    assert len(words_list) == len(predict_indices_list)
    predict_labels_list = []
    for (words, label_ids_list, prop_list, word_len) in zip(words_list, predict_indices_list, predict_props_list, words_len):
      predict_label_list = []
      for label_ids, prop in zip(label_ids_list[:word_len], prop_list[:word_len]):
        if prop >= prop_limit:
          predict_label_list.append(rev_labels_vocab.get(label_ids, data_utils.UNK_ID).encode('utf-8'))
        else:
          predict_label_list.append(rev_labels_vocab.get(4, data_utils.UNK_ID).encode('utf-8'))
      if word_len > num_steps:
        predict_label_list.extend(['O' for _ in range(word_len - num_steps)])
      predict_labels_list.append(' '.join(predict_label_list))
    return predict_labels_list


"""
predict labels, the operation of transfer words to id token is processed by tensorflow tensor
input words list, now, only support one element list
"""
def tensor_predict(words_list):
  num_classes = FLAGS.num_classes
  num_layers = FLAGS.num_layers
  num_steps = FLAGS.num_steps
  embedding_size = FLAGS.embedding_size
  hidden_size = FLAGS.hidden_size
  keep_prob = FLAGS.keep_prob
  vocab_size = FLAGS.vocab_size
  vocab_path = FLAGS.vocab_path
  prop_limit = FLAGS.prop_limit
  checkpoint_path = FLAGS.checkpoint_path

  # split 1-D String dense Tensor to words SparseTensor
  sentences = tf.placeholder(dtype=tf.string, shape=[None], name='input_sentences')
  sparse_words = tf.string_split(sentences, delimiter=' ')

  # slice SparseTensor
  valid_indices = tf.less(sparse_words.indices, tf.constant([num_steps], dtype=tf.int64))
  valid_indices = tf.reshape(tf.split(valid_indices, [1, 1], axis=1)[1], [-1])
  valid_sparse_words = tf.sparse_retain(sparse_words, valid_indices)

  excess_indices = tf.greater_equal(sparse_words.indices, tf.constant([num_steps], dtype=tf.int64))
  excess_indices = tf.reshape(tf.split(excess_indices, [1, 1], axis=1)[1], [-1])
  excess_sparse_words = tf.sparse_retain(sparse_words, excess_indices)

  # sparse to dense
  words = tf.sparse_to_dense(sparse_indices=valid_sparse_words.indices,
                             output_shape=[valid_sparse_words.dense_shape[0], num_steps],
                             sparse_values=valid_sparse_words.values,
                             default_value='_PAD')

  # dict words to token ids
  # with open(os.path.join(vocab_path, 'words_vocab.txt'), 'r') as data_file:
  #   words_table_list = [line.strip() for line in data_file if line.strip()]
  # words_table_tensor = tf.constant(words_table_list, dtype=tf.string)
  # words_table = lookup.index_table_from_tensor(mapping=words_table_tensor, default_value=3)
  words_table = lookup.index_table_from_file(os.path.join(vocab_path, 'words_vocab.txt'), default_value=3)
  words_ids = words_table.lookup(words)

  # blstm model predict
  with tf.variable_scope('model', reuse=None):
    logits, _ = model.inference(words_ids, valid_sparse_words.dense_shape[0], num_steps, vocab_size, embedding_size,
                                hidden_size, keep_prob, num_layers, num_classes, is_training=False)

  # using softmax
  # props = tf.nn.softmax(logits)
  # max_prop_values, max_prop_indices = tf.nn.top_k(props, k=1)
  # predict_scores = tf.reshape(max_prop_values, shape=[-1, num_steps])
  # predict_labels_ids = tf.reshape(max_prop_indices, shape=[-1, num_steps])
  # predict_labels_ids = tf.to_int64(predict_labels_ids)

  # using crf
  logits = tf.reshape(logits, shape=[-1, num_steps, num_classes])
  transition_params = tf.get_variable("transitions", [num_classes, num_classes])
  sequence_length = tf.constant(num_steps, shape=[logits.get_shape()[0]], dtype=tf.int64)
  predict_labels_ids, _ = crf_utils.crf_decode(logits, transition_params, sequence_length)
  predict_labels_ids = tf.to_int64(predict_labels_ids)
  predict_scores = tf.constant(1.0, shape=predict_labels_ids.get_shape(), dtype=tf.float32)

  # replace untrusted prop that less than prop_limit
  trusted_prop_flag = tf.greater_equal(predict_scores, tf.constant(prop_limit, dtype=tf.float32))
  replace_prop_labels_ids = tf.to_int64(tf.fill(tf.shape(predict_labels_ids), 4))
  predict_labels_ids = tf.where(trusted_prop_flag, predict_labels_ids, replace_prop_labels_ids)

  # dict token ids to labels
  # with open(os.path.join(vocab_path, 'labels_vocab.txt'), 'r') as data_file:
  #   labels_table_list = [line.strip() for line in data_file if line.strip()]
  # labels_table_tensor = tf.constant(labels_table_list, dtype=tf.string)
  # labels_table = lookup.index_to_string_table_from_tensor(mapping=labels_table_tensor, default_value='O')
  labels_table = lookup.index_to_string_table_from_file(os.path.join(vocab_path, 'labels_vocab.txt'), default_value='O')
  predict_labels = labels_table.lookup(predict_labels_ids)

  # extract real blstm predict label in dense and save to sparse
  valid_sparse_predict_labels = tf.SparseTensor(indices=valid_sparse_words.indices,
                                                values=tf.gather_nd(predict_labels, valid_sparse_words.indices),
                                                dense_shape=valid_sparse_words.dense_shape)

  # create excess label SparseTensor with 'O'
  excess_sparse_predict_labels = tf.SparseTensor(indices=excess_sparse_words.indices,
                                                 values=tf.fill(tf.shape(excess_sparse_words.values), 'O'),
                                                 dense_shape=excess_sparse_words.dense_shape)

  # concat SparseTensor
  sparse_predict_labels = tf.SparseTensor(
    indices=tf.concat(axis=0, values=[valid_sparse_predict_labels.indices, excess_sparse_predict_labels.indices]),
    values=tf.concat(axis=0, values=[valid_sparse_predict_labels.values, excess_sparse_predict_labels.values]),
    dense_shape=excess_sparse_predict_labels.dense_shape)
  sparse_predict_labels = tf.sparse_reorder(sparse_predict_labels)

  # join SparseTensor to 1-D String dense Tensor
  # remain issue, num_split should equal the real size, but here limit to 1
  join_labels_list = []
  slice_labels_list = tf.sparse_split(sp_input=sparse_predict_labels, num_split=1, axis=0)
  for slice_labels in slice_labels_list:
    slice_labels = slice_labels.values
    join_labels = tf.reduce_join(slice_labels, reduction_indices=0, separator=' ')
    join_labels_list.append(join_labels)
  format_predict_labels = tf.stack(join_labels_list, name='predict_labels')

  saver = tf.train.Saver()
  tables_init_op = tf.tables_initializer()
  with tf.Session() as sess:
    sess.run(tables_init_op)
    ckpt = tf.train.get_checkpoint_state(checkpoint_path)
    if ckpt and ckpt.model_checkpoint_path:
      print('read model from {}'.format(ckpt.model_checkpoint_path))
      saver.restore(sess, ckpt.model_checkpoint_path)
    else:
      print('No checkpoint file found at %s' % checkpoint_path)
      return
    # crf tensor
    predict_labels_list = sess.run(format_predict_labels, feed_dict={sentences: words_list})
    # save graph into .pb file
    graph = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, ["init_all_tables", "predict_labels"])
    tf.train.write_graph(graph, '.', 'ner_graph.pb', as_text=False)
    return predict_labels_list


"""
predict data_filename, save the predict result into predict_filename
"""
def file_predict(data_filename, predict_filename):
  print 'Predict file ' + data_filename
  words_list = []
  labels_list = []
  with open(data_filename, mode='r') as data_file:
    for line in data_file:
      word_list, label_list = data_utils.split(line)
      if word_list and label_list:
        words_list.append(' '.join(word_list))
        labels_list.append(' '.join(label_list))
  predict_labels_list = words_predict(words_list)
  with open(predict_filename, mode='w') as predict_file:
    for (words, labels, predict_labels) in zip(words_list, labels_list, predict_labels_list):
      predict_file.write('Passage: ' + words + '\n')
      predict_file.write('Label: ' + labels + '\n')
      predict_file.write('PredictLabel: ' + predict_labels + '\n' + '\n')


"""
predict data_filename, save the predict result into predict_filename
the label is split into single word, -B -M -E -S
"""
def single_word_file_predict(data_filename, predict_filename):
  print 'Predict file ' + data_filename
  sentence_list = []
  words_list = []
  labels_list = []
  with open(data_filename, mode='r') as data_file:
    for line in data_file:
      word_list, label_list = data_utils.split(line)
      if word_list and label_list:
        sentence_list.append(''.join(word_list))
        words_list.append(' '.join(word_list))
        labels_list.append(' '.join(label_list))
  predict_labels_list = words_predict(words_list)
  word_predict_label_list = []
  word_category_list = []
  word_predict_category_list = []
  for (words, labels, predict_labels) in zip(words_list, labels_list, predict_labels_list):
    word_list = words.split()
    label_list = labels.split()
    predict_label_list = predict_labels.split()
    word_predict_label = ' '.join([word + '/' + predict_label for (word, predict_label) in zip(word_list, predict_label_list)])
    word_predict_label_list.append(word_predict_label)
    # merge label
    merge_word_list, merge_label_list = data_utils.merge_label(word_list, label_list)
    word_category = ' '.join([word + '/' + label for (word, label) in zip(merge_word_list, merge_label_list) if label != 'O'])
    word_category_list.append(word_category)
    # merge predict label
    merge_predict_word_list, merge_predict_label_list = data_utils.merge_label(word_list, predict_label_list)
    word_predict_category = ' '.join([predict_word + '/' + predict_label for (predict_word, predict_label) in
                                      zip(merge_predict_word_list, merge_predict_label_list) if predict_label != 'O'])
    word_predict_category_list.append(word_predict_category)
  with open(predict_filename, mode='w') as predict_file:
    for (sentence, word_predict_label, word_predict_category, word_category) in \
        zip(sentence_list, word_predict_label_list, word_category_list, word_predict_category_list):
      predict_file.write('Passage: ' + sentence + '\n')
      predict_file.write('SinglePredict: ' + word_predict_label + '\n')
      predict_file.write('Merge: ' + word_category + '\n')
      predict_file.write('MergePredict: ' + word_predict_category + '\n')
      predict_file.write('\n')


def main(_):
  datasets_path = FLAGS.datasets_path

  words_list = ['范冰冰 在 娱乐圈 拥有 很多 粉丝',
                '赵丽颖 在 电视剧 花千骨 中 扮演 女主角',
                '明天 去 看 电影 长城',
                '雷军 是 小米 公司 创始人',
                '《 速度与激情 》 总 车辆 建筑 损失 费用 超过 5亿美元',
                ]
  # predict_labels_list = words_predict(words_list)
  # for predict_labels in predict_labels_list:
  #   print predict_labels

  words_list = ['范冰冰 在 娱乐圈 拥有 很多 粉丝']
  # predict_labels_list = tensor_predict(words_list)
  # for predict_labels in predict_labels_list:
  #   print predict_labels

  # file predict
  # file_predict(os.path.join(datasets_path, 'test.txt'), os.path.join(datasets_path, 'test_predict.txt'))

  # file predict, and the label is split into single word
  single_word_file_predict(os.path.join(datasets_path, 'test.txt'), os.path.join(datasets_path, 'test_predict.txt'))


if __name__ == '__main__':
  tf.app.run()