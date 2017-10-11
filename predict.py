#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
import os
import sys
from itertools import izip
import numpy as np
import tensorflow as tf
import tensorflow.contrib.crf as crf
import data_utils
import tfrecords_utils
import model
import train

reload(sys)
sys.setdefaultencoding('utf8')

FLAGS = tf.app.flags.FLAGS
# tf.app.flags.DEFINE_float('prop_limit', 0.99999, 'limit predict prop')
tf.app.flags.DEFINE_float('prop_limit', 0.9, 'limit predict prop')


def file_to_ner(data_filename, predict_filename):
  print 'Tokenizing data in ' + data_filename
  num_steps = FLAGS.num_steps

  words_list = []
  labels_list = []
  with open(data_filename, mode='r') as data_file:
    for line in data_file:
      _, word_list, label_list = data_utils.split(line)
      if word_list and label_list:
        words_list.append(' '.join([word for word in word_list]))
        if len(label_list) > num_steps:
          label_list = label_list[:num_steps]
        labels_list.append(' '.join([label for label in label_list]))
  for words in words_list:
    print words.decode('string_escape')
  predict_labels_list = predict(words_list)
  with open(predict_filename, mode='w') as predict_file:
    for (words, predict_labels) in zip(words_list, predict_labels_list):
      predict_file.write(words + b'\n')
      predict_file.write('[' + predict_labels + ']' + b'\n')
  return labels_list, predict_labels_list


def predict(words_list):
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
  props = tf.nn.softmax(logits)
  max_prop_values, max_prop_indices = tf.nn.top_k(props, k=1)
  predict_props = tf.reshape(max_prop_values, shape=[predict_batch_size, num_steps])
  predict_indices = tf.reshape(max_prop_indices, shape=[predict_batch_size, num_steps])

  # crf
  # transition_params = tf.get_variable("transitions", [num_classes, num_classes])

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

    # crf
    # result, transition = sess.run([logits, transition_params], feed_dict={inputs_placeholder: inputs_data_list})
    # predict_indices, _ = crf.viterbi_decode(result, transition)
    # predict_indices_list = np.reshape(predict_indices, newshape=[predict_batch_size, num_steps])
    # predict_props_list = np.ones((predict_batch_size, num_steps))

    assert len(words_list) == len(predict_indices_list)
    predict_labels_list = []
    for (words, label_ids_list, prop_list, word_len) in zip(words_list, predict_indices_list, predict_props_list, words_len):
      print words.decode('string_escape')
      predict_label_list = []
      for label_ids, prop in zip(label_ids_list[:word_len], prop_list[:word_len]):
        if prop >= prop_limit:
          predict_label_list.append(rev_labels_vocab.get(label_ids, data_utils.UNK_ID).encode('utf-8'))
        else:
          predict_label_list.append(rev_labels_vocab.get(4, data_utils.UNK_ID).encode('utf-8'))
      print predict_label_list
      predict_labels_list.append(' '.join(predict_label_list))
    return predict_labels_list


def evaluate(labels_list, predict_labels_list):
  celebrity_count = 0
  predict_celebrity_count = 0
  common_celebrity_count = 0
  pastime_count = 0
  predict_pastime_count = 0
  common_pastime_count = 0
  merchant_count = 0
  predict_merchant_count = 0
  common_merchant_count = 0
  book_count = 0
  predict_book_count = 0
  common_book_count = 0

  label_list = []
  for labels in labels_list:
    label_list.extend(labels.upper().split())
  predict_label_list = []
  for predict_labels in predict_labels_list:
    predict_label_list.extend(predict_labels.split())

  assert len(label_list) == len(predict_label_list)
  for (label, predict_label) in zip(label_list, predict_label_list):
    if label == 'CELEBRITY':
      celebrity_count += 1
    elif label == 'PASTIME':
      pastime_count += 1
    elif label == 'MERCHANT':
      merchant_count += 1
    elif label == 'BOOK':
      book_count += 1

    if predict_label == 'CELEBRITY':
      predict_celebrity_count += 1
    elif predict_label == 'PASTIME':
      predict_pastime_count += 1
    elif predict_label == 'MERCHANT':
      predict_merchant_count += 1
    elif predict_label == 'BOOK':
      predict_book_count += 1

    if label == 'CELEBRITY' and predict_label == 'CELEBRITY':
      common_celebrity_count += 1
    elif label == 'PASTIME' and predict_label == 'PASTIME':
      common_pastime_count += 1
    elif label == 'MERCHANT' and predict_label == 'MERCHANT':
      common_merchant_count += 1
    elif label == 'BOOK' and predict_label == 'BOOK':
      common_book_count += 1

  min_num = 0.0000000000001
  celebrity_score = {}
  celebrity_score['precision_score'] = common_celebrity_count / (predict_celebrity_count + min_num)
  celebrity_score['recall_score'] = common_celebrity_count / (celebrity_count + min_num)
  celebrity_score['f_score'] = celebrity_score['precision_score'] * celebrity_score['recall_score'] * 2 / \
                               (celebrity_score['precision_score'] + celebrity_score['recall_score'] + min_num)
  pastime_score = {}
  pastime_score['precision_score'] = common_pastime_count / (predict_pastime_count + min_num)
  pastime_score['recall_score'] = common_pastime_count / (pastime_count + min_num)
  pastime_score['f_score'] = pastime_score['precision_score'] * pastime_score['recall_score'] * 2 / \
                             (pastime_score['precision_score'] + pastime_score['recall_score'] + min_num)
  merchant_score = {}
  merchant_score['precision_score'] = common_merchant_count / (predict_merchant_count + min_num)
  merchant_score['recall_score'] = common_merchant_count / (merchant_count + min_num)
  merchant_score['f_score'] = merchant_score['precision_score'] * merchant_score['recall_score'] * 2 / \
                             (merchant_score['precision_score'] + merchant_score['recall_score'] + min_num)
  book_score = {}
  book_score['precision_score'] = common_book_count / (predict_book_count + min_num)
  book_score['recall_score'] = common_book_count / (book_count + min_num)
  book_score['f_score'] = book_score['precision_score'] * book_score['recall_score'] * 2 / \
                             (book_score['precision_score'] + book_score['recall_score'] + min_num)
  all_score = {}
  all_score['precision_score'] = (common_celebrity_count + common_pastime_count + common_merchant_count + common_book_count) / \
                                 (predict_celebrity_count + predict_pastime_count + predict_merchant_count + predict_book_count + min_num)
  all_score['recall_score'] = (common_celebrity_count + common_pastime_count + common_merchant_count + common_book_count) / \
                              (celebrity_count + pastime_count + merchant_count + book_count + min_num)
  all_score['f_score'] = all_score['precision_score'] * all_score['recall_score'] * 2 / \
                         (all_score['precision_score'] + all_score['recall_score'] + min_num)
  print 'celebrity_score : '
  print celebrity_score
  print 'pastime_score : '
  print pastime_score
  print 'merchant_score : '
  print merchant_score
  print 'book_score : '
  print book_score
  print 'all_score : '
  print all_score
  return celebrity_score, pastime_score, merchant_score, book_score, all_score


def main(_):
  datasets_path = FLAGS.datasets_path

  words_list = ['朱玲玲 是 港姐 出身',
                '范冰冰 在 娱乐圈 拥有 很多 粉丝',
                '赵丽颖 在 电视剧 花千骨 中 扮演 女主角',
                '叫上 小明 明天 去 长城 玩',
                '明天 去 看 电影 长城',
                '雷军 是 小米 公司 创始人',
                '武林外传 主演 同剧 不同命 ， 一半 已 淡出 娱乐圈',
                '2月 20 日 , 游轮 来到 意大利 一个 港口 , 游轮 员工 清点 乘客 时 发现 , 李英蕾 竟然 不见 踪影 。',
                '小米 手机 发布会 闹 乌 龙 ， ppt 有 2处 明显 错误 ， 但 雷军 也 没 发现',
                '陈羽凡 上海 组 露天 酒局 与 白百何 同 城 不见 疑 婚变',
                '《 速度与激情 》 总 车辆 建筑 损失 费用 超过 5亿美元',
                '足 总 杯 - 威廉 2 球 阿扎尔 传 射 切尔西 4- 2 热 刺 进 决赛',
                '联 杯 下 周 开 打 ： 科 贝尔 领衔 哈勒普 辛吉斯 均 将 出 战',
                '有 哪些 好看 的 俄罗斯 电影 ？'
                ]
  predict_labels_list = predict(words_list)

  # labels_list, predict_labels_list = file_to_ner(os.path.join(datasets_path, 'test.txt'),
  #                                                os.path.join(datasets_path, 'test_predict.txt'))
  # celebrity_score, pastime_score, merchant_score, book_score, all_score = evaluate(labels_list, predict_labels_list)

if __name__ == '__main__':
  tf.app.run()
