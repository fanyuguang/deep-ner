#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
import os
import sys
import tensorflow as tf
import data_utils

reload(sys)
sys.setdefaultencoding('utf8')

FLAGS = tf.app.flags.FLAGS


"""
evaluate the score of model predict, data_filename is created by file_predict
"""
def evaluate(data_filename):
  print 'Evaluate file ' + data_filename
  label_count = dict()
  with open(data_filename, 'r') as data_file:
    for line in data_file:
      line = line.strip()
      if line and line[:6] == 'Label:':
        labels = line.split()[1:]
      if line and line[:13] == 'PredictLabel:':
        predict_labels = line.split()[1:]
        if labels and predict_labels:
          for label, predict_label in zip(labels, predict_labels):
            if label not in label_count:
              label_count[label] = [0] * 3
            if predict_label not in label_count:
              label_count[predict_label] = [0] * 3
            # index 0 is label, index 1 is predict label, index 2 is common label
            label_count[label][0] += 1
            label_count[predict_label][1] += 1
            if label == predict_label:
              label_count[label][2] += 1
  min_num = 0.0000000000001
  label_scores = dict()
  for label, count in label_count.items():
    precision_score = count[2] / (count[1] + min_num)
    recall_score = count[2] / (count[0] + min_num)
    f_score = precision_score * recall_score * 2 / (precision_score + recall_score + min_num)
    label_scores[label] = [precision_score, recall_score, f_score]
    print label + ': [precision: ' + str(precision_score) + ', recall: ' + str(recall_score) + ', f_score: ' + str(f_score) + ']'


"""
evaluate the score of model predict, data_filename is created by single_word_file_predict
"""
def single_word_file_evaluate(data_filename):
  print 'Evaluate file ' + data_filename
  label_count = dict()
  with open(data_filename, 'r') as data_file:
    for line in data_file:
      line = line.strip()
      if line and line[:6] == 'Merge:':
        word_labels = line.split()[1:]
      if line and line[:13] == 'MergePredict:':
        word_predict_labels = line.split()[1:]
        word_list, label_list = data_utils.split('||'.join(word_labels))
        predict_word_list, predict_label_list = data_utils.split('||'.join(word_predict_labels))
        # index 0 is label, index 1 is predict label, index 2 is common label
        while word_list and label_list and predict_word_list and predict_label_list:
          word = word_list[0]
          label = label_list[0]
          if label not in label_count:
            label_count[label] = [0] * 3
          label_count[label][0] += 1
          predict_index = -1
          is_find_same = False
          for predict_word, predict_label in zip(predict_word_list, predict_label_list):
            predict_index += 1
            if label == predict_label and data_utils.judge_same_word(word, predict_word):
              label_count[predict_label][1] += 1
              label_count[predict_label][2] += 1
              is_find_same = True
              break
          del word_list[0]
          del label_list[0]
          if is_find_same:
            del predict_word_list[predict_index]
            del predict_label_list[predict_index]
        for label in label_list:
          if label not in label_count:
            label_count[label] = [0] * 3
          label_count[label][0] += 1
        for predict_label in predict_label_list:
          if predict_label not in label_count:
            label_count[predict_label] = [0] * 3
          label_count[predict_label][1] += 1
  min_num = 0.0000000000001
  label_scores = dict()
  for label, count in label_count.items():
    precision_score = count[2] / (count[1] + min_num)
    recall_score = count[2] / (count[0] + min_num)
    f_score = precision_score * recall_score * 2 / (precision_score + recall_score + min_num)
    label_scores[label] = [precision_score, recall_score, f_score]
    print label + ': [precision: ' + str(precision_score) + ', recall: ' + str(recall_score) + ', f_score: ' + str(f_score) + ']'


def main(_):
  datasets_path = FLAGS.datasets_path
  # evaluate(os.path.join(datasets_path, 'test_predict.txt'))
  single_word_file_evaluate(os.path.join(datasets_path, 'test_predict.txt'))


if __name__ == '__main__':
  tf.app.run()