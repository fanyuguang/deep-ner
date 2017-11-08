#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import os
import re
import sys
import random
import tensorflow as tf

reload(sys)
sys.setdefaultencoding('utf8')

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('raw_data_path', 'data/raw-data/', 'raw data directory')
tf.app.flags.DEFINE_string('datasets_path', 'data/datasets/', 'datasets directory')
tf.app.flags.DEFINE_string('vocab_path', 'data/vocab/', 'vocab directory')
tf.app.flags.DEFINE_integer('vocab_size', 10000, 'vocab size')
tf.app.flags.DEFINE_float('train_percent', 0.8, 'train percent')
tf.app.flags.DEFINE_float('val_percent', 0.1, 'val test percent')

_PAD = b'_PAD'
_GO = b'_GO'
_EOS = b'_EOS'
_UNK = b'_UNK'
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

SPECIFIC_SYMBOLS = [u'《', u'》', u'〗', u'↔', u'͓', u'｀', u'¥', u'⑦', u'√', u'⑨', u'▼', u'⑥', u'⑤', u'〖', u'Ⓜ',
                    u'⑩', u'⛴', u'☘', u'■', u'［', u'〕', u'◇', u'⏰', u'⛈', u'➕', u'▪', u'❤', u'☞', u'④', u'⊙',
                    u'③', u'』', u'⭐', u'『', u'②', u'①', u'★', u'」', u'「', u'【', u'】', u'#', u'\ufe0f']
VALID_LABELS = set([u'CELEBRITY', u'PASTIME', u'BOOK', u'MERCHANT', u'ADDRESS'])


"""
split sentence with format 'word1/label1||word2/label2' to word list and label list
"""
def split(sentence):
  if not isinstance(sentence, unicode):
    sentence = sentence.decode('utf-8')
  sentence = sentence.strip()
  sentence = ''.join(sentence.split())
  word_label_list = re.split('\|{2}', sentence)
  word_list = []
  label_list = []
  for word_label in word_label_list:
    if word_label:
      word_label_pair = re.split('/', word_label)
      if len(word_label_pair) >= 2:
        word = (''.join(word_label_pair[:-1]))
        label = word_label_pair[-1]
        if word and label:
          word_list.append(word)
          label_list.append(label)
  return word_list, label_list


"""
upper label, and delete invalid label
delete specific symbols, for example, '《', '》'
add the sentences delete specific symbols into new data list
"""
def format_data(data_filename, format_data_filename):
  print 'Format data file ' + data_filename
  format_data_list = []
  with open(data_filename, 'r') as data_file:
    for line in data_file:
      word_label_list = []
      exist_specific = False
      delete_specific_list = []
      word_list, label_list = split(line)
      if word_list and label_list:
        for word, label in zip(word_list, label_list):
          # format label
          label = label.upper()
          if label not in VALID_LABELS:
            label = 'O'
          word_label_list.append(word + '/' + label)
          # format word
          if word in SPECIFIC_SYMBOLS:
            exist_specific = True
          else:
            delete_specific_list.append(word + '/' + label)
        if word_label_list:
          format_data_list.append('||'.join(word_label_list))
        if exist_specific and delete_specific_list:
          format_data_list.append('||'.join(delete_specific_list))
  with open(format_data_filename, 'w') as format_data_file:
    for data in format_data_list:
      format_data_file.write(data + '\n')


"""
split label of sentence to -B -M -E -S
"""
def split_label(sentence):
  single_word_label_list = []
  word_list, label_list = split(sentence)
  if word_list and label_list:
    for word, label in zip(word_list, label_list):
      single_label_list = []
      word_length = len(word)
      if label == 'O':
        single_label_list = ['O' for _ in range(word_length)]
      else:
        if word_length == 1:
          single_label_list = [label + '-S']
        elif word_length == 2:
          single_label_list = [label + '-B', label + '-E']
        else:
          single_label_list.append(label + '-B')
          for index in range(word_length - 2):
            single_label_list.append(label + '-M')
          single_label_list.append(label + '-E')
      for (single_word, single_label) in zip(word, single_label_list):
        single_word_label = single_word + '/' + single_label
        single_word_label_list.append(single_word_label)
  split_sentence = '||'.join(single_word_label_list)
  return split_sentence


"""
split label of file to -B -M -E -S
"""
def split_label_file(data_filename, split_data_filename):
  print 'Split label file ' + data_filename
  with open(split_data_filename, 'w') as new_data_file:
    with open(data_filename, mode='r') as raw_data_file:
      for line in raw_data_file:
        sentence = split_label(line.strip())
        new_data_file.write(sentence + '\n')


"""
merge split label, example label-B label-M label-E label-S to label
"""
def merge_label(word_list, label_list):
  merge_word_list = []
  merge_label_list = []
  category = ''
  category_word_list = []
  for (word, label) in zip(word_list, label_list):
    if word and label:
      if (len(label) > 1 and label.find('-B') == len(label) - 2):
        if category_word_list:
          merge_word_list.extend(category_word_list)
          merge_label_list.extend(['O'] * len(category_word_list))
          category_word_list = []
        category = label[0:-2]
        category_word_list.append(word)
      elif (len(label) > 1 and label.find('-M') == len(label) - 2):
        category_word_list.append(word)
        if category != label[0:-2]:
          merge_word_list.extend(category_word_list)
          merge_label_list.extend(['O'] * len(category_word_list))
          category = ''
          category_word_list = []
      elif (len(label) > 1 and label.find('-E') == len(label) - 2):
        category_word_list.append(word)
        if category == label[0:-2]:
          merge_word_list.append(''.join(category_word_list))
          merge_label_list.append(category)
        else:
          merge_word_list.extend(category_word_list)
          merge_label_list.extend(['O'] * len(category_word_list))
        category = ''
        category_word_list = []
      elif (len(label) > 1 and label.find('-S') == len(label) - 2):
        if category_word_list:
          merge_word_list.extend(category_word_list)
          merge_label_list.extend(['O'] * len(category_word_list))
          category_word_list = []
        category = label[0:-2]
        merge_word_list.append(word)
        merge_label_list.append(category)
        category = ''
      elif (label == 'O'):
        category_word_list.append(word)
        merge_word_list.extend(category_word_list)
        merge_label_list.extend(['O'] * len(category_word_list))
        category = ''
        category_word_list = []
      else:
        raise ValueError('Merge_label input exists invalid data.')
  return merge_word_list, merge_label_list


"""
split dataset to train set, validation set, test set
store sets into datasets dir
"""
def prepare_datasets(raw_data_filename, train_percent, val_percent, datasets_path):
  print 'Prepare datasets ' + raw_data_filename
  data_list = []
  with open(raw_data_filename, mode='r') as raw_data_file:
    for line in raw_data_file:
      line = line.strip()
      if line:
        data_list.append(line)
  random.shuffle(data_list)

  data_size = len(data_list)
  train_validation_index = int(data_size * train_percent)
  validation_test_index = int(data_size * (train_percent + val_percent))

  train_data_list = data_list[:train_validation_index]
  validation_data_list = data_list[train_validation_index:validation_test_index]
  test_data_list = data_list[validation_test_index:]

  list_to_file(train_data_list, os.path.join(datasets_path, 'train.txt'))
  list_to_file(validation_data_list, os.path.join(datasets_path, 'validation.txt'))
  list_to_file(test_data_list, os.path.join(datasets_path, 'test.txt'))


"""
count word and label from file
"""
def count_vocabulary(data_filename):
  words_count = {}
  labels_count = {}
  label_words_vocab = {}
  with open(data_filename, mode='r') as data_file:
    for line in data_file:
      word_list, label_list = split(line)
      for (word, label) in zip(word_list, label_list):
        if word in words_count:
          words_count[word] += 1
        else:
          words_count[word] = 1
        if label in labels_count:
          labels_count[label] += 1
        else:
          labels_count[label] = 1
        if label in label_words_vocab:
          if word not in label_words_vocab[label]:
            label_words_vocab[label].append(word)
        else:
          label_words_vocab[label] = [word]
  return words_count, labels_count, label_words_vocab


"""
write list into file, one element per line
"""
def list_to_file(data_list, data_filename):
  with open(data_filename, mode='w') as data_file:
    for data in data_list:
      data = data.strip()
      if data:
        data_file.write(data + '\n')


def sort_vocabulary(vocab_count, vocab_filename, vocab_size):
  vocab_list = _START_VOCAB + sorted(vocab_count, key=vocab_count.get, reverse=True)
  if len(vocab_list) > vocab_size:
    vocab_list = vocab_list[:vocab_size]
  list_to_file(vocab_list, vocab_filename)
  vocab = dict([(x, y) for (y, x) in enumerate(vocab_list)])
  return vocab


"""
count and create vocabulary of word and label
store vocabulary into vocab dir
"""
def create_vocabulary(data_filename, vocab_path, vocab_size):
  print 'Creating vocabulary ' + data_filename
  words_count, labels_count, label_words_vocab = count_vocabulary(data_filename)
  words_vocab = sort_vocabulary(words_count, os.path.join(vocab_path, 'words_vocab.txt'), vocab_size)
  labels_vocab = sort_vocabulary(labels_count, os.path.join(vocab_path, 'labels_vocab.txt'), vocab_size)
  for labels, word_list in label_words_vocab.iteritems():
    list_to_file(word_list, os.path.join(vocab_path, (labels + '.txt')))
  return words_vocab, labels_vocab, label_words_vocab


"""
restore vocabulary from vocab file
"""
def initialize_single_vocabulary(vocab_filename):
  if os.path.exists(vocab_filename):
    data_list = []
    with open(vocab_filename, mode='r') as vocab_file:
      for line in vocab_file:
        data_list.append(line.strip().decode('utf-8'))
      data_list.extend(vocab_file.readlines())
    vocab = dict([(x, y) for (y, x) in enumerate(data_list)])
    return vocab
  else:
    raise ValueError('Vocabulary file %s not found.', vocab_filename)


"""
restore vocabulary of word and label from vocab file
"""
def initialize_vocabulary(vocab_path):
  print 'Initialize vocabulary ' + vocab_path
  words_vocab = initialize_single_vocabulary(os.path.join(vocab_path, 'words_vocab.txt'))
  labels_vocab = initialize_single_vocabulary(os.path.join(vocab_path, 'labels_vocab.txt'))
  return words_vocab, labels_vocab


"""
transfer words to id token
"""
def words_to_token_ids(word_list, vocab):
  return [vocab.get(word.decode('utf-8'), UNK_ID) for word in word_list]


"""
split sentence to words and labels, and transfer it to id token
"""
def sentence_to_token_ids(sentence, words_vocab, labels_vocab):
  word_list, label_list = split(sentence)
  word_ids = words_to_token_ids(word_list, words_vocab)
  label_ids = words_to_token_ids(label_list, labels_vocab)
  assert len(word_ids) == len(label_ids)
  return word_ids, label_ids


"""
transfer file to id token
"""
def data_to_token_ids(data_filename, words_vocab, labels_vocab):
  print 'Tokenizing data in ' + data_filename
  word_ids_list = []
  label_ids_list = []
  with open(data_filename, mode='r') as data_file:
    for line in data_file:
      word_ids, label_ids = sentence_to_token_ids(line, words_vocab, labels_vocab)
      if word_ids and label_ids:
        word_ids_list.append(' '.join([str(tok) for tok in word_ids]))
        label_ids_list.append(' '.join([str(tok) for tok in label_ids]))
  return word_ids_list, label_ids_list


"""
align length of words to align_size
"""
def align_word(words, align_size):
  word_list = words.strip().split()
  words_count = len(word_list)
  if words_count < align_size:
    padding = ' '.join([str(PAD_ID) for _ in range(align_size - words_count)])
    if words_count:
      return (words + ' ' + padding)
    else:
      return padding
  else:
    words_truncate = ' '.join(word_list[0:align_size])
    return words_truncate


"""
judge two words is same
"""
def judge_same_word(word, predict_word):
  word_matrix = [[0 for _ in range(len(predict_word) + 1)] for _ in range(len(word) + 1)]
  common_max = 0
  for i in range(len(word)):
    for j in range(len(predict_word)):
      if word[i] == predict_word[j]:
        word_matrix[i + 1][j + 1] = word_matrix[i][j] + 1
        if word_matrix[i + 1][j + 1] > common_max:
          common_max = word_matrix[i + 1][j + 1]
  min_len = len(word) if len(word) < len(predict_word) else len(predict_word)
  if common_max >= 6 or common_max == min_len:
    return True
  else:
    return False


def main():
  raw_data_path = FLAGS.raw_data_path
  datasets_path = FLAGS.datasets_path
  vocab_path = FLAGS.vocab_path
  vocab_size = FLAGS.vocab_size
  train_percent = FLAGS.train_percent
  val_percent = FLAGS.val_percent

  format_data(os.path.join(raw_data_path, 'data.txt'), os.path.join(raw_data_path, 'format_data.txt'))
  split_label_file(os.path.join(raw_data_path, 'format_data.txt'), os.path.join(raw_data_path, 'split_data.txt'))
  prepare_datasets(os.path.join(raw_data_path, 'split_data.txt'), train_percent, val_percent, datasets_path)
  vocab = create_vocabulary(os.path.join(datasets_path, 'train.txt'), vocab_path, vocab_size)


if __name__ == '__main__':
  main()