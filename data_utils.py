#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import os
import random
import re
import sys

import tensorflow as tf

reload(sys)
sys.setdefaultencoding('utf8')

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('raw_data_path', 'data/raw-data/', 'raw data directory')
tf.app.flags.DEFINE_string('datasets_path', 'data/datasets/', 'datasets directory')
tf.app.flags.DEFINE_string('vocab_path', 'data/vocab/', 'vocab directory')
tf.app.flags.DEFINE_integer('vocab_size', 200000, 'vocab size')
tf.app.flags.DEFINE_float('train_percent', 1.0, 'train percent')
tf.app.flags.DEFINE_float('val_percent', 0.0, 'val test percent')

_PAD = b'_PAD'
_GO = b'_GO'
_EOS = b'_EOS'
_UNK = b'_UNK'
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3


def split(sentence):
  if not isinstance(sentence, unicode):
    sentence = sentence.decode()
  id_index = sentence.find('\t')
  id = None
  if id_index != -1:
    id = sentence[:id_index].strip()
  else:
    id_index = 0
  sentence = sentence[id_index:].strip()
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
  return id, word_list, label_list


def count_vocabulary(data_filename):
  words_count = {}
  labels_count = {}
  label_words_vocab = {}
  with open(data_filename, mode='r') as data_file:
    for line in data_file:
      _, word_list, label_list = split(line.strip())
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
        print word + '/' + label + ' ',
      print ''
  return words_count, labels_count, label_words_vocab


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


def create_vocabulary(data_filename, vocab_path, vocab_size):
  print 'Creating vocabulary ' + data_filename
  words_count, labels_count, label_words_vocab = count_vocabulary(data_filename)
  words_vocab = sort_vocabulary(words_count, os.path.join(vocab_path, 'words_vocab.txt'), vocab_size)
  labels_vocab = sort_vocabulary(labels_count, os.path.join(vocab_path, 'labels_vocab.txt'), vocab_size)
  for labels, word_list in label_words_vocab.iteritems():
    list_to_file(word_list, os.path.join(vocab_path, (labels + '.txt')))
  return words_vocab, labels_vocab, label_words_vocab


def initialize_single_vocabulary(vocab_filename):
  if os.path.exists(vocab_filename):
    data_list = []
    with open(vocab_filename, mode='r') as vocab_file:
      for line in vocab_file:
        data_list.append(line.strip().decode())
      data_list.extend(vocab_file.readlines())
    vocab = dict([(x, y) for (y, x) in enumerate(data_list)])
    return vocab
  else:
    raise ValueError('Vocabulary file %s not found.', vocab_filename)


def initialize_vocabulary(vocab_path):
  print 'Initialize vocabulary ' + vocab_path
  words_vocab = initialize_single_vocabulary(os.path.join(vocab_path, 'words_vocab.txt'))
  labels_vocab = initialize_single_vocabulary(os.path.join(vocab_path, 'labels_vocab.txt'))
  return words_vocab, labels_vocab


def words_to_token_ids(word_list, vocab):
  return [vocab.get(word.decode(), UNK_ID) for word in word_list]


def sentence_to_token_ids(sentence, words_vocab, labels_vocab):
  _, word_list, label_list = split(sentence)
  word_ids = words_to_token_ids(word_list, words_vocab)
  label_ids = words_to_token_ids(label_list, labels_vocab)
  assert len(word_ids) == len(label_ids)
  return word_list, label_list, word_ids, label_ids


def data_to_token_ids(data_filename, words_vocab, labels_vocab):
  print 'Tokenizing data in ' + data_filename
  words_list = []
  labels_list = []
  word_ids_list = []
  label_ids_list = []
  with open(data_filename, mode='r') as data_file:
    for line in data_file:
      result = sentence_to_token_ids(line, words_vocab, labels_vocab)
      word_list, label_list, word_ids, label_ids = result
      if word_ids and label_ids and len(word_list) == len(word_ids):
        words_list.append(' '.join(word_list))
        labels_list.append(' '.join(label_list))
        word_ids_list.append(' '.join([str(tok) for tok in word_ids]))
        label_ids_list.append(' '.join([str(tok) for tok in label_ids]))
  return words_list, labels_list, word_ids_list, label_ids_list


def align_word(words, vector_size):
  word_list = words.strip().split()
  words_count = len(word_list)
  if words_count < vector_size:
    padding = ' '.join([str(PAD_ID) for _ in range(vector_size - words_count)])
    if words_count:
      return (words + ' ' + padding)
    else:
      return padding
  else:
    words_padding = ' '.join([word for index, word in enumerate(word_list) if index < vector_size])
    return words_padding


def shuffle_data(data_filename, shuffle_data_filename):
  print 'Shuffle data in ' + data_filename
  line_list = []
  with open(data_filename, mode='r') as data_file:
    for line in data_file:
      line_list.append(line.strip())
  random.shuffle(line_list)
  with open(shuffle_data_filename, mode='w') as shuffle_data_file:
    for line in line_list:
      shuffle_data_file.write(line + '\n')


def format_data(data_list):
  new_data_list = []
  for line in data_list:
    new_word_label_list = []
    id, word_list, label_list = split(line.strip())
    if id and word_list and label_list:
      for word, label in zip(word_list, label_list):
        label = label.upper()
        if label == 'MISC' or label == 'BRAND' or label == 'EBRAND' or label == 'LOCATION' or label == 'ENTITY':
          label = 'O'
        new_word_label_list.append(word + '/' + label)
      if new_word_label_list:
        new_data_list.append(id + '\t' + '||'.join(new_word_label_list))
  return new_data_list


def separate_id_word_label(data_list):
  id_list = []
  word_label_list = []
  for data in data_list:
    if not isinstance(data, unicode):
      data = data.decode()
    id_index = data.find('\t')
    id = data[:id_index].strip()
    word_label = data[id_index:].strip()
    if id and word_label:
      id_list.append(id)
      word_label_list.append(word_label)
  return id_list, word_label_list


def extract_guillemet(data_list):
  new_data_list = []
  for line in data_list:
    sentence = line.strip()
    if '《' not in sentence and '》' not in sentence:
      continue
    _, word_list, label_list = split(sentence)
    if word_list and label_list:
      new_word_label_list = []
      for word, label in zip(word_list, label_list):
        if word == '《' or word == '》':
          continue
        new_word_label_list.append(word + '/' + label)
      if new_word_label_list:
         new_data_list.append('||'.join(new_word_label_list))
  return new_data_list


def prepare_datasets(raw_data_filename, train_percent, val_percent, datasets_path):
  print 'Prepare datasets......'
  data_list = []
  with open(raw_data_filename, mode='r') as raw_data_file:
    for line in raw_data_file:
      data_list.append(line.strip())
  data_list = format_data(data_list)
  random.shuffle(data_list)

  data_size = len(data_list)
  train_validation_index = int(data_size * train_percent)
  validation_test_index = int(data_size * (train_percent + val_percent))

  train_list = data_list[:train_validation_index]
  validation_list = data_list[train_validation_index:validation_test_index]
  test_list = data_list[validation_test_index:]

  train_id_list, train_data_list = separate_id_word_label(train_list)
  validation_id_list, validation_data_list = separate_id_word_label(validation_list)
  test_id_list, test_data_list = separate_id_word_label(test_list)

  list_to_file(train_id_list, os.path.join(datasets_path, 'train_id.txt'))
  list_to_file(validation_id_list, os.path.join(datasets_path, 'validation_id.txt'))
  list_to_file(test_id_list, os.path.join(datasets_path, 'test_id.txt'))

  without_guillemet_train_list = extract_guillemet(train_data_list)
  train_data_list.extend(without_guillemet_train_list)
  random.shuffle(train_data_list)

  list_to_file(train_data_list, os.path.join(datasets_path, 'train.txt'))
  list_to_file(validation_data_list, os.path.join(datasets_path, 'validation.txt'))
  list_to_file(test_data_list, os.path.join(datasets_path, 'test.txt'))


def cut_guillemets(data_filename, new_data_filename):
  with open(data_filename, mode='r') as data_file:
    with open(new_data_filename, mode='w') as new_data_file:
      count = 1
      for line in data_file:
        print 'count : %d' % (count)
        count += 1
        _, word_list, label_list = split(line.strip())
        word_label_list = []
        for word, label in zip(word_list, label_list):
          if word != '《' and word != '》':
            word_label_list.append(word + '/' + label)
        word_label = '||'.join(word_label_list)
        print word_label
        new_data_file.write(word_label + '\n')


# oral_sentence.txt
def add_data(data_filename, data_list):
  with open(data_filename, mode='r') as data_file:
    for line in data_file:
      line = line.strip()
      if line and line not in data_list:
        data_list.append(line)
  return data_list


# oral_sentence.txt
def sort_data(data_filename, new_data_filename):
  data_list = []
  data_list = add_data(data_filename, data_list)
  data_list.sort(key=lambda x: len(x))
  with open(new_data_filename, mode='w') as new_data_file:
    for data in data_list:
      new_data_file.write(data + '\n')


# oral_sentence.txt
def to_word_label(data_filename, new_data_filename):
  with open(data_filename, mode='r') as data_file:
    with open(new_data_filename, mode='w') as new_data_file:
      for line in data_file:
        word_list = line.strip().split()
        word_label_list = []
        for word in word_list:
          label = 'o'
          word_label_list.append(word + '/' + label)
        if word_label_list:
          word_label = '||'.join(word_label_list)
          new_data_file.write(word_label + '\n')


# oral_sentence.txt
def replace_data(pastime_list, data_list, pastime_flag):
  new_data_list = []
  for pastime in pastime_list:
    print pastime
    all_count = 0
    count = 0
    while count < 1 and all_count < len(data_list):
      all_count += 1
      sentence = random.choice(data_list)
      if pastime_flag in sentence:
        count += 1
        _, word_list, label_list = split(sentence.strip())
        word_label_list = []
        for word, label in zip(word_list, label_list):
          if label == pastime_flag:
            word = pastime
          word_label_list.append(word + '/' + label)
        new_data_list.append('||'.join(word_label_list))
  return new_data_list


# oral_sentence.txt
def create_data(celebrity_filename, pastime_filename, data1_filename, data2_filename, new_data_filename):
  celebrity_list = []
  with open(celebrity_filename, mode='r') as celebrity_file:
    for line in celebrity_file:
      line = line.strip()
      if line:
        celebrity_list.append(line)
  pastime_list = []
  with open(pastime_filename, mode='r') as pastime_file:
    for line in pastime_file:
      line = line.strip()
      if line:
        pastime_list.append(line)
  data1_list = []
  with open(data1_filename, mode='r') as data1_file:
    for line in data1_file:
      line = line.strip()
      if line:
        data1_list.append(line)
  data2_list = []
  with open(data2_filename, mode='r') as data2_file:
    for line in data2_file:
      line = line.strip()
      if line:
        data2_list.append(line)
  new_data1_pastime_list = replace_data(pastime_list, data1_list, 'PASTIME')
  new_data2_celebrity_list = replace_data(celebrity_list, data2_list, 'CELEBRITY')
  new_data2_pastime_list = replace_data(pastime_list, data2_list, 'PASTIME')
  new_data_list = data1_list
  new_data_list.extend(data2_list)
  new_data_list.extend(new_data1_pastime_list)
  new_data_list.extend(new_data2_celebrity_list)
  new_data_list.extend(new_data2_pastime_list)
  random.shuffle(new_data_list)
  list_to_file(new_data_list, new_data_filename)


def main():
  raw_data_path = FLAGS.raw_data_path
  datasets_path = FLAGS.datasets_path
  vocab_path = FLAGS.vocab_path
  vocab_size = FLAGS.vocab_size
  train_percent = FLAGS.train_percent
  val_percent = FLAGS.val_percent

  prepare_datasets(os.path.join(raw_data_path, 'data.txt'), train_percent, val_percent, datasets_path)

  # shuffle_data(os.path.join(train_path, 'train.txt'), os.path.join(train_path, 'shuffle_train.txt'))

  words_vocab, labels_vocab, labels_words_vocab = create_vocabulary(os.path.join(datasets_path, 'train.txt'),
                                                                    vocab_path, vocab_size)


if __name__ == '__main__':
  main()
