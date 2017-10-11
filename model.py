#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import tensorflow.contrib.crf as crf
import numpy as np


def inference(inputs, batch_size, num_steps, vocab_size, embedding_size, hidden_size, keep_prob, num_layers,
              num_classes, is_training, use_lstm=True, use_bidirectional_rnn=True):
  with tf.device('/cpu:0'):
    embedding = tf.get_variable('embedding', [vocab_size, embedding_size], initializer=tf.random_uniform_initializer(),
                                dtype=tf.float32)
    inputs_embedding = tf.nn.embedding_lookup(embedding, inputs)
  if is_training and keep_prob < 1:
    inputs_embedding = tf.nn.dropout(inputs_embedding, keep_prob)
  inputs_embedding = tf.unstack(inputs_embedding, axis=1)

  initializer = tf.random_uniform_initializer(-0.1, 0.1)

  if use_lstm:
    forward_single_cell = rnn.LSTMCell(num_units=hidden_size, initializer=initializer, forget_bias=1.0)
  else:
    forward_single_cell = rnn.GRUCell(num_units=hidden_size)
  if is_training and keep_prob < 1.0:
    forward_single_cell = rnn.DropoutWrapper(forward_single_cell, output_keep_prob=keep_prob)
  forward_rnn_cell = rnn.MultiRNNCell([forward_single_cell for _ in range(num_layers)])

  if use_lstm:
    backward_single_cell = rnn.LSTMCell(num_units=hidden_size, initializer=initializer, forget_bias=1.0)
  else:
    backward_single_cell = rnn.GRUCell(num_units=hidden_size)
  if is_training and keep_prob < 1.0:
    backward_single_cell = rnn.DropoutWrapper(backward_single_cell, output_keep_prob=keep_prob)
  backward_rnn_cell = rnn.MultiRNNCell([backward_single_cell for _ in range(num_layers)])

  bi_flag = 1
  if use_bidirectional_rnn:
    bi_flag = 2
    outputs, forward_final_state, backward_final_state = rnn.static_bidirectional_rnn(forward_rnn_cell, backward_rnn_cell,
                                                                                      inputs_embedding, dtype=tf.float32,
                                                                                      sequence_length=[num_steps] * batch_size)
    final_state = (tf.concat([forward_final_state[0], backward_final_state[0]], axis=2),
                   tf.concat([forward_final_state[1], backward_final_state[1]], axis=2))
  else:
    outputs, final_state = rnn.static_rnn(forward_rnn_cell, inputs_embedding, dtype=tf.float32,
                                          sequence_length=[num_steps] * batch_size)

  output = tf.reshape(tf.concat(outputs, axis=1), shape=[-1, bi_flag * hidden_size])

  weights = tf.get_variable('weights', [bi_flag * hidden_size, num_classes], dtype=tf.float32)
  biases = tf.get_variable('biases', [num_classes], dtype=tf.float32)
  logits = tf.matmul(output, weights) + biases

  return logits, final_state


def loss(logits, labels):
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
  loss = tf.reduce_mean(cross_entropy, name='loss')
  return loss


def crf_loss(logits, labels, batch_size, num_steps, num_classes):
  logits = tf.reshape(logits, shape=[batch_size, num_steps, num_classes])
  labels = tf.cast(labels, tf.int32)
  sequence_lengths = np.full(batch_size, num_steps - 1, dtype=np.int32)
  sequence_lengths = tf.constant(sequence_lengths)
  log_likelihood, transition_params = crf.crf_log_likelihood(logits, labels, sequence_lengths)
  loss = tf.reduce_mean(-log_likelihood, name='loss')
  return loss


def accuracy(logits, labels):
  props = tf.nn.softmax(logits)
  prediction_labels = tf.arg_max(props, 1)
  correct_prediction = tf.equal(prediction_labels, labels)
  accuracy_value = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  return accuracy_value


def slice_seq(logits, labels, words_len, batch_size, num_steps):
  labels = tf.reshape(labels, shape=[-1])
  slice_indices = tf.constant([], dtype=tf.int64)
  for index in range(batch_size):
    sub_slice_indices = tf.range(words_len[index])
    sub_slice_indices = tf.add(tf.constant(index * num_steps, dtype=tf.int64), sub_slice_indices)
    slice_indices = tf.concat([slice_indices, sub_slice_indices], axis=0)
  slice_logits = tf.gather(logits, slice_indices)
  slice_labels = tf.gather(labels, slice_indices)
  return slice_logits, slice_labels