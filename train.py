#!/usr/bin/env python
# -*- coding: utf-8 -*-

import datetime
import os
import tensorflow as tf
import tfrecords_utils
import model

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('checkpoint_path', 'checkpoint/', 'checkpoint directory')
tf.app.flags.DEFINE_string('tensorboard_path', 'tensorboard/', 'tensorboard output')
tf.app.flags.DEFINE_float('learning_rate', 0.01, 'learning rate')
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.9, 'learning rate decay factor')
tf.app.flags.DEFINE_integer('decay_steps', 10000, 'decay steps')
tf.app.flags.DEFINE_integer('num_layers', 2, 'lstm layers')
tf.app.flags.DEFINE_integer('embedding_size', 100, 'word embedding size')
tf.app.flags.DEFINE_integer('hidden_size', 100, 'lstm hidden size')
tf.app.flags.DEFINE_float('keep_prob', 0.5, 'keep prob')
tf.app.flags.DEFINE_float('clip_norm', 5.0, 'clipping ratio')
tf.app.flags.DEFINE_integer('num_classes', 21, 'named entity classes')


def main(_):
  learning_rate = FLAGS.learning_rate
  learning_rate_decay_factor = FLAGS.learning_rate_decay_factor
  num_layers = FLAGS.num_layers
  num_steps = FLAGS.num_steps
  embedding_size = FLAGS.embedding_size
  hidden_size = FLAGS.hidden_size
  keep_prob = FLAGS.keep_prob
  batch_size = FLAGS.batch_size
  vocab_size = FLAGS.vocab_size
  clip_norm = FLAGS.clip_norm
  num_classes = FLAGS.num_classes
  checkpoint_path = FLAGS.checkpoint_path
  tensorboard_path = FLAGS.tensorboard_path

  tfrecords_path = FLAGS.tfrecords_path
  train_tfrecords_filename = os.path.join(tfrecords_path, 'train.tfrecords')
  validate_tfrecords_filename = os.path.join(tfrecords_path, 'validate.tfrecords')
  train_data = tfrecords_utils.read_and_decode(train_tfrecords_filename)
  train_batch_features, train_batch_labels, train_words_len_batch = train_data
  validate_data = tfrecords_utils.read_and_decode(validate_tfrecords_filename)
  validate_batch_features, validate_batch_labels, validate_words_len_batch = validate_data

  with tf.device('/cpu:0'):
    global_step = tf.Variable(0, name='global_step', trainable=False)
  # Decay the learning rate exponentially based on the number of steps.
  decay_steps = FLAGS.decay_steps
  lr = tf.train.exponential_decay(learning_rate, global_step, decay_steps, learning_rate_decay_factor, staircase=True)
  optimizer = tf.train.RMSPropOptimizer(lr)

  with tf.variable_scope('model'):
    logits, final_state = model.inference(train_batch_features, batch_size, num_steps, vocab_size, embedding_size,
                                          hidden_size, keep_prob, num_layers, num_classes, is_training=True)
  train_batch_labels = tf.to_int64(train_batch_labels)

  # Loss of cross entropy between logits and labels
  # slice_logits, slice_train_batch_labels = model.slice_seq(logits, train_batch_labels, train_words_len_batch,
  #                                                          batch_size, num_steps)
  # loss = model.loss(slice_logits, slice_train_batch_labels)

  # Loss of crf
  loss = model.crf_loss(logits, train_batch_labels, batch_size, num_steps, num_classes)

  with tf.variable_scope('model', reuse=True):
    accuracy_logits, final_state_valid = model.inference(validate_batch_features, batch_size, num_steps, vocab_size,
                                                         embedding_size, hidden_size, keep_prob, num_layers,
                                                         num_classes, is_training=False)
  validate_batch_labels = tf.to_int64(validate_batch_labels)
  slice_accuracy_logits, slice_validate_batch_labels = model.slice_seq(accuracy_logits, validate_batch_labels,
                                                           validate_words_len_batch, batch_size, num_steps)
  accuracy = model.accuracy(slice_accuracy_logits, slice_validate_batch_labels)

  # summary
  tf.summary.scalar('loss', loss)
  tf.summary.scalar('accuracy', accuracy)
  tf.summary.scalar('lr', lr)
  if not os.path.exists(tensorboard_path):
    os.makedirs(tensorboard_path)

  # compute and update gradient
  # train_op = optimizer.minimize(loss, global_step=global_step)

  # computer, clip and update gradient
  gradients, variables = zip(*optimizer.compute_gradients(loss))
  clip_gradients, _ = tf.clip_by_global_norm(gradients, clip_norm)
  train_op = optimizer.apply_gradients(zip(clip_gradients, variables), global_step=global_step)

  init_op = tf.global_variables_initializer()

  saver = tf.train.Saver(max_to_keep=None)
  if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)
  checkpoint_filename = os.path.join(checkpoint_path, 'model.ckpt')

  with tf.Session() as sess:
    summary_op = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_path, sess.graph)
    sess.run(init_op)

    ckpt = tf.train.get_checkpoint_state(checkpoint_path)
    if ckpt and ckpt.model_checkpoint_path:
      print('Continue training from the model {}'.format(ckpt.model_checkpoint_path))
      saver.restore(sess, ckpt.model_checkpoint_path)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    start_time = datetime.datetime.now()
    try:
      while not coord.should_stop():
        _, loss_value, step = sess.run([train_op, loss, global_step])
        if step % 100 == 0:
          accuracy_value, summary_value, lr_value = sess.run([accuracy, summary_op, lr])
          end_time = datetime.datetime.now()
          print('[{}] Step: {}, loss: {}, accuracy: {}, lr: {}'.format(end_time - start_time, step, loss_value,
                                                                       accuracy_value, lr_value))
          if step % 1000 == 0:
            writer.add_summary(summary_value, step)
            saver.save(sess, checkpoint_filename, global_step=step)
            print 'save model to ' + checkpoint_filename + '-' + str(step)
          start_time = end_time
    except tf.errors.OutOfRangeError:
      print('Done training after reading all data')
    finally:
      coord.request_stop()
    coord.join(threads)


if __name__ == '__main__':
  tf.app.run()