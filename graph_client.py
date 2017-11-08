#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import tensorflow as tf

reload(sys)
sys.setdefaultencoding('utf8')

inputs_data_list = ["地点 ： 广州市 南丰 国际 会展 中心"]
with open('ner_graph.pb', 'r') as graph_file:
  graph_def = tf.GraphDef()
  graph_def.ParseFromString(graph_file.read())
  graph_out = tf.import_graph_def(graph_def, input_map={'input_sentences': inputs_data_list},
                                  return_elements=['init_all_tables', 'predict_labels'], name='graph_out')

tables_init_op = tf.tables_initializer()
with tf.Session() as sess:
  sess.run(tables_init_op)
  result = sess.run(graph_out)
  print result