# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Copyright 2018 The Google AI Language Team Authors.
BASED ON Google_BERT.
@Author: Wei Yi
"""""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import collections
import json
import re
import pickle
import os

import modeling
import tokenization
import tensorflow as tf


flags = tf.flags

FLAGS = flags.FLAGS


flags.DEFINE_string("input_file", None, "")

flags.DEFINE_string("output_file", None, "")

flags.DEFINE_string("layers", "-1,-2,-3,-4", "")

flags.DEFINE_integer("max_prediction", 1, "")

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_integer(
    "max_seq_length", 64,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer("batch_size", 32, "Batch size for predictions.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_string("master", None,
                    "If using a TPU, the address of the master.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_bool(
    "use_one_hot_embeddings", False,
    "If True, tf.one_hot will be used for embedding lookups, otherwise "
    "tf.nn.embedding_lookup will be used. On TPUs, this should be True "
    "since it is much faster.")


class InputExample(object):

  def __init__(self, unique_id, text_a, text_b):
    self.unique_id = unique_id
    self.text_a = text_a
    self.text_b = text_b


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
    self.unique_id = unique_id
    self.tokens = tokens
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.input_type_ids = input_type_ids


def input_fn_builder(features, seq_length):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  all_unique_ids = []
  all_input_ids = []
  all_input_mask = []
  all_input_type_ids = []

  for feature in features:
    all_unique_ids.append(feature.unique_id)
    all_input_ids.append(feature.input_ids)
    all_input_mask.append(feature.input_mask)
    all_input_type_ids.append(feature.input_type_ids)

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    num_examples = len(features)

    d = tf.data.Dataset.from_tensor_slices({
        "unique_ids":
            tf.constant(all_unique_ids, shape=[num_examples], dtype=tf.int64),
        "input_ids":
            tf.constant(
                all_input_ids, shape=[num_examples, seq_length],
                dtype=tf.int32),
        "input_mask":
            tf.constant(
                all_input_mask,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "input_type_ids":
            tf.constant(
                all_input_type_ids,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
    })

    d = d.batch(batch_size=batch_size, drop_remainder=False)
    return d

  return input_fn


def model_fn_builder(bert_config, init_checkpoint, layer_indexes, use_tpu,
                     use_one_hot_embeddings):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    unique_ids = features["unique_ids"]
    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    input_type_ids = features["input_type_ids"]

    model = modeling.BertModel(
        config=bert_config,
        is_training=False,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=input_type_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    output_layer = model.get_sequence_output()

    hidden_size = output_layer.shape[-1].value

    output_weight = tf.get_variable(
        "output_weights", [5, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02)
    )
    output_bias = tf.get_variable(
        "output_bias", [5], initializer=tf.zeros_initializer()
    )

    output_layer = tf.reshape(output_layer, [-1, hidden_size])
    logits = tf.matmul(output_layer, output_weight, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    logits = tf.reshape(logits, [-1, FLAGS.max_seq_length, 5])
    probabilities = tf.nn.softmax(logits, axis=-1)
    prediction = tf.argmax(probabilities, axis=-1)

    if mode != tf.estimator.ModeKeys.PREDICT:
      raise ValueError("Only PREDICT modes are supported: %s" % (mode))

    tvars = tf.trainable_variables()
    scaffold_fn = None
    (assignment_map,
     initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(
         tvars, init_checkpoint)
    if use_tpu:

      def tpu_scaffold():
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
        return tf.train.Scaffold()

      scaffold_fn = tpu_scaffold
    else:
      tf.train.init_from_checkpoint(init_checkpoint, assignment_map)


    tf.logging.info("**** INIT_FROM_CKPT ****")
    """
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)
    """
    predictions = {
        "unique_id": unique_ids,
        "prediction": prediction
    }


    output_spec = tf.contrib.tpu.TPUEstimatorSpec(
        mode=mode, predictions=predictions, scaffold_fn=scaffold_fn)
    return output_spec

  return model_fn


def convert_examples_to_features(examples, seq_length, tokenizer):
  """Loads a data file into a list of `InputBatch`s."""

  features = []
  for (ex_index, example) in enumerate(examples):
    tokens_a = tokenizer.tokenize(example.text_a)

    tokens_b = None
    if example.text_b:
      tokens_b = tokenizer.tokenize(example.text_b)

    if tokens_b:
      # Modifies `tokens_a` and `tokens_b` in place so that the total
      # length is less than the specified length.
      # Account for [CLS], [SEP], [SEP] with "- 3"
      _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
    else:
      # Account for [CLS] and [SEP] with "- 2"
      if len(tokens_a) > seq_length - 2:
        tokens_a = tokens_a[0:(seq_length - 2)]


    tokens = []
    input_type_ids = []
    tokens.append("[CLS]")
    input_type_ids.append(0)
    for token in tokens_a:
      tokens.append(token)
      input_type_ids.append(0)
    tokens.append("[SEP]")
    input_type_ids.append(0)

    if tokens_b:
      for token in tokens_b:
        tokens.append(token)
        input_type_ids.append(1)
      tokens.append("[SEP]")
      input_type_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < seq_length:
      input_ids.append(0)
      input_mask.append(0)
      input_type_ids.append(0)

    assert len(input_ids) == seq_length
    assert len(input_mask) == seq_length
    assert len(input_type_ids) == seq_length

    if ex_index < 5:
      tf.logging.info("*** Example ***")
      tf.logging.info("unique_id: %s" % (example.unique_id))
      tf.logging.info("tokens: %s" % " ".join(
          [tokenization.printable_text(x) for x in tokens]))
    features.append(
        InputFeatures(
            unique_id=example.unique_id,
            tokens=tokens,
            input_ids=input_ids,
            input_mask=input_mask,
            input_type_ids=input_type_ids))
  return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  """Truncates a sequence pair in place to the maximum length."""

  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()


def read_examples(input_file):
  """Read a list of `InputExample`s from an input file."""
  examples = []
  unique_id = -1
  with open(input_file, "r", encoding="utf-8") as reader:
    lines = reader.read().split('\n')
    for line in lines:
      unique_id += 1
      if line == '':
        continue
      line = tokenization.convert_to_unicode(line)
      if not line:
        break
      line = line.strip()
      text_a = None
      text_b = None
      m = None
      if m is None:
        text_a = line[:min(FLAGS.max_seq_length-2, len(line))]
      else:
        text_a = m.group(1)
        text_b = m.group(2)
      examples.append(
          InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b))
  return examples


def main(_):
    while True:
      cnt = 0
      with open(FLAGS.input_file, "r", encoding="utf-8") as reader:
        lines = reader.read().split('\n')
        for line in lines:
          if line == '':
            cnt += 1
      if len(lines) == cnt:
          break
      tf.logging.set_verbosity(tf.logging.INFO)

      layer_indexes = [int(x) for x in FLAGS.layers.split(",")]

      bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

      tokenizer = tokenization.FullTokenizer(
          vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

      is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
      run_config = tf.contrib.tpu.RunConfig(
          master=FLAGS.master,
          tpu_config=tf.contrib.tpu.TPUConfig(
              num_shards=FLAGS.num_tpu_cores,
              per_host_input_for_training=is_per_host))

      examples = read_examples(FLAGS.input_file)

      features = convert_examples_to_features(
          examples=examples, seq_length=FLAGS.max_seq_length, tokenizer=tokenizer)

      unique_id_to_feature = {}
      for feature in features:
        unique_id_to_feature[feature.unique_id] = feature

      model_fn = model_fn_builder(
          bert_config=bert_config,
          init_checkpoint=FLAGS.init_checkpoint,
          layer_indexes=layer_indexes,
          use_tpu=FLAGS.use_tpu,
          use_one_hot_embeddings=FLAGS.use_one_hot_embeddings)

      # If TPU is not available, this will fall back to normal Estimator on CPU
      # or GPU.
      estimator = tf.contrib.tpu.TPUEstimator(
          use_tpu=FLAGS.use_tpu,
          model_fn=model_fn,
          config=run_config,
          predict_batch_size=FLAGS.batch_size)

      input_fn = input_fn_builder(
          features=features, seq_length=FLAGS.max_seq_length)

      with open(FLAGS.input_file, "r", encoding="utf-8") as reader:
        lines = reader.read().split('\n')
      for result in estimator.predict(input_fn, yield_single_examples=True):
        unique_id = int(result["unique_id"])
        pred = result["prediction"]
        tot_bos = 0
        st = 0
        tf.logging.info(pred)
        #tf.logging.info((type(pred)))
        for id in range(len(pred)):
          if pred[id] == 1:
            tot_bos += 1
          if pred[id] == 3 and id == 0:
            st = 1
        max_bos = min(tot_bos, FLAGS.max_prediction)
        bos = 0
        res = list()
        word_pos = 0
        if max_bos > 1:
          temp = list()
          for i in range(st, len(lines[unique_id])+1):
            if pred[i] == 1:
              if len(temp) != 0:
                res.append(temp)
                bos += 1
              temp = list()
              temp.append(lines[unique_id][word_pos])
            else:
              temp.append(lines[unique_id][word_pos])
            if bos == max_bos:
              break
            word_pos += 1
          if bos < max_bos:
            res.append(temp)
        else:
          res.append([lines[unique_id]])
          word_pos = len(lines[unique_id])
        save_name = FLAGS.output_file + "para" + str(unique_id) + ".txt"
        if not os.path.exists(save_name):
          with open(save_name, 'w', encoding="utf-8") as sfile:
            pass
        with open(save_name, 'a', encoding="utf-8") as sfile:
          for line in res:
            sfile.write("".join(line) + '\n')
            tf.logging.info(line)
        lines[unique_id] = lines[unique_id][word_pos:]
      with open(FLAGS.input_file, "w", encoding="utf-8") as file:
        for line in lines:
          file.write(line + '\n')


if __name__ == "__main__":
  flags.mark_flag_as_required("input_file")
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("init_checkpoint")
  flags.mark_flag_as_required("output_file")
  tf.app.run()
