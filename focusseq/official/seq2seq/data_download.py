# -*- coding: utf-8 -*-
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Download and preprocess WMT17 ende training and evaluation datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import tarfile
import six
from six.moves import urllib
from absl import app as absl_app
from absl import flags
import tensorflow as tf
from builtins import zip

from official.seq2seq.utils import tokenizer
from official.utils.flags import core as flags_core

# Use pre-defined minimum count to generate subtoken vocabulary.
_TRAIN_DATA_MIN_COUNT = 6


# Data sources for training/evaluating the transformer translation model.
# If any of the training sources are changed, then either:
#   1) use the flag `--search` to find the best min count or
#   2) update the _TRAIN_DATA_MIN_COUNT constant.
# min_count is the minimum number of times a token must appear in the data
# before it is added to the vocabulary. "Best min count" refers to the value
# that generates a vocabulary set that is closest in size to _TARGET_VOCAB_SIZE.
# Vocabulary constants
def find_file(path, filename, max_depth=5):
    """Returns full filepath if the file is in path or a subdirectory."""
    for root, dirs, files in os.walk(path):
        if filename in files:
            return os.path.join(root, filename)

        # Don't search past max_depth
        depth = root[len(path) + 1:].count(os.sep)
        if depth > max_depth:
            del dirs[:]  # Clear dirs
    return None


def get_raw_files(raw_dir, data_source):
    """Return raw files from source. Downloads/extracts if needed.
  
    Args:
      raw_dir: string directory to store raw files
      data_source: dictionary with
        {"url": url of compressed dataset containing input and target files
         "input": file with data in input language
         "target": file with data in target language}
  
    Returns:
      dictionary with
        {"inputs": list of files containing data in input language
         "targets": list of files containing corresponding data in target language
        }
    """
    raw_files = {}
    input_file = tf.gfile.Exists(os.path.join(raw_dir, data_source["input"]))
    target_file = tf.gfile.Exists(os.path.join(raw_dir, data_source["target"]))

    if input_file and target_file:
        raw_files["input"] = os.path.join(raw_dir, data_source["input"])
        raw_files["target"] = os.path.join(raw_dir, data_source["target"])
        return raw_files
    raise OSError("Raw Data %s does not exist." %data_source)


def txt_line_iterator(path):
    """Iterate through lines of file."""
    with tf.gfile.Open(path) as f:
        for line in f:
            yield line.strip()


def write_file(writer, filename):
    """Write all of lines from file using the writer."""
    for line in txt_line_iterator(filename):
        writer.write(line)
        writer.write("\n")
def serialize_example(example):     
    return example.SerializeToString()

###############################################################################
# Data preprocessing
###############################################################################
def encode_and_save_files(
        subtokenizer_source, subtokenizer_target, data_dir, raw_files, tag, total_shards):
    """Save data from files as encoded Examples in TFrecord format.
  
    Args:
      subtokenizer: Subtokenizer object that will be used to encode the strings.
      data_dir: The directory in which to write the examples
      raw_files: A tuple of (input, target) data files. Each line in the input and
        the corresponding line in target file will be saved in a tf.Example.
      tag: String that will be added onto the file names.
      total_shards: Number of files to divide the data into.
  
    Returns:
      List of all files produced.
    """
    # Create a file for each shard.
    filepaths = [shard_filename(data_dir, tag, n + 1, total_shards)
                 for n in range(total_shards)]

    if all_exist(filepaths):
        tf.logging.info("Files with tag %s already exist." % tag)
        return filepaths

    tf.logging.info("Saving files with tag %s." % tag)
    input_file = raw_files[0]
    target_file = raw_files[1]

    # Write examples to each shard in round robin order.
    tmp_filepaths = [fname + ".incomplete" for fname in filepaths]
    writers = [tf.python_io.TFRecordWriter(fname) for fname in tmp_filepaths]
    counter, shard = 0, 0
    tf.logging.info("Start zip examples from input and target file")

    for counter, (input_line, target_line) in enumerate(zip(
            txt_line_iterator(input_file), txt_line_iterator(target_file))):
        if counter > 0 and counter % 100000 == 0:
            tf.logging.info("\tSaving case %d." % counter)
        example = dict_to_example(
            {"inputs": subtokenizer_source.encode(input_line, add_eos=True),
             "targets": subtokenizer_target.encode(target_line, add_eos=True)})
        writers[shard].write(example.SerializeToString())
        shard = (shard + 1) % total_shards
    for writer in writers:
        writer.close()

    for tmp_name, final_name in zip(tmp_filepaths, filepaths):
        tf.gfile.Rename(tmp_name, final_name)

    tf.logging.info("Saved %d Examples", counter + 1)
    return filepaths
 

def shard_filename(path, tag, shard_num, total_shards):
    """Create filename for data shard."""
    return os.path.join(
        path, "%s-%s-%.5d-of-%.5d" % (FLAGS.problem, tag, shard_num, total_shards))


def shuffle_records(fname):
    """Shuffle records in a single file."""
    # tf.logging.info("Shuffling records in file %s" % fname)

    # Rename file prior to shuffling
    tmp_fname = fname + ".unshuffled"
    fname_origin = fname
    tf.gfile.Rename(fname, tmp_fname)
    reader = tf.python_io.tf_record_iterator(tmp_fname)
    records = []
    for record in reader:
        records.append(record)
        if len(records) % 100000 == 0:
            tf.logging.info("\tRead: %d", len(records))
    random.shuffle(records)

    # Write shuffled records to original file name
    with tf.python_io.TFRecordWriter(fname_origin) as w:
        for count, record in enumerate(records):
            w.write(record)
            if count > 0 and count % 100000 == 0:
                tf.logging.info("\tWriting record: %d" % count)
    tf.gfile.Remove(tmp_fname)


def dict_to_example(dictionary):
    """Converts a dictionary of string->int to a tf.Example."""
    features = {}
    for k, v in six.iteritems(dictionary):
        features[k] = tf.train.Feature(int64_list=tf.train.Int64List(value=v))
    return tf.train.Example(features=tf.train.Features(feature=features))


def all_exist(filepaths):
    """Returns true if all files in the list exist."""
    for fname in filepaths:
        if not tf.gfile.Exists(fname):
            return False
    return True


def make_dir(path):
    if not tf.gfile.Exists(path):
        tf.logging.info("Creating directory %s" % path)
        tf.gfile.MakeDirs(path)


def main(unused_argv):
    """Obtain training and evaluation data for the Transformer model."""
    make_dir(FLAGS.raw_dir)
    make_dir(FLAGS.data_dir)


    tf.logging.info("Step 1/3: Verify raw_data exists")
    ## verify train_data,valid_data,vocab_data exists in raw_dir.##
    _TRAIN_DATA_SOURCES = {"input": _TRAIN_TAG + '.' + str(FLAGS.search) + "." + FLAGS.fro,
                           "target": _TRAIN_TAG + '.' + str(FLAGS.search) + "." + FLAGS.to}
    _EVAL_DATA_SOURCES = {"input": _EVAL_TAG + '.' + str(FLAGS.search) + "." + FLAGS.fro,
                          "target": _EVAL_TAG + '.' + str(FLAGS.search) + "." + FLAGS.to}
    _VOCAB_DATA_SOURCES = {"input": 'vocab.'+str(FLAGS.search)+"."+FLAGS.fro,
                           "target":'vocab.'+str(FLAGS.search)+"."+FLAGS.to,}

    train_files = get_raw_files(FLAGS.raw_dir, _TRAIN_DATA_SOURCES)
    eval_files = get_raw_files(FLAGS.raw_dir, _EVAL_DATA_SOURCES)
    vocab_files = get_raw_files(FLAGS.raw_dir, _VOCAB_DATA_SOURCES )

    
    tf.logging.info("Step 2/3: Building vocabulary and creating subtokenizer ")
    ## make sure vocab_data in data_dir.if not ,copy vocab_data in rwa_dir to  data_dir. ##

    vocab_file_source = os.path.join(FLAGS.data_dir, 'vocab' + '.' + str(FLAGS.search) + '.' + FLAGS.fro)
    vocab_file_target = os.path.join(FLAGS.data_dir, 'vocab' + '.' + str(FLAGS.search) + '.' + FLAGS.to)

    if not tf.gfile.Exists(vocab_file_source):
        fp = open(vocab_files["input"], 'r')
        vocab_source = fp.readlines()
        with open(vocab_file_source, 'w')as fp:
            for word in vocab_source:
                fp.write(word)
            fp.close()
    if not tf.gfile.Exists(vocab_file_target):
        fp = open(vocab_files["target"], 'r')
        vocab_target = fp.readlines()
        with open(vocab_file_target, 'w')as fp:
            for word in vocab_target:
                fp.write(word)
            fp.close()
    ## creating subtokenizer ## 
    subtokenizer_source = tokenizer.Subtokenizer.init_from_files(vocab_file_source, FLAGS.search)
    subtokenizer_target = tokenizer.Subtokenizer.init_from_files(vocab_file_target, FLAGS.search)

    tf.logging.info("Step 3/3: Preprocessing and saving data")
    ## Tokenize and save data as Examples in the TFRecord format.##
    train_files_flat_source = train_files["input"]
    train_files_flat_target = train_files["target"]
    ##save train_data ##
    train_tfrecord_files = encode_and_save_files(subtokenizer_source, subtokenizer_target, FLAGS.data_dir,
                                                 [train_files_flat_source, train_files_flat_target], _TRAIN_TAG,
                                                 _TRAIN_SHARDS)
    tf.logging.info("Shuffling records")
    ## shuffle train_data ##
    for fname in train_tfrecord_files:
        shuffle_records(fname)
    ## save valid_data## 
    encode_and_save_files(subtokenizer_source, subtokenizer_target, FLAGS.data_dir,
                          [eval_files["input"], eval_files["target"]], _EVAL_TAG,
                          _EVAL_SHARDS)


def define_data_download_flags():
    """Add flags specifying data download arguments."""
    flags.DEFINE_string(
        name="data_dir", short_name="dd", default="/tmp/translate_ende",
        help=flags_core.help_wrap(
            "Directory for where the raw dataset is saved."))
    flags.DEFINE_string(
        name="raw_dir", short_name="rd", default="/tmp/translate_ende_raw",
        help=flags_core.help_wrap(
            "Path where the raw data will be downloaded and extracted."))
    flags.DEFINE_integer(
        name="search", default=0,
        help=flags_core.help_wrap(
            "Must set,if we use our own datas, use binary search to find the vocabulary set with size"
            "closest to the target size ."))
    flags.DEFINE_string(
        name='problem', short_name='pp', default="translator", help=flags_core.help_wrap("problem to translate."))
    flags.DEFINE_string(name='fro', default="zh", help=flags_core.help_wrap("language from."))

    flags.DEFINE_string(name='to', default="en", help=flags_core.help_wrap("language to"))


if __name__ == "__main__":
    _TRAIN_TAG = "train"
    _EVAL_TAG = "valid"
    _TRAIN_SHARDS = 100
    _EVAL_SHARDS = 1

    tf.logging.set_verbosity(tf.logging.INFO)

    define_data_download_flags()
    FLAGS = flags.FLAGS

    absl_app.run(main)
