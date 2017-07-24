# Copyright 2015 Google Inc. All Rights Reserved.
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

"""Routine for decoding the SVHN binary file format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

# Process images of this size. Note that this differs from the original SVHN
# image size of 32 x 32. If one alters this number, then the entire model
# architecture will change and any model would need to be retrained.
IMAGE_SIZE = 64
IMAGE_DEPTH = 3

# Global constants describing the SVHN data set.
NUM_CLASSES = 7
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('num_examples_per_epoch_for_train', 225988,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_integer('num_examples_per_epoch_for_test', 13068,
                            """Number of examples to test.""")
tf.app.flags.DEFINE_integer('num_examples_per_epoch_for_valid', 10000,
                            """Number of examples to run.""")


def read_svhn(filename_queue):
  """Reads and parses examples from SVHN data files.

  Recommendation: if you want N-way read parallelism, call this function
  N times.  This will give you N independent Readers reading different
  files & positions within those files, which will give better mixing of
  examples.

  Args:
    filename_queue: A queue of strings with the filenames to read from.

  Returns:
    An object representing a single example, with the following fields:
      height: number of rows in the result (32)
      width: number of columns in the result (32)
      depth: number of color channels in the result (3)
      key: a scalar string Tensor describing the filename & record number
        for this example.
      label: an int32 Tensor with the label in the range 0..9.
      uint8image: a [height, width, depth] uint8 Tensor with the image data
  """

  class SVHNRecord(object):
    pass
  result = SVHNRecord()

  # Dimensions of the images in the SVHN dataset.
  # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the
  # input format.
  len_bytes = 7
  box_num = 5  
  num_class = 11
  label_bytes = box_num * num_class
  pos_num = 4
  box_pos_bytes = box_num * pos_num
  result.height = 64
  result.width = 64
  result.depth = 3
  image_bytes = result.height * result.width * result.depth


  # Read a record, getting filenames from the filename_queue.  No
  # header or footer in the SVHN format, so we leave header_bytes
  # and footer_bytes at their default of 0.
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)    
  features = tf.parse_single_example(
      serialized_example,
      # Defaults are not specified since both keys are required.
      features={
          'len'  : tf.FixedLenFeature([], tf.string),
          'label': tf.FixedLenFeature([], tf.string),
          'box'  : tf.FixedLenFeature([], tf.string),              
          'image_raw': tf.FixedLenFeature([], tf.string),
      })

  length = tf.decode_raw(features['len'], tf.int8)    
  length.set_shape([len_bytes])
  result.length = tf.reshape(length, [len_bytes])

  label = tf.decode_raw(features['label'], tf.int8)
  label.set_shape([label_bytes])
  result.label = tf.reshape(label, [box_num, num_class])

  box = tf.decode_raw(features['box'], tf.float32)
  box.set_shape([box_pos_bytes])
  result.box = tf.reshape(box, [box_num, pos_num]) 

  image = tf.decode_raw(features['image_raw'], tf.uint8)
  image.set_shape([image_bytes])
  result.uint8image = tf.reshape(image, [result.height, result.width, result.depth])
 
  return result


def _generate_image_and_label_batch(image, length, digits, min_queue_examples,
                                    batch_size):
  """Construct a queued batch of images and labels.

  Args:
    image: 3-D Tensor of [height, width, 3] of type.float32.
    label: 1-D Tensor of type.int32
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.
    batch_size: Number of images per batch.

  Returns:
    images: Images. 4D tensor of [batch_size, height, width, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  # Create a queue that shuffles the examples, and then
  # read 'batch_size' images + labels from the example queue.
  num_preprocess_threads = 16
  images, length_batch, digits_batch = tf.train.shuffle_batch(
      [image, length, digits],
      batch_size=batch_size,
      num_threads=num_preprocess_threads,
      capacity=min_queue_examples + 3 * batch_size,
      min_after_dequeue=min_queue_examples)

  # Display the training images in the visualizer.
  tf.image_summary('images', images)

  return images, length_batch, digits_batch


def distorted_inputs(data_dir, batch_size):
  """Construct distorted input for CIFAR training using the Reader ops.

  Args:
    data_dir: Path to the SVHN data directory.
    batch_size: Number of images per batch.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  filenames = [os.path.join(data_dir, 'train_processed.tfrecords')]
  for f in filenames:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)
    else:
      print('Data used to train is: ', f)

  # Create a queue that produces the filenames to read.
  filename_queue = tf.train.string_input_producer(filenames)

  # Read examples from files in the filename queue.
  read_input = read_svhn(filename_queue)
  reshaped_image = tf.cast(read_input.uint8image, tf.float32)
  length =  tf.cast(read_input.length, tf.float32)
  digits = tf.cast(read_input.label, tf.float32)

  height = IMAGE_SIZE
  width = IMAGE_SIZE
  depth = IMAGE_DEPTH

  # Image processing for training the network. Note the many random
  # distortions applied to the image.

  # Randomly crop a [height, width] section of the image.
  # print(reshaped_image)
  # distorted_image = tf.random_crop(reshaped_image, [height, width, 3])
  # print(distorted_image)
  
  # Randomly flip the image horizontally.
  distorted_image = tf.image.random_flip_left_right(reshaped_image)

  # Because these operations are not commutative, consider randomizing
  # randomize the order their operation.
  distorted_image = tf.image.random_brightness(distorted_image,
                                               max_delta=63)
  distorted_image = tf.image.random_contrast(distorted_image,
                                             lower=0.2, upper=1.8)

  # Subtract off the mean and divide by the variance of the pixels.
  # float_image = tf.image.per_image_whitening(distorted_image)
  float_image = (distorted_image - 128.) / 128.

  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(FLAGS.num_examples_per_epoch_for_train *
                           min_fraction_of_examples_in_queue)
  print ('Filling queue with %d CIFAR images before starting to train. '
         'This will take a few minutes.' % min_queue_examples)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(float_image, length, digits,
                                         min_queue_examples, batch_size)


def inputs(eval_data, data_dir, batch_size):
  """Construct input for CIFAR evaluation using the Reader ops.

  Args:
    eval_data: bool, indicating if one should use the train or eval data set.
    data_dir: Path to the SVHN data directory.
    batch_size: Number of images per batch.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  if not eval_data:
    filenames = [os.path.join(data_dir, 'valid_processed.tfrecords')]
    num_examples_per_epoch = FLAGS.num_examples_per_epoch_for_valid
  else:
    filenames = [os.path.join(data_dir, 'test_processed.tfrecords')]
    num_examples_per_epoch = FLAGS.num_examples_per_epoch_for_test

  for f in filenames:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)

  # Create a queue that produces the filenames to read.
  filename_queue = tf.train.string_input_producer(filenames)

  # Read examples from files in the filename queue.
  read_input = read_svhn(filename_queue)
  reshaped_image = tf.cast(read_input.uint8image, tf.float32)
  length =  tf.cast(read_input.length, tf.float32)
  digits = tf.cast(read_input.label, tf.float32)

  height = IMAGE_SIZE
  width = IMAGE_SIZE

  # Image processing for evaluation.
  # Crop the central [height, width] of the image.
  # resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image, width, height)                                                   

  # Subtract off the mean and divide by the variance of the pixels.
  # float_image = tf.image.per_image_whitening(resized_image)
  float_image = (reshaped_image - 128.) / 128.    


  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(num_examples_per_epoch *
                           min_fraction_of_examples_in_queue)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(float_image, length, digits,
                                         min_queue_examples, batch_size)
