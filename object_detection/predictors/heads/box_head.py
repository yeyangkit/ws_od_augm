# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

"""Box Head.

Contains Box prediction head classes for different meta architectures.
All the box prediction heads have a predict function that receives the
`features` as the first argument and returns `box_encodings`.
"""
import functools
import tensorflow as tf

from object_detection.predictors.heads import head

slim = tf.contrib.slim

class ConvolutionalBoxHead(head.Head):
  """Convolutional box prediction head."""

  def __init__(self,
               is_training,
               box_code_size,
               kernel_size,
               use_depthwise=False,
               box_encodings_clip_range=None):
    """Constructor.

    Args:
      is_training: Indicates whether the BoxPredictor is in training mode.
      box_code_size: Size of encoding for each box.
      kernel_size: Size of final convolution kernel.  If the
        spatial resolution of the feature map is smaller than the kernel size,
        then the kernel size is automatically set to be
        min(feature_width, feature_height).
      use_depthwise: Whether to use depthwise convolutions for prediction
        steps. Default is False.
      box_encodings_clip_range: Min and max values for clipping box_encodings.

    Raises:
      ValueError: if min_depth > max_depth.
    """
    super(ConvolutionalBoxHead, self).__init__()
    self._is_training = is_training
    self._box_code_size = box_code_size
    self._kernel_size = kernel_size
    self._use_depthwise = use_depthwise
    self._box_encodings_clip_range = box_encodings_clip_range
    # self._scope = scope # todo sep24

  def predict(self, features, num_predictions_per_location):
    """Predicts boxes.

    Args:
      features: A float tensor of shape [batch_size, height, width, channels]
        containing image features.
      num_predictions_per_location: Number of box predictions to be made per
        spatial location. Int specifying number of boxes per location.

    Returns:
      box_encodings: A float tensors of shape
        [batch_size, num_anchors, q, code_size] representing the location of
        the objects, where q is 1 or the number of classes.
    """
    net = features
    if self._use_depthwise:
      box_encodings = slim.separable_conv2d(
          net, None, [self._kernel_size, self._kernel_size],
          padding='SAME', depth_multiplier=1, stride=1,
          rate=1, scope='BoxEncodingPredictor_depthwise')#      rate=1, scope=self._scope+'_depthwise')
      box_encodings = slim.conv2d(
          box_encodings,
          num_predictions_per_location * self._box_code_size, [1, 1],
          activation_fn=None,
          normalizer_fn=None,
          normalizer_params=None,
          scope='BoxEncodingPredictor')#      scope=self._scope)
    else:
      box_encodings = slim.conv2d(
          net, num_predictions_per_location * self._box_code_size,
          [self._kernel_size, self._kernel_size],
          activation_fn=None,
          normalizer_fn=None,
          normalizer_params=None,
          scope='BoxEncodingPredictor') #  scope=self._scope)
    batch_size = features.get_shape().as_list()[0]
    if batch_size is None:
      batch_size = tf.shape(features)[0]
    # Clipping the box encodings to make the inference graph TPU friendly.
    if self._box_encodings_clip_range is not None:
      box_encodings = tf.clip_by_value(
          box_encodings, self._box_encodings_clip_range.min,
          self._box_encodings_clip_range.max)

    # # Todo sep24: Why [batch_size, num_anchors, q, code_size] but not [batch_size, num_anchors, code_size]?
    # box_encodings = tf.reshape(box_encodings,
    #                            [batch_size, -1, self._box_code_size])
    box_encodings = tf.reshape(box_encodings,
                              [batch_size, -1, 1, self._box_code_size])
    return box_encodings


# TODO(alirezafathi): See if possible to unify Weight Shared with regular
# convolutional box head.
class WeightSharedConvolutionalBoxHead(head.Head):
  """Weight shared convolutional box prediction head.

  This head allows sharing the same set of parameters (weights) when called more
  then once on different feature maps.
  """

  def __init__(self,
               box_code_size,
               kernel_size=3,
               use_depthwise=False,
               box_encodings_clip_range=None,
               return_flat_predictions=True): # , scope='BoxPredictor'
    """Constructor.

    Args:
      box_code_size: Size of encoding for each box.
      kernel_size: Size of final convolution kernel.
      use_depthwise: Whether to use depthwise convolutions for prediction steps.
        Default is False.
      box_encodings_clip_range: Min and max values for clipping box_encodings.
      return_flat_predictions: If true, returns flattened prediction tensor
        of shape [batch, height * width * num_predictions_per_location,
        box_coder]. Otherwise returns the prediction tensor before reshaping,
        whose shape is [batch, height, width, num_predictions_per_location *
        num_class_slots].
    """
    super(WeightSharedConvolutionalBoxHead, self).__init__()
    self._box_code_size = box_code_size
    self._kernel_size = kernel_size
    self._use_depthwise = use_depthwise
    self._box_encodings_clip_range = box_encodings_clip_range
    self._return_flat_predictions = return_flat_predictions
    # self._scope = scope

  def predict(self, features, num_predictions_per_location):
    """Predicts boxes.

    Args:
      features: A float tensor of shape [batch_size, height, width, channels]
        containing image features.
      num_predictions_per_location: Number of box predictions to be made per
        spatial location.

    Returns:
      box_encodings: A float tensor of shape
        [batch_size, num_anchors, code_size] representing the location of
        the objects, or a float tensor of shape [batch, height, width,
        num_predictions_per_location * box_code_size] representing grid box
        location predictions if self._return_flat_predictions is False.
    """
    box_encodings_net = features
    if self._use_depthwise:
      conv_op = functools.partial(slim.separable_conv2d, depth_multiplier=1)
    else:
      conv_op = slim.conv2d
    box_encodings = conv_op(
        box_encodings_net,
        num_predictions_per_location * self._box_code_size,
        [self._kernel_size, self._kernel_size],
        activation_fn=None, stride=1, padding='SAME',
        normalizer_fn=None,
        scope='BoxPredictor') # scope=self._scope)
    batch_size = features.get_shape().as_list()[0]
    if batch_size is None:
      batch_size = tf.shape(features)[0]
    # Clipping the box encodings to make the inference graph TPU friendly.
    if self._box_encodings_clip_range is not None:
      box_encodings = tf.clip_by_value(
          box_encodings, self._box_encodings_clip_range.min,
          self._box_encodings_clip_range.max)
    if self._return_flat_predictions:
      box_encodings = tf.reshape(box_encodings,
                                 [batch_size, -1, self._box_code_size])
    return box_encodings
