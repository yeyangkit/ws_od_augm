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

"""Tensorflow Example proto decoder for object detection.

A decoder to decode string tensors containing serialized tensorflow.Example
protos for object detection.
"""
import tensorflow as tf

from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import parsing_ops

from object_detection.core import data_decoder
from object_detection.core import standard_fields as fields
from object_detection.protos import input_reader_pb2
from object_detection.utils import label_map_util

from object_detection.data_decoders.tf_example_decoder import slim_example_decoder
from object_detection.data_decoders.tf_example_decoder import _BackupHandler
from object_detection.data_decoders.tf_example_decoder import _ClassTensorHandler

class BoundingBox3d(slim_example_decoder.ItemHandler):

    def __init__(self, keys=None, prefix=None):
        self._prefix = prefix
        self._keys = keys
        self._full_keys = [prefix + k for k in keys]
        super(BoundingBox3d, self).__init__(self._full_keys)

    def tensors_to_item(self, keys_to_tensors):
        sides = []
        for key in self._full_keys:
            side = keys_to_tensors[key]
            if isinstance(side, tf.SparseTensor):
                side = side.values
            side = tf.expand_dims(side, 0)
            sides.append(side)

        bounding_box = tf.concat(sides, 0)
        return tf.transpose(bounding_box)

class MultilayerImages(slim_example_decoder.ItemHandler):

    def __init__(self,
                 image_keys=None,
                 layer_channels=None,
                 format_key=None,
                 shape=None,
                 num_channels=3,
                 dtype=dtypes.uint8,
                 dct_method=''):
        if not format_key:
            format_key = 'image/format'

        super(MultilayerImages, self).__init__(image_keys + [format_key])
        self._image_keys = image_keys
        self._layer_channels = layer_channels
        self._format_key = format_key
        self._shape = shape
        self._num_channels = num_channels
        self._dtype = dtype
        self._dct_method = dct_method

    def tensors_to_item(self, keys_to_tensors):
        image_format = keys_to_tensors[self._format_key]
        image_tensor = self._decode(keys_to_tensors[self._image_keys[0]], self._layer_channels[0], image_format)
        for idx in range(1, len(self._image_keys)):
            image_tensor_layer = self._decode(keys_to_tensors[self._image_keys[idx]], self._layer_channels[idx], image_format)
            image_tensor = tf.concat([image_tensor, image_tensor_layer], -1)
        image_tensor.set_shape([None, None, self._num_channels])
        if self._shape is not None:
            image_tensor = array_ops.reshape(image_tensor, self._shape)
        return image_tensor


    def _decode(self, image_buffer, num_channels, image_format):
        def decode_image():
            return image_ops.decode_image(image_buffer, channels=num_channels)

        def decode_jpeg():
            return image_ops.decode_jpeg(
                image_buffer, channels=num_channels, dct_method=self._dct_method)

        def check_jpeg():
            return control_flow_ops.cond(
                image_ops.is_jpeg(image_buffer),
                decode_jpeg,
                decode_image,
                name='cond_jpeg')

        def decode_raw():
            return parsing_ops.decode_raw(image_buffer, out_type=self._dtype)

        pred_fn_pairs = {
            math_ops.logical_or(
                math_ops.equal(image_format, 'raw'),
                math_ops.equal(image_format, 'RAW')): decode_raw,
        }
        image = control_flow_ops.case(
            pred_fn_pairs, default=check_jpeg, exclusive=True)

        return image

class TfMultiLayerDecoder(data_decoder.DataDecoder):
  """Tensorflow Example proto decoder."""

  def __init__(self,
               box_params,
               input_features,
               input_channels,
               load_instance_masks=False,
               instance_mask_type=input_reader_pb2.NUMERICAL_MASKS,
               label_map_proto_file=None,
               use_display_name=False,
               dct_method='',
               num_keypoints=0,
               num_additional_channels=0):
    """Constructor sets keys_to_features and items_to_handlers.

    Args:
      load_instance_masks: whether or not to load and handle instance masks.
      instance_mask_type: type of instance masks. Options are provided in
        input_reader.proto. This is only used if `load_instance_masks` is True.
      label_map_proto_file: a file path to a
        object_detection.protos.StringIntLabelMap proto. If provided, then the
        mapped IDs of 'image/object/class/text' will take precedence over the
        existing 'image/object/class/label' ID.  Also, if provided, it is
        assumed that 'image/object/class/text' will be in the data.
      use_display_name: whether or not to use the `display_name` for label
        mapping (instead of `name`).  Only used if label_map_proto_file is
        provided.
      dct_method: An optional string. Defaults to None. It only takes
        effect when image format is jpeg, used to specify a hint about the
        algorithm used for jpeg decompression. Currently valid values
        are ['INTEGER_FAST', 'INTEGER_ACCURATE']. The hint may be ignored, for
        example, the jpeg library does not have that specific option.
      num_keypoints: the number of keypoints per object.

    Raises:
      ValueError: If `instance_mask_type` option is not one of
        input_reader_pb2.DEFAULT, input_reader_pb2.NUMERICAL, or
        input_reader_pb2.PNG_MASKS.
    """
    del use_display_name
    self.keys_to_features = {
        'id': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='png'),
        'image/filename':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'image/key/sha256':
            tf.FixedLenFeature((), tf.string, default_value=''),

        'layers/height': tf.FixedLenFeature((), tf.int64, default_value=1),
        'layers/width': tf.FixedLenFeature((), tf.int64, default_value=1),

        'layers/detections/encoded': tf.FixedLenFeature((), tf.string),
        'layers/observations/encoded': tf.FixedLenFeature((), tf.string),
        'layers/decay_rate/encoded': tf.FixedLenFeature((), tf.string),
        'layers/intensity/encoded': tf.FixedLenFeature((), tf.string),
        'layers/zmin/encoded': tf.FixedLenFeature((), tf.string),
        'layers/zmax/encoded': tf.FixedLenFeature((), tf.string),
        'layers/occlusions/encoded': tf.FixedLenFeature((), tf.string),
        'layers/rgb/encoded': tf.FixedLenFeature((), tf.string),

        'masks/occupancy_mask': tf.FixedLenFeature((), tf.string),

        'boxes/aligned/x_min': tf.VarLenFeature(tf.float32),
        'boxes/aligned/x_max': tf.VarLenFeature(tf.float32),
        'boxes/aligned/y_min': tf.VarLenFeature(tf.float32),
        'boxes/aligned/y_max': tf.VarLenFeature(tf.float32),

        'boxes/class/label': tf.VarLenFeature(tf.int64),
        'boxes/class/text': tf.VarLenFeature(tf.string),

        'image/class/text':
            tf.VarLenFeature(tf.string),
        'image/class/label':
            tf.VarLenFeature(tf.int64),
        'image/object/area':
            tf.VarLenFeature(tf.float32),
        'image/object/is_crowd':
            tf.VarLenFeature(tf.int64),
        'image/object/difficult':
            tf.VarLenFeature(tf.int64),
        'image/object/group_of':
            tf.VarLenFeature(tf.int64),
        'image/object/weight':
            tf.VarLenFeature(tf.float32),
    }
    for param in box_params:
        self.keys_to_features['boxes/inclined/' + param] = tf.VarLenFeature(tf.float32)

    self._num_input_channels = sum(input_channels)
    image_keys = ['layers/' + input_feature + '/encoded' for input_feature in input_features]

    # We are checking `dct_method` instead of passing it directly in order to
    # ensure TF version 1.6 compatibility.
    if dct_method:
      image = MultilayerImages(
          image_keys=image_keys,
          layer_channels=input_channels,
          format_key='image/format',
          num_channels= self._num_input_channels,
          dct_method=dct_method)
    else:
      image = MultilayerImages(
          image_keys=image_keys, layer_channels=input_channels, format_key='image/format', num_channels=self._num_input_channels)

    occupancy_mask = slim_example_decoder.Image(image_key='masks/occupancy_mask', format_key='image/format', channels=1)

    self.items_to_handlers = {
        fields.InputDataFields.image:
            image,
        fields.InputDataFields.occupancy_mask:
            occupancy_mask,
        fields.InputDataFields.source_id: (
            slim_example_decoder.Tensor('id')),
        fields.InputDataFields.key: (
            slim_example_decoder.Tensor('image/key/sha256')),
        fields.InputDataFields.filename: (
            slim_example_decoder.Tensor('image/filename')),
        # Object boxes and classes.
        fields.InputDataFields.groundtruth_boxes: (
            slim_example_decoder.BoundingBox(['y_min', 'x_min', 'y_max', 'x_max'],
                                             'boxes/aligned/')),
        fields.InputDataFields.groundtruth_boxes_3d: (
            BoundingBox3d(box_params, 'boxes/inclined/')),
        fields.InputDataFields.groundtruth_area:
            slim_example_decoder.Tensor('image/object/area'),
        fields.InputDataFields.groundtruth_is_crowd: (
            slim_example_decoder.Tensor('image/object/is_crowd')),
        fields.InputDataFields.groundtruth_difficult: (
            slim_example_decoder.Tensor('image/object/difficult')),
        fields.InputDataFields.groundtruth_group_of: (
            slim_example_decoder.Tensor('image/object/group_of')),
        fields.InputDataFields.groundtruth_weights: (
            slim_example_decoder.Tensor('image/object/weight')),
    }
    self._num_keypoints = num_keypoints
    if num_keypoints > 0:
      self.keys_to_features['image/object/keypoint/x'] = (
          tf.VarLenFeature(tf.float32))
      self.keys_to_features['image/object/keypoint/y'] = (
          tf.VarLenFeature(tf.float32))
      self.items_to_handlers[fields.InputDataFields.groundtruth_keypoints] = (
          slim_example_decoder.ItemHandlerCallback(
              ['image/object/keypoint/y', 'image/object/keypoint/x'],
              self._reshape_keypoints))
    if load_instance_masks:
      if instance_mask_type in (input_reader_pb2.DEFAULT,
                                input_reader_pb2.NUMERICAL_MASKS):
        self.keys_to_features['image/object/mask'] = (
            tf.VarLenFeature(tf.float32))
        self.items_to_handlers[
            fields.InputDataFields.groundtruth_instance_masks] = (
                slim_example_decoder.ItemHandlerCallback(
                    ['image/object/mask', 'layers/height', 'layers/width'],
                    self._reshape_instance_masks))
      elif instance_mask_type == input_reader_pb2.PNG_MASKS:
        self.keys_to_features['image/object/mask'] = tf.VarLenFeature(tf.string)
        self.items_to_handlers[
            fields.InputDataFields.groundtruth_instance_masks] = (
                slim_example_decoder.ItemHandlerCallback(
                    ['image/object/mask', 'layers/height', 'layers/width'],
                    self._decode_png_instance_masks))
      else:
        raise ValueError('Did not recognize the `instance_mask_type` option.')
    if label_map_proto_file:
      # If the label_map_proto is provided, try to use it in conjunction with
      # the class text, and fall back to a materialized ID.
      label_handler = _BackupHandler(
          _ClassTensorHandler('boxes/class/text', label_map_proto_file, default_value=''),
          slim_example_decoder.Tensor('boxes/class/label'))
      image_label_handler = _BackupHandler(
          _ClassTensorHandler(
              fields.TfExampleFields.image_class_text, label_map_proto_file, default_value=''),
          slim_example_decoder.Tensor(fields.TfExampleFields.image_class_label))
    else:
      label_handler = slim_example_decoder.Tensor('boxes/class/label')
      image_label_handler = slim_example_decoder.Tensor(
          fields.TfExampleFields.image_class_label)
    self.items_to_handlers[
        fields.InputDataFields.groundtruth_classes] = label_handler
    self.items_to_handlers[
        fields.InputDataFields.groundtruth_image_classes] = image_label_handler

  def decode(self, tf_example_string_tensor):
    """Decodes serialized tensorflow example and returns a tensor dictionary.

    Args:
      tf_example_string_tensor: a string tensor holding a serialized tensorflow
        example proto.

    Returns:
      A dictionary of the following tensors.
      fields.InputDataFields.image - 3D uint8 tensor of shape [None, None, 3]
        containing image.
      fields.InputDataFields.occupancy mask - 1D uint8 tensor of shape
        [None, None, 3] containing occupancy mask.
      fields.InputDataFields.source_id - string tensor containing original
        image id.
      fields.InputDataFields.key - string tensor with unique sha256 hash key.
      fields.InputDataFields.filename - string tensor with original dataset
        filename.
      fields.InputDataFields.groundtruth_boxes - 2D float32 tensor of shape
        [None, 4] containing box corners.
      fields.InputDataFields.groundtruth_classes - 1D int64 tensor of shape
        [None] containing classes for the boxes.
      fields.InputDataFields.groundtruth_weights - 1D float32 tensor of
        shape [None] indicating the weights of groundtruth boxes.
      fields.InputDataFields.num_groundtruth_boxes - int32 scalar indicating
        the number of groundtruth_boxes.
      fields.InputDataFields.groundtruth_area - 1D float32 tensor of shape
        [None] containing containing object mask area in pixel squared.
      fields.InputDataFields.groundtruth_is_crowd - 1D bool tensor of shape
        [None] indicating if the boxes enclose a crowd.

    Optional:
      fields.InputDataFields.groundtruth_difficult - 1D bool tensor of shape
        [None] indicating if the boxes represent `difficult` instances.
      fields.InputDataFields.groundtruth_group_of - 1D bool tensor of shape
        [None] indicating if the boxes represent `group_of` instances.
      fields.InputDataFields.groundtruth_keypoints - 3D float32 tensor of
        shape [None, None, 2] containing keypoints, where the coordinates of
        the keypoints are ordered (y, x).
      fields.InputDataFields.groundtruth_instance_masks - 3D float32 tensor of
        shape [None, None, None] containing instance masks.
    """
    serialized_example = tf.reshape(tf_example_string_tensor, shape=[])
    decoder = slim_example_decoder.TFExampleDecoder(self.keys_to_features,
                                                    self.items_to_handlers)
    keys = decoder.list_items()
    tensors = decoder.decode(serialized_example, items=keys)
    tensor_dict = dict(zip(keys, tensors))
    is_crowd = fields.InputDataFields.groundtruth_is_crowd
    tensor_dict[is_crowd] = tf.cast(tensor_dict[is_crowd], dtype=tf.bool)
    tensor_dict[fields.InputDataFields.image].set_shape([None, None, self._num_input_channels])
    tensor_dict[fields.InputDataFields.occupancy_mask].set_shape([None, None, 1])
    tensor_dict[fields.InputDataFields.original_image_spatial_shape] = tf.shape(
        tensor_dict[fields.InputDataFields.image])[:2]
    tensor_dict[fields.InputDataFields.num_groundtruth_boxes] = tf.shape(
        tensor_dict[fields.InputDataFields.groundtruth_boxes])[0]

    def default_groundtruth_weights():
      return tf.ones(
          [tf.shape(tensor_dict[fields.InputDataFields.groundtruth_boxes])[0]],
          dtype=tf.float32)

    tensor_dict[fields.InputDataFields.groundtruth_weights] = tf.cond(
        tf.greater(
            tf.shape(
                tensor_dict[fields.InputDataFields.groundtruth_weights])[0],
            0), lambda: tensor_dict[fields.InputDataFields.groundtruth_weights],
        default_groundtruth_weights)
    return tensor_dict

  def _reshape_keypoints(self, keys_to_tensors):
    """Reshape keypoints.

    The instance segmentation masks are reshaped to [num_instances,
    num_keypoints, 2].

    Args:
      keys_to_tensors: a dictionary from keys to tensors.

    Returns:
      A 3-D float tensor of shape [num_instances, num_keypoints, 2] with values
        in {0, 1}.
    """
    y = keys_to_tensors['image/object/keypoint/y']
    if isinstance(y, tf.SparseTensor):
      y = tf.sparse_tensor_to_dense(y)
    y = tf.expand_dims(y, 1)
    x = keys_to_tensors['image/object/keypoint/x']
    if isinstance(x, tf.SparseTensor):
      x = tf.sparse_tensor_to_dense(x)
    x = tf.expand_dims(x, 1)
    keypoints = tf.concat([y, x], 1)
    keypoints = tf.reshape(keypoints, [-1, self._num_keypoints, 2])
    return keypoints

  def _reshape_instance_masks(self, keys_to_tensors):
    """Reshape instance segmentation masks.

    The instance segmentation masks are reshaped to [num_instances, height,
    width].

    Args:
      keys_to_tensors: a dictionary from keys to tensors.

    Returns:
      A 3-D float tensor of shape [num_instances, height, width] with values
        in {0, 1}.
    """
    height = keys_to_tensors['layers/height']
    width = keys_to_tensors['layers/width']
    to_shape = tf.cast(tf.stack([-1, height, width]), tf.int32)
    masks = keys_to_tensors['image/object/mask']
    if isinstance(masks, tf.SparseTensor):
      masks = tf.sparse_tensor_to_dense(masks)
    masks = tf.reshape(tf.to_float(tf.greater(masks, 0.0)), to_shape)
    return tf.cast(masks, tf.float32)

  def _decode_png_instance_masks(self, keys_to_tensors):
    """Decode PNG instance segmentation masks and stack into dense tensor.

    The instance segmentation masks are reshaped to [num_instances, height,
    width].

    Args:
      keys_to_tensors: a dictionary from keys to tensors.

    Returns:
      A 3-D float tensor of shape [num_instances, height, width] with values
        in {0, 1}.
    """

    def decode_png_mask(image_buffer):
      image = tf.squeeze(
          tf.image.decode_image(image_buffer, channels=1), axis=2)
      image.set_shape([None, None])
      image = tf.to_float(tf.greater(image, 0))
      return image

    png_masks = keys_to_tensors['image/object/mask']
    height = keys_to_tensors['layers/height']
    width = keys_to_tensors['layers/width']
    if isinstance(png_masks, tf.SparseTensor):
      png_masks = tf.sparse_tensor_to_dense(png_masks, default_value='')
    return tf.cond(
        tf.greater(tf.size(png_masks), 0),
        lambda: tf.map_fn(decode_png_mask, png_masks, dtype=tf.float32),
        lambda: tf.zeros(tf.to_int32(tf.stack([0, height, width]))))