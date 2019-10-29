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
"""SSD Meta-architecture definition.

General tensorflow implementation of convolutional Multibox/SSD detection
models.
"""
import abc
import tensorflow as tf

from object_detection.core import box_list
from object_detection.core import box_list_ops
from object_detection.core import matcher
from object_detection.core import model
from object_detection.core import standard_fields as fields
from object_detection.core import target_assigner
from object_detection.utils import ops
from object_detection.utils import shape_utils
from object_detection.utils import visualization_utils
from tensorflow import shape

slim = tf.contrib.slim


class SSDAugmentationHybridSeqMetaArch(model.DetectionModel):
  """SSD Augmentatio Meta-architecture definition."""

  def __init__(self,
               is_training,
               anchor_generator,
               box_predictor,
               augmentation_predictor,
               factor_loss_fused_bel_O,
               factor_loss_fused_bel_F,
               factor_loss_fused_zmax_det,
               factor_loss_fused_obs_zmin,
               factor_loss_augm,
               box_coder,
               feature_extractor,
               encode_background_as_zeros,
               image_resizer_fn,
               non_max_suppression_fn,
               score_conversion_fn,
               use_uncertainty_weighting_loss,
               classification_loss,
               localization_loss,
               classification_loss_weight,
               localization_loss_weight,
               normalize_loss_by_num_matches,
               hard_example_miner,
               target_assigner_instance,
               add_summaries=True,
               normalize_loc_loss_by_codesize=False,
               freeze_batchnorm=False,
               inplace_batchnorm_update=False,
               add_background_class=True,
               explicit_background_class=False,
               random_example_sampler=None,
               expected_loss_weights_fn=None,
               use_confidences_as_targets=False,
               implicit_example_weight=0.5,
               equalization_loss_config=None):
    """SSDMetaArch Constructor.

    TODO(rathodv,jonathanhuang): group NMS parameters + score converter into
    a class and loss parameters into a class and write config protos for
    postprocessing and losses.

    Args:
      is_training: A boolean indicating whether the training version of the
        computation graph should be constructed.
      anchor_generator: an anchor_generator.AnchorGenerator object.
      box_predictor: a box_predictor.BoxPredictor object.
      box_coder: a box_coder.BoxCoder object.
      feature_extractor: a SSDFeatureExtractor object.
      encode_background_as_zeros: boolean determining whether background
        targets are to be encoded as an all zeros vector or a one-hot
        vector (where background is the 0th class).
      image_resizer_fn: a callable for image resizing.  This callable always
        takes a rank-3 image tensor (corresponding to a single image) and
        returns a rank-3 image tensor, possibly with new spatial dimensions and
        a 1-D tensor of shape [3] indicating shape of true image within
        the resized image tensor as the resized image tensor could be padded.
        See builders/image_resizer_builder.py.
      non_max_suppression_fn: batch_multiclass_non_max_suppression
        callable that takes `boxes`, `scores` and optional `clip_window`
        inputs (with all other inputs already set) and returns a dictionary
        hold tensors with keys: `detection_boxes`, `detection_scores`,
        `detection_classes` and `num_detections`. See `post_processing.
        batch_multiclass_non_max_suppression` for the type and shape of these
        tensors.
      score_conversion_fn: callable elementwise nonlinearity (that takes tensors
        as inputs and returns tensors).  This is usually used to convert logits
        to probabilities.
      classification_loss: an object_detection.core.losses.Loss object.
      localization_loss: a object_detection.core.losses.Loss object.
      classification_loss_weight: float
      localization_loss_weight: float
      normalize_loss_by_num_matches: boolean
      hard_example_miner: a losses.HardExampleMiner object (can be None)
      target_assigner_instance: target_assigner.TargetAssigner instance to use.
      add_summaries: boolean (default: True) controlling whether summary ops
        should be added to tensorflow graph.
      normalize_loc_loss_by_codesize: whether to normalize localization loss
        by code size of the box encoder.
      freeze_batchnorm: Whether to freeze batch norm parameters during
        training or not. When training with a small batch size (e.g. 1), it is
        desirable to freeze batch norm update and use pretrained batch norm
        params.
      inplace_batchnorm_update: Whether to update batch norm moving average
        values inplace. When this is false train op must add a control
        dependency on tf.graphkeys.UPDATE_OPS collection in order to update
        batch norm statistics.
      add_background_class: Whether to add an implicit background class to
        one-hot encodings of groundtruth labels. Set to false if training a
        single class model or using groundtruth labels with an explicit
        background class.
      explicit_background_class: Set to true if using groundtruth labels with an
        explicit background class, as in multiclass scores.
      random_example_sampler: a BalancedPositiveNegativeSampler object that can
        perform random example sampling when computing loss. If None, random
        sampling process is skipped. Note that random example sampler and hard
        example miner can both be applied to the model. In that case, random
        sampler will take effect first and hard example miner can only process
        the random sampled examples.
      expected_loss_weights_fn: If not None, use to calculate
        loss by background/foreground weighting. Should take batch_cls_targets
        as inputs and return foreground_weights, background_weights. See
        expected_classification_loss_by_expected_sampling and
        expected_classification_loss_by_reweighting_unmatched_anchors in
        third_party/tensorflow_models/object_detection/utils/ops.py as examples.
      use_confidences_as_targets: Whether to use groundtruth_condifences field
        to assign the targets.
      implicit_example_weight: a float number that specifies the weight used
        for the implicit negative examples.
      equalization_loss_config: a namedtuple that specifies configs for
        computing equalization loss.
    """
    super(SSDAugmentationHybridSeqMetaArch, self).__init__(num_classes=box_predictor.num_classes)
    self._is_training = is_training
    self._freeze_batchnorm = freeze_batchnorm
    self._inplace_batchnorm_update = inplace_batchnorm_update

    self._anchor_generator = anchor_generator
    self._box_predictor = box_predictor
    self._augm_predictor = augmentation_predictor

    self._box_coder = box_coder
    self._feature_extractor = feature_extractor
    self._add_background_class = add_background_class
    self._explicit_background_class = explicit_background_class

    self._factor_loss_fused_bel_O = factor_loss_fused_bel_O
    self._factor_loss_fused_bel_F = factor_loss_fused_bel_F
    self._factor_loss_fused_zmax_det = factor_loss_fused_zmax_det
    self._factor_loss_fused_obs_zmin = factor_loss_fused_obs_zmin
    self._factor_loss_augm = factor_loss_augm

    if add_background_class and explicit_background_class:
      raise ValueError("Cannot have both 'add_background_class' and"
                       " 'explicit_background_class' true.")

    # Needed for fine-tuning from classification checkpoints whose
    # variables do not have the feature extractor scope.
    if self._feature_extractor.is_keras_model:
      # Keras feature extractors will have a name they implicitly use to scope.
      # So, all contained variables are prefixed by this name.
      # To load from classification checkpoints, need to filter out this name.
      self._extract_features_scope = feature_extractor.name
    else:
      # Slim feature extractors get an explicit naming scope
      self._extract_features_scope = 'FeatureExtractor'

    if encode_background_as_zeros:
      background_class = [0]
    else:
      background_class = [1]

    if self._add_background_class:
      num_foreground_classes = self.num_classes
    else:
      num_foreground_classes = self.num_classes - 1

    self._unmatched_class_label = tf.constant(
      background_class + num_foreground_classes * [0], tf.float32)

    self._target_assigner = target_assigner_instance

    self._use_uncertainty_weighting_loss = use_uncertainty_weighting_loss
    self._classification_loss = classification_loss
    self._localization_loss = localization_loss
    self._classification_loss_weight = classification_loss_weight
    self._localization_loss_weight = localization_loss_weight
    self._normalize_loss_by_num_matches = normalize_loss_by_num_matches
    self._normalize_loc_loss_by_codesize = normalize_loc_loss_by_codesize
    self._hard_example_miner = hard_example_miner
    self._random_example_sampler = random_example_sampler
    self._parallel_iterations = 16
    # self._fpn_levels = fpn_levels # todo sep24 hestitate to use this
    self._image_resizer_fn = image_resizer_fn
    self._non_max_suppression_fn = non_max_suppression_fn
    self._score_conversion_fn = score_conversion_fn

    self._anchors = None
    self._indicators = None
    self._add_summaries = add_summaries
    self._batched_prediction_tensor_names = []
    self._expected_loss_weights_fn = expected_loss_weights_fn
    self._use_confidences_as_targets = use_confidences_as_targets
    self._implicit_example_weight = implicit_example_weight

    self._equalization_loss_config = equalization_loss_config

  @property
  def anchors(self):
    if not self._anchors:
      raise RuntimeError('anchors have not been constructed yet!')
    if not isinstance(self._anchors, box_list.BoxList):
      raise RuntimeError('anchors should be a BoxList object, but is not.')
    # if not isinstance(self._anchors, list):
    #   if not isinstance(self._anchors, box_list.BoxList):
    #     raise RuntimeError('anchors should be a BoxList object, but is not.')
    # else:
    #   if not all(
    #     isinstance(anchors, box_list.BoxList) for anchors in self._anchors):
    #     raise ValueError('anchors_batch must be a BoxList or list of BoxLists.')
    return self._anchors

  @property
  def batched_prediction_tensor_names(self):
    if not self._batched_prediction_tensor_names:
      raise RuntimeError('Must call predict() method to get batched prediction '
                         'tensor names.')
    return self._batched_prediction_tensor_names

  def preprocess(self, inputs):
    """Feature-extractor specific preprocessing.

    SSD meta architecture uses a default clip_window of [0, 0, 1, 1] during
    post-processing. On calling `preprocess` method, clip_window gets updated
    based on `true_image_shapes` returned by `image_resizer_fn`.

    Args:
      inputs: a [batch, height_in, width_in, channels] float tensor representing
        a batch of images with values between 0 and 255.0.

    Returns:
      preprocessed_inputs: a [batch, height_out, width_out, channels] float
        tensor representing a batch of images.
      true_image_shapes: int32 tensor of shape [batch, 3] where each row is
        of the form [height, width, channels] indicating the shapes
        of true images in the resized images, as resized images can be padded
        with zeros.

    Raises:
      ValueError: if inputs tensor does not have type tf.float32
    """
    if inputs.dtype is not tf.float32:
      raise ValueError('`preprocess` expects a tf.float32 tensor')
    with tf.name_scope('Preprocessor'):
      # TODO(jonathanhuang): revisit whether to always use batch size as
      # the number of parallel iterations vs allow for dynamic batching.
      outputs = shape_utils.static_or_dynamic_map_fn(
        self._image_resizer_fn,
        elems=inputs,
        dtype=[tf.float32, tf.int32])
      resized_inputs = outputs[0]
      true_image_shapes = outputs[1]

      return (self._feature_extractor.preprocess(resized_inputs),
              true_image_shapes)

  def _compute_clip_window(self, preprocessed_images, true_image_shapes):
    """Computes clip window to use during post_processing.

    Computes a new clip window to use during post-processing based on
    `resized_image_shapes` and `true_image_shapes` only if `preprocess` method
    has been called. Otherwise returns a default clip window of [0, 0, 1, 1].

    Args:
      preprocessed_images: the [batch, height, width, channels] image
          tensor.
      true_image_shapes: int32 tensor of shape [batch, 3] where each row is
        of the form [height, width, channels] indicating the shapes
        of true images in the resized images, as resized images can be padded
        with zeros. Or None if the clip window should cover the full image.

    Returns:
      a 2-D float32 tensor of the form [batch_size, 4] containing the clip
      window for each image in the batch in normalized coordinates (relative to
      the resized dimensions) where each clip window is of the form [ymin, xmin,
      ymax, xmax] or a default clip window of [0, 0, 1, 1].

    """
    if true_image_shapes is None:
      return tf.constant([0, 0, 1, 1], dtype=tf.float32)

    resized_inputs_shape = shape_utils.combined_static_and_dynamic_shape(
      preprocessed_images)
    true_heights, true_widths, _ = tf.unstack(
      tf.cast(true_image_shapes, dtype=tf.float32), axis=1)
    padded_height = tf.cast(resized_inputs_shape[1], dtype=tf.float32)
    padded_width = tf.cast(resized_inputs_shape[2], dtype=tf.float32)
    return tf.stack(
      [
        tf.zeros_like(true_heights),
        tf.zeros_like(true_widths), true_heights / padded_height,
                                    true_widths / padded_width
      ],
      axis=1)

  def _concat_augm_pyramid(self, prediction_dict, feature_maps):
    feature_maps_augm = []
    augm_maps = []
    augm_map = tf.concat([prediction_dict['belief_F_prediction'],
                          prediction_dict['belief_O_prediction'],
                          prediction_dict['z_max_detections_prediction'],
                          prediction_dict['z_min_observations_prediction'],
                          prediction_dict['belief_U_prediction'],
                          prediction_dict['z_min_detections_prediction'],
                          prediction_dict['detections_drivingCorridor_prediction']], axis=3)

    for idx in range(4):
      # _, resized_augm_map, _ = self._image_resizer_fn(augm_map,feature_maps[idx])
      resized_augm_map = tf.image.resize_bicubic(augm_map,
                                                 (tf.shape(feature_maps[idx])[1], tf.shape(feature_maps[idx])[2]))
      augm_maps.append(resized_augm_map)

    for idx, augm_level in enumerate(augm_maps):
      feature_maps_augm.append(tf.concat([feature_maps[idx], augm_level], axis=-1))
    return feature_maps_augm

  def _conv_augm_pyramid(self, prediction_dict, feature_maps, conv_feature_maps_num):

    def _conv_bn_relu(x, filters, ksize, stride):
      x = tf.layers.conv2d(x, filters=filters, kernel_size=ksize, strides=stride, padding='same')
      x = tf.nn.relu(x)
      return x

    def _conv_block(x, filters, training, stack_size, ksize, name):
      with tf.variable_scope(name):
        for i in range(stack_size):
          with tf.variable_scope('augmented2features_conv_%i' % i):
            x = _conv_bn_relu(x, filters=filters, ksize=ksize, stride=1)
        return x

    def _create_unet(x, f, kernel_size, stack_size, output_channel, depth, training):

      skips = []
      level_outputs = []

      for i in range(depth):
        x = _conv_block(x, filters=f * (2 ** i), training=training, stack_size=stack_size, ksize=kernel_size,
                        name="augmented2features_enc_%i" % i)
        skips.append(x)
        x = tf.layers.max_pooling2d(x, pool_size=2, strides=2, name='augmented2features_pool_%i' % (i + 1))

      x = _conv_block(x, filters=f * (2 ** (depth - 1)), training=training, stack_size=stack_size,
                      ksize=kernel_size, name="augmented2features_deep")
      with tf.variable_scope("augmented2features_deep_end"):
        x = tf.layers.conv2d(x, filters=output_channel, kernel_size=1, strides=1,
                             padding='same')
      level_outputs.append(x)

      for i in reversed(range(depth - 1)):
        with tf.variable_scope('augmented2features_up_conv_%i' % (i + 1)):
          x = tf.layers.conv2d_transpose(x, filters=f * (2 ** i), kernel_size=kernel_size, strides=2,
                                         padding='same')
          # if FLAGS.layer_norm:
          #     x = clayer.layer_norm(x, scale=False)
          x = tf.nn.relu(x)
        x = tf.concat([skips.pop(), x], axis=3)

        x = _conv_block(x, filters=f * (2 ** i), training=training, stack_size=stack_size, ksize=kernel_size,
                        name="augmented2features_dec_%i" % i)

        with tf.variable_scope("augmented2features_end_%i" % i):
          x = tf.layers.conv2d(x, filters=output_channel, kernel_size=1, strides=1,
                               padding='same')
        level_outputs.append(x)

      print("level_outputs-------------------------------------------------------------------------------")
      print(level_outputs)

      return level_outputs

    feature_maps_augm = []
    augm_map_dic = tf.concat([prediction_dict['belief_F_prediction'],
                              prediction_dict['belief_O_prediction'],
                              prediction_dict['z_max_detections_prediction'],
                              prediction_dict['z_min_observations_prediction'],
                              prediction_dict['belief_U_prediction'],
                              prediction_dict['z_min_detections_prediction'],
                              prediction_dict['detections_drivingCorridor_prediction']], axis=3)

    level_outputs = _create_unet(augm_map_dic, f=2, kernel_size=3, stack_size=1,
                                 output_channel=conv_feature_maps_num, depth=4, training=True)

    for idx, augm_level in enumerate(level_outputs):
      feature_maps_augm.append(tf.concat([feature_maps[3 - idx], augm_level], axis=-1))
    return feature_maps_augm

  # def _concat_augm_preprocessed_inputs(self, prediction_dict, preprocessed_inputs):
  #     augm = tf.concat([prediction_dict['belief_F_prediction'],
  #                           prediction_dict['belief_O_prediction'],
  #                           prediction_dict['z_max_detections_prediction'],
  #                           prediction_dict['z_min_observations_prediction'],
  #                           prediction_dict['belief_U_prediction'],
  #                           prediction_dict['z_min_detections_prediction'],
  #                           prediction_dict['detections_drivingCorridor_prediction']], axis=3)
  #     concated_inputs = tf.concat([augm, preprocessed_inputs], axis=3)
  #     return concated_inputs

  def _concat_augm_preprocessed_inputs(self, prediction_dict, preprocessed_inputs):

    belief_F_prediction = tf.stop_gradient(prediction_dict['belief_F_prediction'])
    belief_O_prediction = tf.stop_gradient(prediction_dict['belief_O_prediction'])
    z_max_detections_prediction = tf.stop_gradient(prediction_dict['z_max_detections_prediction'])
    z_min_observations_prediction = tf.stop_gradient(prediction_dict['z_min_observations_prediction'])
    belief_U_prediction = tf.stop_gradient(prediction_dict['belief_U_prediction'])
    z_min_detections_prediction = tf.stop_gradient(prediction_dict['z_min_detections_prediction'])
    detections_drivingCorridor_prediction = tf.stop_gradient(
      prediction_dict['detections_drivingCorridor_prediction'])
    intensity_prediction = tf.stop_gradient(prediction_dict['intensity_prediction'])
    augm = tf.concat([belief_F_prediction,
                      belief_O_prediction,
                      z_max_detections_prediction,
                      z_min_observations_prediction,
                      belief_U_prediction,
                      z_min_detections_prediction,
                      detections_drivingCorridor_prediction,
                      intensity_prediction], axis=3)
    concated_inputs = tf.concat([augm, preprocessed_inputs], axis=3)
    return concated_inputs

  def _multiResUnet_block(self, x, depth, name):
    """
    reference https://arxiv.org/abs/1902.04049
    """
    x = tf.layers.conv2d(x, depth, [1, 1], name="multiRes_Block_{}_bottleneckIn".format(name), padding='same')
    x = tf.nn.relu(x, name="multiRes_Block_{}_bottleneckIn_relu".format(name))
    x1 = tf.layers.conv2d(x, depth, [3, 3], name="multiRes_Block_{}_inceptionConv3".format(name), padding='same')
    x1 = tf.nn.relu(x1, name="multiRes_Block_{}_inceptionConv3_relu".format(name))
    x2 = tf.layers.conv2d(x1, depth, [3, 3], name="multiRes_Block_{}_inceptionConv5".format(name), padding='same')
    x2 = tf.nn.relu(x2, name="multiRes_Block_{}_inceptionConv5_relu".format(name))
    x3 = tf.layers.conv2d(x2, depth, [3, 3], name="multiRes_Block_{}_inceptionConv7".format(name), padding='same')
    x3 = tf.nn.relu(x3, name="multiRes_Block_{}_inceptionConv7_relu".format(name))
    x4 = tf.concat((x1, x2, x3), axis=3, name="multiRes_Block_{}_concat".format(name))
    x4 = tf.layers.conv2d(x4, depth, [1, 1], name="multiRes_Block_{}_bottleneckOut".format(name), padding='same')
    x4 = tf.nn.relu(x4, name="multiRes_Block_{}_bottleneckOut_relu".format(name))
    x += x4
    return x

  def _multiResUnet_block_v2(self, x, depth, depth_out, name):
    """
    reference https://arxiv.org/abs/1902.04049
    """
    x = tf.layers.conv2d(x, depth, [1, 1], name="multiRes_Block_{}_bottleneckIn".format(name), padding='same')
    x = tf.nn.relu(x, name="multiRes_Block_{}_bottleneckIn_relu".format(name))
    x1 = tf.layers.conv2d(x, depth, [3, 3], name="multiRes_Block_{}_inceptionConv3".format(name), padding='same')
    x1 = tf.nn.relu(x1, name="multiRes_Block_{}_inceptionConv3_relu".format(name))
    x2 = tf.layers.conv2d(x1, depth, [3, 3], name="multiRes_Block_{}_inceptionConv5".format(name), padding='same')
    x2 = tf.nn.relu(x2, name="multiRes_Block_{}_inceptionConv5_relu".format(name))
    x3 = tf.layers.conv2d(x2, depth, [3, 3], name="multiRes_Block_{}_inceptionConv7".format(name), padding='same')
    x3 = tf.nn.relu(x3, name="multiRes_Block_{}_inceptionConv7_relu".format(name))
    x4 = tf.concat((x1, x2, x3), axis=3, name="multiRes_Block_{}_concat".format(name))
    x4 = tf.layers.conv2d(x4, depth, [1, 1], name="multiRes_Block_{}_bottleneckOut".format(name), padding='same')
    x4 = tf.nn.relu(x4, name="multiRes_Block_{}_bottleneckOut_relu".format(name))
    x += x4
    x = tf.layers.conv2d(x, depth_out, [1, 1], name="multiRes_Block_{}_bottleneckEnd".format(name), padding='same')
    return tf.nn.relu(x, name="multiRes_Block_{}_End_relu".format(name))

  def _multiResUnet_resPath(self, x, depth, stack_size, name):
    """
    reference https://arxiv.org/abs/1902.04049
    """
    for i in range(stack_size):
      x1 = tf.layers.conv2d(x, depth, [3, 3], name="multiRes_shortcut_{}_stack{}_conv3".format(name, i),
                            padding='same')
      x1 = tf.nn.relu(x1, name="multiresPath_{}_3_relu".format(name))
      x2 = tf.layers.conv2d(x, depth, [1, 1], name="multiRes_shortcut_{}_stack{}_conv1".format(name, i),
                            padding='same')
      x2 = tf.nn.relu(x2, name="multiresPath_{}_1_relu".format(name))
      x = x1 + x2
    return x

  def _multiResUnet_block_light(self, x, depth, depth_out, name):
    """
    reference https://arxiv.org/abs/1902.04049
    """
    x = slim.conv2d(x, depth, [1, 1], scope="multiRes_Block_{}_bottleneckIn".format(name))
    # x1 = slim.conv2d(x, depth_out, [3, 3], scope="multiRes_Block_{}_inceptionConv3".format(name))
    # x2 = slim.conv2d(x1, depth, [3, 3], scope="multiRes_Block_{}_inceptionConv5".format(name))
    # x3 = slim.conv2d(x2, depth, [3, 3], scope="multiRes_Block_{}_inceptionConv7".format(name))
    # x4 = tf.concat((x1, x2), axis=3, name="multiRes_Block_{}_concat".format(name))
    # x4 = slim.conv2d(x4, depth, [1, 1], scope="multiRes_Block_{}_bottleneckOut".format(name))
    # x += x4
    # x = slim.conv2d(x, depth_out, [1, 1], scope="multiRes_Block_{}_bottleneckEnd".format(name))
    return tf.nn.relu(x, name="multiRes_Block_{}_relu".format(name))

  def _concat_augm_feature_maps(self, feature_maps, feature_maps_augm):
    concated_fm = []
    for i in range(4):
      output = self._multiResUnet_block_light(tf.concat([feature_maps[i], feature_maps_augm[i]], axis=3),
                                              depth=128, depth_out=128,
                                              name='bottleneck_merge_before_box_predictor{}'.format(i))
      concated_fm.append(output)
    return concated_fm

  # def _sum_augm_head_feature_maps(self, feature_maps, feature_maps_augm):
  #     summed_fm = []
  #     for i in range(4):
  #         summed_fm.append(feature_maps[i] + tf.layers.conv2d(feature_maps_augm['bels_prediction_head_BEL_F'], filters=128, kernel_size=2, strides=[pow(2, i+1), pow(2, i+1)]))
  #     return summed_fm

  def _sum_augm_feature_maps(self, feature_maps, feature_maps_augm):
    summed_fm = []
    for i in range(4):
      summed_fm.append(feature_maps[i] + tf.layers.conv2d(feature_maps_augm, filters=128, kernel_size=2,
                                                          strides=[pow(2, i + 1), pow(2, i + 1)]))
    return summed_fm

  def _bels_prediction_head(self, singleFrame_inputs, feature_maps_bels, filters, depth):
    singleFrame_inputs = tf.layers.conv2d(singleFrame_inputs, filters=filters, kernel_size=3,
                                          name='bels_prediction_head_for_singleFrame_inputs', padding='same')
    singleFrame_inputs = tf.nn.relu(singleFrame_inputs, name='bels_prediction_head_for_singleFrame_inputs_relu')
    feature_maps_bels_sum = singleFrame_inputs \
                            + tf.layers.conv2d_transpose(feature_maps_bels[0], filters=filters, kernel_size=1,
                                                         strides=[2, 2], padding='same') \
                            + tf.layers.conv2d_transpose(feature_maps_bels[1], filters=filters, strides=[4, 4],
                                                         kernel_size=1, padding='same') \
                            + tf.layers.conv2d_transpose(feature_maps_bels[2], filters=filters, strides=[8, 8],
                                                         kernel_size=1, padding='same') \
                            + tf.layers.conv2d_transpose(feature_maps_bels[3], filters=filters, strides=[16, 16],
                                                         kernel_size=1, padding='same')
    for i in range(depth):
      feature_maps_bels_sum = tf.layers.conv2d(feature_maps_bels_sum, filters=filters, kernel_size=3,
                                               activation='relu',
                                               name='bels_prediction_head_conv3x3_{}'.format(i), padding='same')
    feature_maps_bels_sum = tf.layers.conv2d(feature_maps_bels_sum, filters=3, kernel_size=1, activation='softmax',
                                             name='bels_prediction_head_softmax', padding='same')
    pred_bel_F = tf.expand_dims(feature_maps_bels_sum[:, :, :, 0], axis=3)
    pred_bel_O = tf.expand_dims(feature_maps_bels_sum[:, :, :, 1], axis=3)
    pred_bel_U = tf.expand_dims(feature_maps_bels_sum[:, :, :, 2], axis=3)
    predictions = {
      'bels_prediction_head_BEL_O': [],
      'bels_prediction_head_BEL_F': [],
      'bels_prediction_head_BEL_U': [],
    }
    predictions['bels_prediction_head_BEL_O'] = pred_bel_O
    predictions['bels_prediction_head_BEL_F'] = pred_bel_F
    predictions['bels_prediction_head_BEL_U'] = pred_bel_U
    return predictions



  def predict(self, preprocessed_inputs, true_image_shapes):  # features todo sep24 hestitate to use
    """Predicts unpostprocessed tensors from input tensor.

    This function takes an input batch of images and runs it through the forward
    pass of the network to yield unpostprocessesed predictions.

    A side effect of calling the predict method is that self._anchors is
    populated with a box_list.BoxList of anchors.  These anchors must be
    constructed before the postprocess or loss functions can be called.

    Args:
      preprocessed_inputs: a [batch, height, width, channels] image tensor.
      true_image_shapes: int32 tensor of shape [batch, 3] where each row is
        of the form [height, width, channels] indicating the shapes
        of true images in the resized images, as resized images can be padded
        with zeros.

    Returns:
      prediction_dict: a dictionary holding "raw" prediction tensors:
        1) preprocessed_inputs: the [batch, height, width, channels] image
          tensor.
        2) box_encodings: 4-D float tensor of shape [batch_size, num_anchors,
          box_code_dimension] containing predicted boxes.
        3) class_predictions_with_background: 3-D float tensor of shape
          [batch_size, num_anchors, num_classes+1] containing class predictions
          (logits) for each of the anchors.  Note that this tensor *includes*
          background class predictions (at class index 0).
        4) feature_maps: a list of tensors where the ith tensor has shape
          [batch, height_i, width_i, depth_i].
        5) anchors: 2-D float tensor of shape [num_anchors, 4] containing
          the generated anchors in normalized coordinates.
    """

    prepr_inputs_det = tf.expand_dims(preprocessed_inputs[0, :, :, 0], axis=-1)
    prepr_inputs_obs = tf.expand_dims(preprocessed_inputs[0, :, :, 1], axis=-1)
    prepr_inputs_occ = tf.expand_dims(preprocessed_inputs[0, :, :, 2], axis=-1)
    prepr_inputs_int = tf.expand_dims(preprocessed_inputs[0, :, :, 3], axis=-1)
    prepr_inputs_zmin = tf.expand_dims(preprocessed_inputs[0, :, :, 4], axis=-1)
    prepr_inputs_zmax = tf.expand_dims(preprocessed_inputs[0, :, :, 5], axis=-1)

    ##  SEQUENTIAL MODEL BEFORE THE FPN ##
    print("\n-------------------------------------singleFrame_input:")
    singleFrame_input = tf.identity(preprocessed_inputs, name='identity_singleFrame')
    print(singleFrame_input)

    print("\n-------------------------------------predictor_augm_dict:")
    predictor_augm_dict = self._augm_predictor.predict(None, singleFrame_input)
    print(predictor_augm_dict)

    print("\n-------------------------------------preprocessed_inputs:")
    preprocessed_inputs = self._concat_augm_preprocessed_inputs(predictor_augm_dict, singleFrame_input)
    print(preprocessed_inputs)

    if self._inplace_batchnorm_update:
      batchnorm_updates_collections = None
    else:
      batchnorm_updates_collections = tf.GraphKeys.UPDATE_OPS
    if self._feature_extractor.is_keras_model:
      feature_maps = self._feature_extractor(preprocessed_inputs)
    else:
      with slim.arg_scope([slim.batch_norm],
                          is_training=(self._is_training and
                                       not self._freeze_batchnorm),
                          updates_collections=batchnorm_updates_collections):
        with tf.variable_scope(None, self._extract_features_scope,
                               [preprocessed_inputs]):
          # feature_maps, feature_maps_bels = self._feature_extractor.extract_features_shared_encoder_for_beliefs(preprocessed_inputs)
          feature_maps = self._feature_extractor.extract_features(preprocessed_inputs)

    self._add_histogram_summaries()
    feature_map_spatial_dims = self._get_feature_map_spatial_dims(
      feature_maps)
    image_shape = shape_utils.combined_static_and_dynamic_shape(
      preprocessed_inputs)
    self._anchors = box_list_ops.concatenate(
      self._anchor_generator.generate(
        feature_map_spatial_dims,
        im_height=image_shape[1],
        im_width=image_shape[2]))

    # Grid Maps Augmentation
    # predictor_augm_dict = self._augm_predictor.predict(feature_maps_augm, preprocessed_inputs)
    # predictor_augm_dict_stop = tf.stop_gradient(predictor_augm_dict)

    ##   USE THE AUGMENTATION MAPS TO HELP THE OBJECT DETECTION
    # feature_maps = self._conv_augm_pyramid(predictor_augm_dict, feature_maps, conv_feature_maps_num=8)
    # feature_maps = self._concat_augm_pyramid(predictor_augm_dict, feature_maps)

    use_multi_level_augm_predict_head = False

    if use_multi_level_augm_predict_head is True:
      bels_htc_prediction_dict = self._bels_prediction_head(preprocessed_inputs, feature_maps_bels, filters=16,
                                                            depth=2)

      predictions_dict = {
        'preprocessed_inputs': preprocessed_inputs,
        'feature_maps': feature_maps,
        'bels_prediction_head_BEL_O': bels_htc_prediction_dict['bels_prediction_head_BEL_O'],
        'bels_prediction_head_BEL_F': bels_htc_prediction_dict['bels_prediction_head_BEL_F'],
        'bels_prediction_head_BEL_U': bels_htc_prediction_dict['bels_prediction_head_BEL_U'],
        'anchors': self._anchors.get()
      }
    else:

      predictions_dict = {
        'preprocessed_inputs': preprocessed_inputs,
        'feature_maps': feature_maps,
        'anchors': self._anchors.get()
      }

    '''------------------------------------------Visulization for Tensorboard----------------------------------------'''
    pred_z_min_detections = predictor_augm_dict['z_min_detections_prediction']
    pred_detections_drivingCorridor = predictor_augm_dict['detections_drivingCorridor_prediction']
    pred_z_max_detections = predictor_augm_dict['z_max_detections_prediction']
    pred_z_min_observations = predictor_augm_dict['z_min_observations_prediction']
    pred_intensity = predictor_augm_dict['intensity_prediction']

    label_z_min_detections = self.groundtruth_lists(fields.InputDataFields.groundtruth_z_min_detections)
    label_detections_drivingCorridor = self.groundtruth_lists(
      fields.InputDataFields.groundtruth_detections_drivingCorridor)
    label_z_max_detections = self.groundtruth_lists(fields.InputDataFields.groundtruth_z_max_detections)
    label_z_min_observations = self.groundtruth_lists(fields.InputDataFields.groundtruth_z_min_observations)
    label_intensity = self.groundtruth_lists(fields.InputDataFields.groundtruth_intensity)

    z_max_detections = tf.expand_dims(tf.concat(
      (prepr_inputs_zmax, pred_z_max_detections[0, :, :, :], tf.cast(label_z_max_detections[0], dtype=float)),
      axis=1), 0)
    z_min_detections = tf.expand_dims(tf.concat(
      (prepr_inputs_zmin, pred_z_min_detections[0, :, :, :], tf.cast(label_z_min_detections[0], dtype=float)),
      axis=1), 0)
    detections_drivingCorridor = tf.expand_dims(
      tf.concat((prepr_inputs_det, pred_detections_drivingCorridor[0, :, :, :],
                 tf.cast(label_detections_drivingCorridor[0], dtype=float)), axis=1), 0)
    intensity = tf.expand_dims(tf.concat(
      (prepr_inputs_int, pred_intensity[0, :, :, :], tf.cast(label_intensity[0], dtype=float)), axis=1), 0)
    z_min_observations = tf.expand_dims(tf.concat(
      (prepr_inputs_obs, prepr_inputs_occ, pred_z_min_observations[0, :, :, :],
       tf.cast(label_z_min_observations[0], dtype=float)), axis=1), 0)

    tf.summary.image("zMin_l_input__m_pred__r_target", z_min_detections, family="final_inputs_for_OD")
    tf.summary.image("zMax__l_input__m_pred__r_target", z_max_detections, family="final_inputs_for_OD")
    tf.summary.image("det__l_input__m_pred__r_target", detections_drivingCorridor, family="final_inputs_for_OD")
    tf.summary.image("obs_inputs__l_s_obs__m_s_occ__r_f_obsZMin", z_min_observations, family="final_inputs_for_OD")
    tf.summary.image("int__l_input__m_pred__r_target", intensity, family="final_inputs_for_OD")

    pred_bel_U = predictor_augm_dict['belief_U_prediction']
    pred_bel_F = predictor_augm_dict['belief_F_prediction']
    pred_bel_O = predictor_augm_dict['belief_O_prediction']

    label_bel_U_list = self.groundtruth_lists(fields.InputDataFields.groundtruth_bel_U)
    label_bel_F_list = self.groundtruth_lists(fields.InputDataFields.groundtruth_bel_F)
    label_bel_O_list = self.groundtruth_lists(fields.InputDataFields.groundtruth_bel_O)

    label_bel_U = tf.expand_dims(tf.cast(label_bel_U_list[0], dtype=float), axis=0)
    label_bel_O = tf.expand_dims(tf.cast(label_bel_O_list[0], dtype=float), axis=0)
    label_bel_F = tf.expand_dims(tf.cast(label_bel_F_list[0], dtype=float), axis=0)

    label_bel_U = label_bel_U / 255.
    label_bel_O = label_bel_O / 255.
    label_bel_F = label_bel_F / 255.

    if use_multi_level_augm_predict_head is True:
      bels_prediction_head_BEL_O = predictions_dict['bels_prediction_head_BEL_O']
      bels_prediction_head_BEL_F = predictions_dict['bels_prediction_head_BEL_F']
      bels_prediction_head_BEL_U = predictions_dict['bels_prediction_head_BEL_U']

      #   these bels are in (0,1)
      bel_o = tf.expand_dims(tf.concat(
        (pred_bel_O[0, :, :, :], bels_prediction_head_BEL_O[0, :, :, :], tf.cast(label_bel_O[0], dtype=float)),
        axis=1), 0)
      bel_f = tf.expand_dims(tf.concat(
        (pred_bel_F[0, :, :, :], bels_prediction_head_BEL_F[0, :, :, :], tf.cast(label_bel_F[0], dtype=float)),
        axis=1), 0)
      bel_u = tf.expand_dims(tf.concat(
        (pred_bel_U[0, :, :, :], bels_prediction_head_BEL_U[0, :, :, :], tf.cast(label_bel_U[0], dtype=float)),
        axis=1), 0)

      tf.summary.image('bel_O_leftPredSeq_midPredHead_rightLabel', bel_o,
                       family="final_inputs_for_OD_headWatcher")
      tf.summary.image('bel_F_leftPredSeq_midPredHead_rightLabel', bel_f,
                       family="final_inputs_for_OD_headWatcher")
      tf.summary.image('bel_U_leftPredSeq_midPredHead_rightLabel', bel_u,
                       family="final_inputs_for_OD_headWatcher")
    else:
      bel_o = tf.expand_dims(tf.concat(
        (pred_bel_O[0, :, :, :], tf.cast(label_bel_O[0], dtype=float)),
        axis=1), 0)
      bel_f = tf.expand_dims(tf.concat(
        (pred_bel_F[0, :, :, :], tf.cast(label_bel_F[0], dtype=float)),
        axis=1), 0)
      bel_u = tf.expand_dims(tf.concat(
        (pred_bel_U[0, :, :, :], tf.cast(label_bel_U[0], dtype=float)),
        axis=1), 0)
      tf.summary.image('bel_O_leftPredSeq_rightLabel', bel_o,
                       family="final_inputs_for_OD")
      tf.summary.image('bel_F_leftPredSeq_midPredHead_rightLabel', bel_f,
                       family="final_inputs_for_OD")
      tf.summary.image('bel_U_leftPredSeq_midPredHead_rightLabel', bel_u,
                       family="final_inputs_for_OD")
    '''-----------------------------------------------------------------------------------------------------'''

    ##   concate the fm from augmentation branch with the ones from detection branch   ##
    if use_multi_level_augm_predict_head is True:
      feature_maps = self._sum_augm_feature_maps(feature_maps, bels_prediction_head_BEL_F)
    else:
      feature_maps = self._sum_augm_feature_maps(feature_maps, pred_bel_F)
      # # feature_maps = multiResUnet_block(feature_maps, 128, name="bottleneck_before_box_predictor")

    ##  Box Predictor
    if self._box_predictor.is_keras_model:
      predictor_results_dict = self._box_predictor(feature_maps)
    else:
      with slim.arg_scope([slim.batch_norm],
                          is_training=(self._is_training and
                                       not self._freeze_batchnorm),
                          updates_collections=batchnorm_updates_collections):
        predictor_results_dict = self._box_predictor.predict(
          feature_maps, self._anchor_generator.num_anchors_per_location())

    # augmentation
    for prediction_augm_key, prediction_augm_list in iter(predictor_augm_dict.items()):
      predictions_dict[prediction_augm_key] = prediction_augm_list

    for prediction_key, prediction_list in iter(predictor_results_dict.items()):
      prediction = tf.concat(prediction_list, axis=1)
      if (prediction_key == 'box_3d_encodings' and prediction.shape.ndims == 6 and
        prediction.shape[2] == 1):
        prediction = tf.squeeze(prediction, axis=2)
      predictions_dict[prediction_key] = prediction
    self._batched_prediction_tensor_names = [x for x in predictions_dict
                                             if x != 'anchors']

    return predictions_dict

  def _get_feature_map_spatial_dims(self, feature_maps):
    """Return list of spatial dimensions for each feature map in a list.

    Args:
      feature_maps: a list of tensors where the ith tensor has shape
          [batch, height_i, width_i, depth_i].

    Returns:
      a list of pairs (height, width) for each feature map in feature_maps
    """
    feature_map_shapes = [
      shape_utils.combined_static_and_dynamic_shape(
        feature_map) for feature_map in feature_maps
    ]
    return [(shape[1], shape[2]) for shape in feature_map_shapes]

  def postprocess(self, prediction_dict, true_image_shapes):
    """Converts prediction tensors to final detections.

    This function converts raw predictions tensors to final detection results by
    slicing off the background class, decoding box predictions and applying
    non max suppression and clipping to the image window.

    See base class for output format conventions.  Note also that by default,
    scores are to be interpreted as logits, but if a score_conversion_fn is
    used, then scores are remapped (and may thus have a different
    interpretation).

    Args:
      prediction_dict: a dictionary holding prediction tensors with
        1) preprocessed_inputs: a [batch, height, width, channels] image
          tensor.
        2) box_encodings: 3-D float tensor of shape [batch_size, num_anchors,
          box_code_dimension] containing predicted boxes.
        3) class_predictions_with_background: 3-D float tensor of shape
          [batch_size, num_anchors, num_classes+1] containing class predictions
          (logits) for each of the anchors.  Note that this tensor *includes*
          background class predictions.
        4) mask_predictions: (optional) a 5-D float tensor of shape
          [batch_size, num_anchors, q, mask_height, mask_width]. `q` can be
          either number of classes or 1 depending on whether a separate mask is
          predicted per class.
      true_image_shapes: int32 tensor of shape [batch, 3] where each row is
        of the form [height, width, channels] indicating the shapes
        of true images in the resized images, as resized images can be padded
        with zeros. Or None, if the clip window should cover the full image.

    Returns:
      detections: a dictionary containing the following fields
        detection_boxes: [batch, max_detections, 4] tensor with post-processed
          detection boxes.
        detection_scores: [batch, max_detections] tensor with scalar scores for
          post-processed detection boxes.
        detection_multiclass_scores: [batch, max_detections,
          num_classes_with_background] tensor with class score distribution for
          post-processed detection boxes including background class if any.
        detection_classes: [batch, max_detections] tensor with classes for
          post-processed detection classes.
        detection_keypoints: [batch, max_detections, num_keypoints, 2] (if
          encoded in the prediction_dict 'box_encodings')
        detection_masks: [batch_size, max_detections, mask_height, mask_width]
          (optional)
        num_detections: [batch]
        raw_detection_boxes: [batch, total_detections, 4] tensor with decoded
          detection boxes before Non-Max Suppression.
        raw_detection_score: [batch, total_detections,
          num_classes_with_background] tensor of multi-class scores for raw
          detection boxes.
    Raises:
      ValueError: if prediction_dict does not contain `box_encodings` or
        `class_predictions_with_background` fields.
    """
    if ('box_3d_encodings' not in prediction_dict or
      'class_predictions_with_background' not in prediction_dict):
      raise ValueError('prediction_dict does not contain expected entries.')
    if 'anchors' not in prediction_dict:
      prediction_dict['anchors'] = self.anchors.get()
    with tf.name_scope('Postprocessor'):
      preprocessed_images = prediction_dict['preprocessed_inputs']
      box_3d_encodings = prediction_dict['box_3d_encodings']
      box_3d_encodings = tf.identity(box_3d_encodings, 'raw_box_3d_encodings')
      class_predictions_with_background = (
        prediction_dict['class_predictions_with_background'])
      detection_boxes_3d = self._batch_decode_3d(box_3d_encodings, prediction_dict[
        'anchors'])  # todo sep24 added, prediction_dict['anchors']
      detection_boxes = tf.identity(detection_boxes_3d, 'box_3d_to_convert')
      boxes_shape = shape_utils.combined_static_and_dynamic_shape(detection_boxes)
      detection_boxes = tf.reshape(detection_boxes, [-1, 6])
      detection_boxes = box_list.Box3dList(detection_boxes)
      detection_boxes = detection_boxes.convert_to_boxlist()
      detection_boxes = tf.reshape(detection_boxes.get(), [boxes_shape[0], boxes_shape[1], 4])
      detection_boxes = tf.expand_dims(detection_boxes, axis=2)
      detection_boxes_3d = tf.identity(detection_boxes_3d, 'raw_box_3d_locations')
      detection_boxes_3d = tf.expand_dims(detection_boxes_3d, axis=2)

      detection_scores_with_background = self._score_conversion_fn(
        class_predictions_with_background)
      detection_scores = tf.identity(detection_scores_with_background,
                                     'raw_box_scores')
      if self._add_background_class or self._explicit_background_class:
        detection_scores = tf.slice(detection_scores, [0, 0, 1], [-1, -1, -1])
      additional_fields = None

      batch_size = (
        shape_utils.combined_static_and_dynamic_shape(preprocessed_images)[0])

      if 'feature_maps' in prediction_dict:
        feature_map_list = []
        for feature_map in prediction_dict['feature_maps']:
          feature_map_list.append(tf.reshape(feature_map, [batch_size, -1]))
        box_features = tf.concat(feature_map_list, 1)
        box_features = tf.identity(box_features, 'raw_box_features')
      additional_fields = {
        'multiclass_scores': detection_scores_with_background
      }
      # if detection_keypoints is not None:
      #   detection_keypoints = tf.identity(
      #       detection_keypoints, 'raw_keypoint_locations')
      #   additional_fields[fields.BoxListFields.keypoints] = detection_keypoints
      (nmsed_boxes, nmsed_boxes_3d, nmsed_scores, nmsed_classes,
       nmsed_additional_fields, num_detections) = self._non_max_suppression_fn(
        detection_boxes,
        detection_boxes_3d,
        detection_scores,
        clip_window=self._compute_clip_window(preprocessed_images,
                                              true_image_shapes),
        additional_fields=additional_fields)
      detection_dict = {
        fields.DetectionResultFields.detection_boxes:
          nmsed_boxes,
        fields.DetectionResultFields.detection_boxes_3d:
          nmsed_boxes_3d,
        fields.DetectionResultFields.detection_scores:
          nmsed_scores,
        fields.DetectionResultFields.detection_classes:
          nmsed_classes,
        fields.DetectionResultFields.detection_multiclass_scores:
          nmsed_additional_fields['multiclass_scores'],
        fields.DetectionResultFields.num_detections:
          tf.cast(num_detections, dtype=tf.float32),
        fields.DetectionResultFields.raw_detection_boxes:
          tf.squeeze(detection_boxes, axis=2),
        fields.DetectionResultFields.raw_detection_scores:
          detection_scores_with_background
      }
      for feat_idx in range(1, 5):
        if 'feature_maps' in prediction_dict:
          detection_dict[
            fields.DetectionResultFields.feature_map + '_level_{}'.format(feat_idx)] = \
            prediction_dict['feature_maps'][feat_idx - 1]

      detection_dict[
        fields.DetectionResultFields.detections_drivingCorridor_prediction] = prediction_dict[
        'detections_drivingCorridor_prediction']
      detection_dict[
        fields.DetectionResultFields.z_min_detections_prediction] = prediction_dict[
        'z_min_detections_prediction']
      detection_dict[
        fields.DetectionResultFields.z_max_detections_prediction] = prediction_dict[
        'z_max_detections_prediction']
      detection_dict[
        fields.DetectionResultFields.z_min_observations_prediction] = prediction_dict[
        'z_min_observations_prediction']
      detection_dict[
        fields.DetectionResultFields.intensity_prediction] = prediction_dict['intensity_prediction']
      detection_dict[
        fields.DetectionResultFields.belief_F_prediction] = prediction_dict['belief_F_prediction']
      detection_dict[
        fields.DetectionResultFields.belief_O_prediction] = prediction_dict['belief_O_prediction']
      detection_dict[
        fields.DetectionResultFields.belief_U_prediction] = prediction_dict['belief_U_prediction']

      return detection_dict

  def boxes2mask(self, gt_boxes, img_shape):

    def make_box_representation(x_min, x_max, y_min, y_max, mask):
      outer_box_width = mask.shape[2]

      x_max = tf.Print(x_max,[x_max],message='x_max')
      x_min = tf.Print(x_min,[x_min],message='x_min')
      y_max = tf.Print(x_max,[y_max],message='y_max')
      y_min = tf.Print(y_min,[y_min],message='y_min')
      # print("x_max")
      # print(x_max)
      x_min = tf.cast(x_min, dtype=tf.int16)

      # print("x_min")
      # print(x_min)
      x_max = tf.cast(x_max, dtype=tf.int16)
      y_min = tf.cast(y_min, dtype=tf.int16)
      y_max = tf.cast(y_max, dtype=tf.int16)

      # x_min = x_min.eval()
      #
      # print("x_min")
      # print(x_min)
      # x_max = x_max.eval()
      # y_min = y_min.eval()
      # y_max = y_max.eval()


      x, y = x_max - x_min, y_max - y_min


      # print(y)
      # print(outer_box_width)
      outer_box_width = tf.constant(outer_box_width, dtype=tf.int16)

      print("outerboxwidth")
      print(outer_box_width)
      x = tf.Print(x,[x],message='x is')


      inner_box = tf.ones((1, y, x, 1))

      left_padding = tf.zeros((1, y, x_min, 1))
      right_padding = tf.zeros((1, y, (outer_box_width - x_max),1))

      mask = tf.concat([left_padding, inner_box, right_padding], axis=2)

      top_padding = tf.zeros((1, y_min, outer_box_width, 1))
      bottom_padding = tf.zeros((1, outer_box_width - y_max, outer_box_width, 1))

      mask = tf.concat([top_padding, mask, bottom_padding], axis=1)

      return mask

    boxes_mask = tf.zeros([img_shape[0], img_shape[1], img_shape[2], 1], tf.float32)
    print("gt_boxes:")
    print(gt_boxes)
    for img_num in range(gt_boxes.shape[0]):
      box_mask = tf.zeros([1, img_shape[1], img_shape[2], 1], tf.float32)
      for i in range(gt_boxes.shape[1]):
        # box_mask = make_box_representation(gt_boxes[img_num, i, 1], gt_boxes[img_num, i, 3], gt_boxes[img_num, i, 0], gt_boxes[img_num, i, 2], box_mask)
        gt_boxes[img_num, i, 1]
        # boxes_mask[]
        print("box_mask:")
        print(box_mask)
        boxes_mask[img_num,:,:,:] = box_mask + boxes_mask[img_num,:,:,:]
    return boxes_mask

  def loss(self, prediction_dict, true_image_shapes, category_index, scope=None):
    """Compute scalar loss tensors with respect to provided groundtruth.

    Calling this function requires that groundtruth tensors have been
    provided via the provide_groundtruth function.

    Args:
      prediction_dict: a dictionary holding prediction tensors with
        1) box_encodings: 3-D float tensor of shape [batch_size, num_anchors,
          box_code_dimension] containing predicted boxes.
        2) class_predictions_with_background: 3-D float tensor of shape
          [batch_size, num_anchors, num_classes+1] containing class predictions
          (logits) for each of the anchors. Note that this tensor *includes*
          background class predictions.
        3) augmentation loss
      true_image_shapes: int32 tensor of shape [batch, 3] where each row is
        of the form [height, width, channels] indicating the shapes
        of true images in the resized images, as resized images can be padded
        with zeros.
      scope: Optional scope name.

    Returns:
      a dictionary mapping loss keys (`localization_loss` and
        `classification_loss`) to scalar tensors representing corresponding loss
        values.
    """
    with tf.name_scope(scope, 'Loss', prediction_dict.values()):
      weights = None
      if self.groundtruth_has_field(fields.BoxListFields.weights):
        weights = self.groundtruth_lists(fields.BoxListFields.weights)
      confidences = None
      if self.groundtruth_has_field(fields.BoxListFields.confidences):
        confidences = self.groundtruth_lists(fields.BoxListFields.confidences)
      (batch_cls_targets, batch_cls_weights, batch_reg_targets_3d,
       batch_reg_weights, batch_match) = self._assign_targets(
        self.groundtruth_lists(fields.BoxListFields.boxes),
        self.groundtruth_lists(fields.BoxListFields.boxes_3d),
        self.groundtruth_lists(fields.BoxListFields.classes),
        weights, confidences)
      match_list = [matcher.Match(match) for match in tf.unstack(batch_match)]
      if self._add_summaries:
        self._summarize_classwise_target_assignment(
          self.groundtruth_lists(fields.BoxListFields.boxes),
          self.groundtruth_lists(fields.BoxListFields.classes),
          match_list, category_index)

      if self._random_example_sampler:
        batch_cls_per_anchor_weights = tf.reduce_mean(
          batch_cls_weights, axis=-1)
        batch_sampled_indicator = tf.cast(
          shape_utils.static_or_dynamic_map_fn(
            self._minibatch_subsample_fn,
            [batch_cls_targets, batch_cls_per_anchor_weights],
            dtype=tf.bool,
            parallel_iterations=self._parallel_iterations,
            back_prop=True), dtype=tf.float32)
        batch_reg_weights = tf.multiply(batch_sampled_indicator,
                                        batch_reg_weights)
        batch_cls_weights = tf.multiply(
          tf.expand_dims(batch_sampled_indicator, -1),
          batch_cls_weights)

      losses_mask = None
      if self.groundtruth_has_field(fields.InputDataFields.is_annotated):
        losses_mask = tf.stack(self.groundtruth_lists(
          fields.InputDataFields.is_annotated))
      location_losses_3d = self._localization_loss(
        prediction_dict['box_3d_encodings'],
        batch_reg_targets_3d,
        ignore_nan_targets=True,
        weights=batch_reg_weights,
        losses_mask=losses_mask)
      cls_losses = self._classification_loss(
        prediction_dict['class_predictions_with_background'],
        batch_cls_targets,
        weights=batch_cls_weights,
        losses_mask=losses_mask)

      if self._expected_loss_weights_fn:
        # Need to compute losses for assigned targets against the
        # unmatched_class_label as well as their assigned targets.
        # simplest thing (but wasteful) is just to calculate all losses
        # twice
        batch_size, num_anchors, num_classes = batch_cls_targets.get_shape()
        unmatched_targets = tf.ones([batch_size, num_anchors, 1
                                     ]) * self._unmatched_class_label

        unmatched_cls_losses = self._classification_loss(
          prediction_dict['class_predictions_with_background'],
          unmatched_targets,
          weights=batch_cls_weights,
          losses_mask=losses_mask)

        if cls_losses.get_shape().ndims == 3:
          batch_size, num_anchors, num_classes = cls_losses.get_shape()
          cls_losses = tf.reshape(cls_losses, [batch_size, -1])
          unmatched_cls_losses = tf.reshape(unmatched_cls_losses,
                                            [batch_size, -1])
          batch_cls_targets = tf.reshape(
            batch_cls_targets, [batch_size, num_anchors * num_classes, -1])
          batch_cls_targets = tf.concat(
            [1 - batch_cls_targets, batch_cls_targets], axis=-1)

          location_losses_3d = tf.tile(location_losses_3d, [1, num_classes])

        foreground_weights, background_weights = (
          self._expected_loss_weights_fn(batch_cls_targets))

        cls_losses = (
          foreground_weights * cls_losses +
          background_weights * unmatched_cls_losses)

        location_losses_3d *= foreground_weights

        classification_loss = tf.reduce_sum(cls_losses)
        localization_loss_3d = tf.reduce_sum(location_losses_3d)
      elif self._hard_example_miner:
        cls_losses = ops.reduce_sum_trailing_dimensions(cls_losses, ndims=2)
        (localization_loss_3d, classification_loss) = self._apply_hard_mining(
          location_losses_3d, cls_losses, prediction_dict, match_list)
        if self._add_summaries:
          self._hard_example_miner.summarize()
      else:
        cls_losses = ops.reduce_sum_trailing_dimensions(cls_losses, ndims=2)
        localization_loss_3d = tf.reduce_sum(location_losses_3d)
        classification_loss = tf.reduce_sum(cls_losses)

      # Optionally normalize by number of positive matches
      normalizer = tf.constant(1.0, dtype=tf.float32)
      if self._normalize_loss_by_num_matches:
        normalizer = tf.maximum(tf.cast(tf.reduce_sum(batch_reg_weights),
                                        dtype=tf.float32),
                                1.0)

      localization_loss_normalizer = normalizer
      if self._normalize_loc_loss_by_codesize:
        localization_loss_normalizer *= self._box_coder.code_size_3d

        loc_loss_weight = self._localization_loss_weight
        cls_loss_weight = self._classification_loss_weight

        localization_loss = tf.multiply((self._localization_loss_weight / localization_loss_normalizer),
                                        localization_loss_3d,
                                        name='localization_loss')
      classification_loss = tf.multiply((self._classification_loss_weight / normalizer),
                                        classification_loss,
                                        name='classification_loss')
      if self._use_uncertainty_weighting_loss:
        log_var_loc = tf.get_variable('log_variance_localization', dtype=tf.float32, trainable=True,
                                      initializer=0.0)
        localization_loss_new = localization_loss * tf.exp(-log_var_loc) + log_var_loc
        tf.summary.scalar('log_var_loc', log_var_loc, family="custom_loss")
        tf.summary.scalar('loc_loss_UNC_WEIGHT', localization_loss_new / localization_loss,
                          family="custom_loss")
        tf.summary.scalar('loc_loss_UNC_WEIGHTcoeff',tf.exp(-log_var_loc),
                          family="custom_loss")
        localization_loss = localization_loss_new

        log_var_cls = tf.get_variable('log_variance_classification', dtype=tf.float32, trainable=True,
                                      initializer=0.0)
        # Boltzmann (Gibbs) distribution for classification, therefore multiply classification weight by 2
        classification_loss_new = classification_loss * tf.exp(-log_var_cls) + log_var_cls
        tf.summary.scalar('log_var_cls', log_var_cls, family="custom_loss")
        tf.summary.scalar('cls_loss_UNC_WEIGHT', classification_loss_new / classification_loss,
                          family="custom_loss")
        tf.summary.scalar('cls_loss_UNC_WEIGHTcoeff',tf.exp(-log_var_cls),
                          family="custom_loss")
        classification_loss = classification_loss_new

      # augmentation
      pred_z_min_detections = prediction_dict['z_min_detections_prediction']
      pred_detections_drivingCorridor = prediction_dict['detections_drivingCorridor_prediction']
      pred_bel_U = prediction_dict['belief_U_prediction']
      pred_z_max_detections = prediction_dict['z_max_detections_prediction']
      pred_z_min_observations = prediction_dict['z_min_observations_prediction']
      pred_bel_F = prediction_dict['belief_F_prediction']
      pred_bel_O = prediction_dict['belief_O_prediction']
      pred_intensity = prediction_dict['intensity_prediction']

      label_z_min_detections = self.groundtruth_lists(fields.InputDataFields.groundtruth_z_min_detections)
      label_detections_drivingCorridor = self.groundtruth_lists(
        fields.InputDataFields.groundtruth_detections_drivingCorridor)
      label_bel_U_list = self.groundtruth_lists(fields.InputDataFields.groundtruth_bel_U)
      label_z_max_detections = self.groundtruth_lists(fields.InputDataFields.groundtruth_z_max_detections)
      label_z_min_observations = self.groundtruth_lists(fields.InputDataFields.groundtruth_z_min_observations)
      label_bel_F_list = self.groundtruth_lists(fields.InputDataFields.groundtruth_bel_F)
      label_bel_O_list = self.groundtruth_lists(fields.InputDataFields.groundtruth_bel_O)
      label_intensity = self.groundtruth_lists(fields.InputDataFields.groundtruth_intensity)
      label_boxes_list = self.groundtruth_lists(fields.BoxListFields.boxes)
      print(label_boxes_list)

      # LOSSES #
      with tf.name_scope("augm_losses_and_weights"):
        label_bel_U = tf.expand_dims(tf.cast(label_bel_U_list[0], dtype=float), axis=0)
        label_bel_O = tf.expand_dims(tf.cast(label_bel_O_list[0], dtype=float), axis=0)
        label_bel_F = tf.expand_dims(tf.cast(label_bel_F_list[0], dtype=float), axis=0)
        label_boxes = tf.expand_dims(label_boxes_list[0], axis=0)

        size_of_batch = len(label_bel_F_list)
        if size_of_batch >= 2:
          for index in range(size_of_batch - 1):
            label_bel_U = tf.concat([label_bel_U,
                                     tf.expand_dims(tf.cast(label_bel_U_list[index + 1], dtype=float),
                                                    axis=0)],
                                    axis=0)
            label_bel_F = tf.concat([label_bel_F,
                                     tf.expand_dims(tf.cast(label_bel_F_list[index + 1], dtype=float),
                                                    axis=0)],
                                    axis=0)
            label_bel_O = tf.concat([label_bel_O,
                                     tf.expand_dims(tf.cast(label_bel_O_list[index + 1], dtype=float),
                                                    axis=0)],
                                    axis=0)
            label_boxes = tf.concat([label_boxes,tf.expand_dims(label_boxes_list[index + 1], axis=0)], axis=0)
        # label_bel_U = tf.squeeze(label_bel_U, axis=0)
        # label_bel_O = tf.squeeze(label_bel_O, axis=0)
        # label_bel_F = tf.squeeze(label_bel_F, axis=0)

        label_bel_U = label_bel_U / 255.
        label_bel_O = label_bel_O / 255.
        label_bel_F = label_bel_F / 255.

        # print("pred_bel_F")
        # print(pred_bel_F)
        # tf.Print(pred_bel_F,[pred_bel_F],'pred_bel_F')
        # print("pred_bel_U")
        # print(pred_bel_U)
        # pred_bel_U = tf.Print(pred_bel_U,[pred_bel_U],message='pred_bel_U')
        #
        # print("label_bel_F")
        # print(label_bel_F)
        # tf.Print(label_bel_F,[label_bel_F],'label_bel_F')
        # print("label_bel_U")
        # print(label_bel_U)
        # label_bel_U = tf.Print(label_bel_U,[label_bel_U],message='label_bel_U')
        # tf.summary.scalar('tf.reduce_min(pred_bel_U)', tf.reduce_min(pred_bel_U), family="watcher")
        # tf.summary.scalar('tf.reduce_min(label_bel_U)',tf.reduce_min(label_bel_U), family="watcher")
        # tf.summary.scalar('tf.reduce_max(pred_bel_U)', tf.reduce_max(pred_bel_U), family="watcher")
        # tf.summary.scalar('tf.reduce_max(label_bel_U)',tf.reduce_max(label_bel_U), family="watcher")
        # label_bel_U = tf.Print(label_bel_U, [tf.reduce_max(label_bel_U)],  message='tf.reduce_max(label_bel_U)')
        # label_bel_U = tf.Print(label_bel_U, [tf.reduce_min(label_bel_U)], message='tf.reduce_min(label_bel_U)')

        # wLC10 = self._my_weights_label_cert(labels, 10.)
        #  weights=wLC10

        # softmax_cross_entropy_BELS = tf.nn.softmax_cross_entropy_with_logits(label_bel_F, pred_bel_F) \
        # * self._factor_loss_fused_bel_F  # todo

        '''-----------------------------------optional head for bel prediction---------------------------------------------------'''
        use_multi_level_augm_predict_head = False
        if use_multi_level_augm_predict_head is True:
          head_pred_bel_F = prediction_dict['bels_prediction_head_BEL_F']
          head_pred_bel_O = prediction_dict['bels_prediction_head_BEL_O']
          head_pred_bel_U = prediction_dict['bels_prediction_head_BEL_U']
          head_augm_loss_belO = self._my_loss_L1(head_pred_bel_O, label_bel_O,
                                                 xBiggerY=2.) * self._factor_loss_fused_bel_O
          head_augm_loss_belF = self._my_loss_L1(head_pred_bel_F, label_bel_F,
                                                 xBiggerY=1.) * self._factor_loss_fused_bel_F
          head_augm_loss_belU = self._my_loss_L1(head_pred_bel_U, label_bel_U,
                                                 xBiggerY=1.) * self._factor_loss_fused_bel_F / 5
          tf.summary.scalar('head_augm_loss_belO_', head_augm_loss_belO, family="custom_loss")
          tf.summary.scalar('head_augm_loss_belF', head_augm_loss_belF, family="custom_loss")
          tf.summary.scalar('head_augm_loss_belU', head_augm_loss_belU, family="custom_loss")
          head_L1_BELS = head_augm_loss_belO + head_augm_loss_belF + head_augm_loss_belU
        '''-----------------------------------------------------------------------------------------------------'''

        augm_loss_belO = self._my_loss_L1(pred_bel_O, label_bel_O, xBiggerY=3.) * self._factor_loss_fused_bel_O
        augm_loss_belF = self._my_loss_L1(pred_bel_F, label_bel_F, xBiggerY=1.) * self._factor_loss_fused_bel_F
        augm_loss_belU = self._my_loss_L1(pred_bel_U, label_bel_U,
                                          xBiggerY=1.) * self._factor_loss_fused_bel_F / 100
        tf.summary.scalar('augm_loss_belO', augm_loss_belO, family="custom_loss")
        tf.summary.scalar('augm_loss_belF', augm_loss_belF, family="custom_loss")
        tf.summary.scalar('augm_loss_belU', augm_loss_belU, family="custom_loss")
        # tf.summary.scalar('augm_metric_falsePositive', augm_loss_belO)
        # tf.summary.scalar('augm_metric_falseNegative', augm_loss_belF)
        L1_BELS = augm_loss_belO + augm_loss_belF + augm_loss_belU

        if self._use_uncertainty_weighting_loss:
          log_var_augm_bels = tf.get_variable('log_variance_augmentation_bels', dtype=tf.float32, trainable=True,
                                         initializer=0.0)
          # L1_BELS_new = 0.5 * L1_BELS * tf.exp(-log_var_augm_bels) + 0.5 * log_var_augm_bels
          L1_BELS_new = L1_BELS * tf.exp(-log_var_augm_bels) + log_var_augm_bels
          tf.summary.scalar('log_var_augm_bels', log_var_augm_bels, family="custom_loss")
          tf.summary.scalar('augm_bels_loss_UNC_WEIGHTcoeff', tf.exp(-log_var_augm_bels), family="custom_loss")
          L1_BELS = L1_BELS_new



        metrics_dict = self._get_my_metric_dict(pred_bel_O, pred_bel_F, pred_bel_U, label_bel_F, label_bel_O,
                                                label_bel_U, L1_BELS)
        self._summarize_grid_maps_augmentation(metrics_dict)

        '''--------------------------------------mask for target data------------------------------------------------------'''
        bel_cert_mask = 1 - label_bel_U  # no offset ( quantisierungsfehler auch verknueftiger )
        bel_o_mask = label_bel_O + 1
        # gt_boxes_mask = 1 + self.boxes2mask(label_boxes, bel_cert_mask.shape) * 2
        gt_boxes_mask = 1

        # bel_cert_mask_img = tf.expand_dims(
        #   tf.concat(((1 - pred_bel_U)[0, :, :, :], tf.cast(bel_cert_mask[0], dtype=float),
        #              bel_o_mask[0, :, :, :], gt_boxes_mask[0, :, :, :], bel_cert_mask * bel_o_mask * gt_boxes_mask[0,:,:,:]), axis=1), 0)
        # tf.summary.image('1-belU___belCertMask___belOMask___gtBoxesMask___result', bel_cert_mask_img)
        bel_cert_mask_img = tf.expand_dims(
          tf.concat(((1 - pred_bel_U)[0, :, :, :], tf.cast(bel_cert_mask[0], dtype=float),
                     bel_o_mask[0, :, :, :], bel_cert_mask[0] * bel_o_mask[0]), axis=1), 0)
        tf.summary.image('1-belU___belCertMask___belOMask___result', bel_cert_mask_img)

        augm_loss_zminObs = self._my_loss_L1(pred_z_min_observations, label_z_min_observations,
                                             weights=bel_o_mask * gt_boxes_mask,  # (bel_cert_mask+0.5)/1.5 rescale
                                             xBiggerY=1.) * self._factor_loss_fused_obs_zmin
        augm_loss_zmaxDet = self._my_loss_L1(pred_z_max_detections, label_z_max_detections,
                                             weights=bel_cert_mask * bel_o_mask * gt_boxes_mask,
                                             xBiggerY=2.) * self._factor_loss_fused_zmax_det
        augm_loss_zminDet = self._my_loss_L1(pred_z_min_detections, label_z_min_detections,
                                             weights=bel_cert_mask * bel_o_mask * gt_boxes_mask,
                                             xBiggerY=2.) * self._factor_loss_fused_zmax_det
        augm_loss_detDC = self._my_loss_L1(pred_detections_drivingCorridor, label_detections_drivingCorridor,
                                           weights=bel_cert_mask * bel_o_mask * gt_boxes_mask,
                                           xBiggerY=10.) * self._factor_loss_fused_zmax_det * 5
        augm_loss_int = self._my_loss_L1(pred_intensity, label_intensity,
                                         weights=bel_cert_mask * bel_o_mask * gt_boxes_mask,
                                         xBiggerY=10.) * self._factor_loss_fused_zmax_det * 5
        tf.summary.scalar('augm_loss_zminObs', augm_loss_zminObs, family="custom_loss")
        tf.summary.scalar('augm_loss_zmaxDet', augm_loss_zmaxDet, family="custom_loss")
        tf.summary.scalar('augm_loss_zminDet', augm_loss_zminDet, family="custom_loss")
        tf.summary.scalar('augm_loss_detDC', augm_loss_detDC, family="custom_loss")
        tf.summary.scalar('augm_loss_int', augm_loss_int, family="custom_loss")
        L1_MAPS = augm_loss_zminObs + augm_loss_zmaxDet + augm_loss_zminDet + augm_loss_detDC + augm_loss_int

        if self._use_uncertainty_weighting_loss:
          log_var_augm_maps = tf.get_variable('log_variance_augmentation_maps', dtype=tf.float32, trainable=True,
                                         initializer=0.0)
          # augm_loss_new = 0.5 * augm_loss * tf.exp(-log_var_augm_bels) + 0.5 * log_var_augm_bels
          L1_MAPS_new = L1_MAPS * tf.exp(-log_var_augm_maps) + log_var_augm_maps
          tf.summary.scalar('log_var_augm_maps', log_var_augm_maps, family="custom_loss")
          tf.summary.scalar('augm_maps_loss_UNC_WEIGHTcoeff', tf.exp(-log_var_augm_maps), family="custom_loss")
          L1_MAPS = L1_MAPS_new

        # L1x2 = self._my_loss_L1_metric(pred_bel_F, label_bel_F, xBiggerY=2.) \
        #        + self._my_loss_L1_metric(pred_bel_O, label_bel_O, xBiggerY=2.)
        # L2 = self._my_loss_L2_metric(pred_bel_F, label_bel_F) + self._my_loss_L2_metric(pred_bel_O, label_bel_O)
        if use_multi_level_augm_predict_head is True:
          augm_combined_loss = head_L1_BELS + L1_BELS + L1_MAPS
        else:
          augm_combined_loss = L1_BELS + L1_MAPS

        tf.summary.scalar('augm_combined_loss', augm_combined_loss)
        # augm_combined_loss = tf.Print(augm_combined_loss, [augm_combined_loss],
        #                               message="augm_combined_loss L1 : ")

      # if self._use_uncertainty_weighting_loss:
      #     log_var_augm = tf.get_variable( name='log_variance_augm', dtype=tf.float32, initializer=0.0)
      #     # log_var_augm = tf.Print(log_var_augm, [log_var_augm],
      #     #                         message='augm_loss WEIGHTED BY UNCERTAINTY KENDAL(log_var_augm)')
      #     augm_loss_weight *= tf.exp(-log_var_augm)
      #     augm_loss_weight = tf.Print(augm_loss_weight, [augm_loss_weight],
      #                             message='augm_loss WEIGHTED BY UNCERTAINTY KENDAL(augm_loss_weight)')

      augm_loss = tf.multiply(self._factor_loss_augm,
                              augm_combined_loss,
                              name='augm_loss')

      # localization_loss = tf.Print(localization_loss, [localization_loss], 'localization loss:')
      # classification_loss = tf.Print(classification_loss, [classification_loss], 'classification loss:')
      # augm_loss = tf.Print(augm_loss, [augm_loss], 'augm_loss:')

      tf.summary.scalar('localization_LOSS', localization_loss, family="custom_loss")
      tf.summary.scalar('classification_LOSS', classification_loss, family="custom_loss")
      tf.summary.scalar('augmentation_LOSS', augm_loss, family="custom_loss")
      loss_dict = {
        'Loss/localization_loss': localization_loss,
        'Loss/classification_loss': classification_loss,
        'Loss/augmentation_loss': augm_loss
      }

      # ####                tensorboard visulization                ####
      # bel_o = tf.expand_dims(tf.concat((pred_bel_O[0, :, :, :], tf.cast(label_bel_O[0], dtype=float)), axis=1), 0)
      # bel_f = tf.expand_dims(tf.concat((pred_bel_F[0, :, :, :], tf.cast(label_bel_F[0], dtype=float)), axis=1), 0)
      # bel_u = tf.expand_dims(tf.concat((pred_bel_U[0, :, :, :], tf.cast(label_bel_U[0], dtype=float)), axis=1), 0)
      #
      # z_min_observations = tf.expand_dims(tf.concat(
      #     (pred_z_min_observations[0, :, :, :], tf.cast(label_z_min_observations[0], dtype=float)), axis=1), 0)
      # z_max_detections = tf.expand_dims(tf.concat(
      #     (pred_z_max_detections[0, :, :, :], tf.cast(label_z_max_detections[0], dtype=float)), axis=1), 0)
      # z_min_detections = tf.expand_dims(tf.concat(
      #     (pred_z_min_detections[0, :, :, :], tf.cast(label_z_min_detections[0], dtype=float)), axis=1), 0)
      # detections_drivingCorridor = tf.expand_dims(tf.concat(
      #     (pred_detections_drivingCorridor[0, :, :, :],
      #         tf.cast(label_detections_drivingCorridor[0], dtype=float)), axis=1), 0)
      #
      # # z_min_observations = tf.squeeze(tf.concat(pred_z_min_observations, label_z_min_observations), axis=1)
      # # z_max_detections = tf.squeeze(tf.concat(pred_z_max_detections, label_z_max_detections), axis=1)
      #
      # # tf.summary.image('preprocessed_Input', preprcossed_input, family="augmentated_maps")
      # tf.summary.image('bel_O_leftPred_rightLabel', bel_o * 255, family="augmentated_maps")
      # tf.summary.image('bel_F_leftPred_rightLabel', bel_f * 255, family="augmentated_maps")
      # tf.summary.image('z_min_observations_leftPred_rightLabel', z_min_observations,
      #                  family="augmentated_maps")
      # tf.summary.image('z_max_detections_leftPred_rightLabel', z_max_detections, family="augmentated_maps")
      # tf.summary.image('bel_U_leftPred_rightLabel', bel_u * 255, family="augmentated_maps")
      # tf.summary.image('detections_drivingCorridor_leftPred_rightLabel', detections_drivingCorridor,
      #                  family="augmentated_maps")
      # tf.summary.image('z_min_detections_leftPred_rightLabel', z_min_detections, family="augmentated_maps")

    return loss_dict

  def _my_weights_label_cert(self, label_bel_F, label_bel_O, factor):
    with tf.variable_scope("weights_label_cert"):
      cert = label_bel_O + label_bel_F
      return (1. / factor) + (1. - (1. / factor)) * cert

  def _my_loss_L1(self, pred, label, weights=None, xBiggerY=1., name=None):
    """L1 loss with optional weight and optional assymmetry."""
    # with tf.name_scope(name or "loss_L1"):
    # pred = tf.cast(pred, tf.float32)
    loss = tf.abs(pred - label) + tf.multiply((xBiggerY - 1.) / (xBiggerY + 1.), (label - pred))
    if weights is not None:
      loss = tf.multiply(loss, weights)
    return tf.reduce_mean(loss)

  def _my_loss_L1_metric(self, pred, label, weights=None, xBiggerY=1., name=None):
    """L1 loss with optional weight and optional assymmetry."""
    # with tf.name_scope(name or "loss_L1"):
    # pred = tf.cast(pred, tf.float32)
    loss = tf.abs(pred - label) + tf.multiply((xBiggerY - 1.) / (xBiggerY + 1.), (pred - label))
    if weights is not None:
      loss = tf.multiply(loss, weights)
    return tf.reduce_mean(loss)

  def _my_loss_L2_metric(self, pred, label, weights=None, name=None):
    """L2 loss with optional weight and optional assymmetry."""
    with tf.name_scope(name or "loss_L2"):
      loss = tf.square(pred - label)
      if weights is not None:
        loss = tf.multiply(loss, weights)
      return tf.reduce_mean(loss)

  def _get_my_metric_dict(self, pred_bel_O, pred_bel_F, pred_bel_U, label_bel_F, label_bel_O, label_bel_U,
                          trained_loss):

    with tf.name_scope("false_occ"):
      false_occ = tf.reduce_mean(tf.cast(tf.greater(pred_bel_O + label_bel_F, 1.), dtype=tf.float32))
    with tf.name_scope("false_free"):
      false_free = tf.reduce_mean(tf.cast(tf.greater(pred_bel_F + label_bel_O, 1.), dtype=tf.float32))

    with tf.name_scope("false_occ_2"):
      false_occ_2 = tf.reduce_mean(
        (pred_bel_O + label_bel_F - 1) * tf.cast(tf.greater(pred_bel_O + label_bel_F, 1.), dtype=tf.float32))
    with tf.name_scope("false_free_2"):
      false_free_2 = tf.reduce_mean(
        (pred_bel_F + label_bel_O - 1) * tf.cast(tf.greater(pred_bel_F + label_bel_O, 1.), dtype=tf.float32))

    with tf.name_scope("self_conflict"):
      self_conflict = tf.reduce_mean(tf.cast(tf.greater(pred_bel_F + pred_bel_O, 1.), dtype=tf.float32))

    with tf.name_scope("rel_uncertainty"):
      pred_uncertainty = tf.reduce_sum(1. - pred_bel_F - pred_bel_O)
      label_uncertainty = tf.reduce_sum(1. - label_bel_F - label_bel_O)
      rel_uncertainty = tf.divide(pred_uncertainty, label_uncertainty)

    with tf.name_scope("uncertainty_system_error"):
      recalculated_pred_uncertainty = 1. - pred_bel_F - pred_bel_O
      diff_recalculated_predicted_U = self._my_loss_L1_metric(recalculated_pred_uncertainty, pred_bel_U)
      recalculated_label_uncertainty = 1. - label_bel_F - label_bel_O
      diff_recalculated_label_U = self._my_loss_L1_metric(recalculated_label_uncertainty, label_bel_U)

    with tf.name_scope("losses"):
      L1 = self._my_loss_L1_metric(pred_bel_F, label_bel_F) + self._my_loss_L1_metric(pred_bel_O, label_bel_O)
      L2 = self._my_loss_L2_metric(pred_bel_F, label_bel_F) + self._my_loss_L2_metric(pred_bel_O, label_bel_O)

    with tf.name_scope("losses_wLC10"):
      wLC10 = self._my_weights_label_cert(label_bel_F, label_bel_O, 10.)

      L1_wLC10 = self._my_loss_L1_metric(pred_bel_F, label_bel_F, weights=wLC10) + self._my_loss_L1_metric(
        pred_bel_O, label_bel_O,
        weights=wLC10)
      L2_wLC10 = self._my_loss_L2_metric(pred_bel_F, label_bel_F, weights=wLC10) + self._my_loss_L2_metric(
        pred_bel_O, label_bel_O,
        weights=wLC10)

    metric_ops = {
      "met_false_occ": false_occ,
      "met_false_free": false_free,
      "met_false_occ_2": false_occ_2,
      "met_false_free_2": false_free_2,
      "met_self_conflict": self_conflict,
      "met_diff_recalculated_label_U": diff_recalculated_label_U,
      "met_diff_recalculated_predicted_U": diff_recalculated_predicted_U,
      "met_rel_uncertainty": rel_uncertainty,
      "met_L1": L1,
      "met_L2": L2,
      "met_L1_wLC10": L1_wLC10,
      "met_L2_wLC10": L2_wLC10,
      "met_trained_loss": trained_loss
    }
    return metric_ops

  def _summarize_grid_maps_augmentation(self, metrics_ops):
    """Creates tensorflow summaries for the grid-map-augmentation."""
    tf.summary.scalar("met_false_occ", metrics_ops["met_false_occ"], family='Augmentation_metics')
    tf.summary.scalar("met_false_free", metrics_ops["met_false_free"], family='Augmentation_metics')
    tf.summary.scalar("met_false_occ_2", metrics_ops["met_false_occ_2"], family='Augmentation_metics')
    tf.summary.scalar("met_false_free_2", metrics_ops["met_false_free_2"], family='Augmentation_metics')
    tf.summary.scalar("met_self_conflict", metrics_ops["met_self_conflict"], family='Augmentation_metics')
    tf.summary.scalar("met_diff_recalculated_label_U", metrics_ops["met_diff_recalculated_label_U"],
                      family='Augmentation_metics')
    tf.summary.scalar("met_diff_recalculated_predicted_U", metrics_ops["met_diff_recalculated_predicted_U"],
                      family='Augmentation_metics')
    tf.summary.scalar("met_rel_uncertainty", metrics_ops["met_rel_uncertainty"], family='Augmentation_metics')
    tf.summary.scalar("met_L1", metrics_ops["met_L1"], family='Augmentation_metics')
    tf.summary.scalar("met_L2", metrics_ops["met_L2"], family='Augmentation_metics')
    tf.summary.scalar("met_L1_wLC10", metrics_ops["met_L1_wLC10"], family='Augmentation_metics')
    tf.summary.scalar("met_L2_wLC10", metrics_ops["met_L2_wLC10"], family='Augmentation_metics')
    tf.summary.scalar("met_trained_loss", metrics_ops["met_trained_loss"], family='Augmentation_metics')

  def _minibatch_subsample_fn(self, inputs):
    """Randomly samples anchors for one image.

    Args:
      inputs: a list of 2 inputs. First one is a tensor of shape [num_anchors,
        num_classes] indicating targets assigned to each anchor. Second one
        is a tensor of shape [num_anchors] indicating the class weight of each
        anchor.

    Returns:
      batch_sampled_indicator: bool tensor of shape [num_anchors] indicating
        whether the anchor should be selected for loss computation.
    """
    cls_targets, cls_weights = inputs
    if self._add_background_class:
      # Set background_class bits to 0 so that the positives_indicator
      # computation would not consider background class.
      background_class = tf.zeros_like(tf.slice(cls_targets, [0, 0], [-1, 1]))
      regular_class = tf.slice(cls_targets, [0, 1], [-1, -1])
      cls_targets = tf.concat([background_class, regular_class], 1)
    positives_indicator = tf.reduce_sum(cls_targets, axis=1)
    return self._random_example_sampler.subsample(
      tf.cast(cls_weights, tf.bool),
      batch_size=None,
      labels=tf.cast(positives_indicator, tf.bool))

  def _summarize_anchor_classification_loss(self, class_ids, cls_losses):
    positive_indices = tf.where(tf.greater(class_ids, 0))
    positive_anchor_cls_loss = tf.squeeze(
      tf.gather(cls_losses, positive_indices), axis=1)
    visualization_utils.add_cdf_image_summary(positive_anchor_cls_loss,
                                              'PositiveAnchorLossCDF')
    negative_indices = tf.where(tf.equal(class_ids, 0))
    negative_anchor_cls_loss = tf.squeeze(
      tf.gather(cls_losses, negative_indices), axis=1)
    visualization_utils.add_cdf_image_summary(negative_anchor_cls_loss,
                                              'NegativeAnchorLossCDF')

  def _assign_targets(self,
                      groundtruth_boxes_list,
                      groundtruth_boxes_3d_list,
                      groundtruth_classes_list,
                      groundtruth_weights_list=None,
                      groundtruth_confidences_list=None):
    """Assign groundtruth targets.

    Adds a background class to each one-hot encoding of groundtruth classes
    and uses target assigner to obtain regression and classification targets.

    Args:
      groundtruth_boxes_list: a list of 2-D tensors of shape [num_boxes, 4]
        containing coordinates of the groundtruth boxes.
          Groundtruth boxes are provided in [y_min, x_min, y_max, x_max]
          format and assumed to be normalized and clipped
          relative to the image window with y_min <= y_max and x_min <= x_max.
      groundtruth_classes_list: a list of 2-D one-hot (or k-hot) tensors of
        shape [num_boxes, num_classes] containing the class targets with the 0th
        index assumed to map to the first non-background class.
      groundtruth_keypoints_list: (optional) a list of 3-D tensors of shape
        [num_boxes, num_keypoints, 2]
      groundtruth_weights_list: A list of 1-D tf.float32 tensors of shape
        [num_boxes] containing weights for groundtruth boxes.
      groundtruth_confidences_list: A list of 2-D tf.float32 tensors of shape
        [num_boxes, num_classes] containing class confidences for
        groundtruth boxes.

    Returns:
      batch_cls_targets: a tensor with shape [batch_size, num_anchors,
        num_classes],
      batch_cls_weights: a tensor with shape [batch_size, num_anchors],
      batch_reg_targets: a tensor with shape [batch_size, num_anchors,
        box_code_dimension]
      batch_reg_weights: a tensor with shape [batch_size, num_anchors],
      match_list: a list of matcher.Match objects encoding the match between
        anchors and groundtruth boxes for each image of the batch,
        with rows of the Match objects corresponding to groundtruth boxes
        and columns corresponding to anchors.
    """
    groundtruth_boxlists = [
      box_list.BoxList(boxes) for boxes in groundtruth_boxes_list
    ]
    groundtruth_boxlists_3d = [
      box_list.Box3dList(boxes) for boxes in groundtruth_boxes_3d_list
    ]
    anchors = self.anchors

    train_using_confidences = (self._is_training and
                               self._use_confidences_as_targets)
    if self._add_background_class:
      groundtruth_classes_with_background_list = [
        tf.pad(one_hot_encoding, [[0, 0], [1, 0]], mode='CONSTANT')
        for one_hot_encoding in groundtruth_classes_list
      ]
      if train_using_confidences:
        groundtruth_confidences_with_background_list = [
          tf.pad(groundtruth_confidences, [[0, 0], [1, 0]], mode='CONSTANT')
          for groundtruth_confidences in groundtruth_confidences_list
        ]
    else:
      groundtruth_classes_with_background_list = groundtruth_classes_list

    if train_using_confidences:
      return target_assigner.batch_assign_confidences(
        self._target_assigner,
        anchors,
        groundtruth_boxlists,
        groundtruth_boxlists_3d,
        groundtruth_confidences_with_background_list,
        groundtruth_weights_list,
        self._unmatched_class_label,
        self._add_background_class,
        self._implicit_example_weight)
    else:
      return target_assigner.batch_assign_targets(
        self._target_assigner,
        anchors,
        groundtruth_boxlists,
        groundtruth_boxlists_3d,
        groundtruth_classes_with_background_list,
        self._unmatched_class_label,
        groundtruth_weights_list)

  def _summarize_target_assignment(self, groundtruth_boxes_list, match_list):
    """Creates tensorflow summaries for the input boxes and anchors.

    This function creates four summaries corresponding to the average
    number (over images in a batch) of (1) groundtruth boxes, (2) anchors
    marked as positive, (3) anchors marked as negative, and (4) anchors marked
    as ignored.

    Args:
      groundtruth_boxes_list: a list of 2-D tensors of shape [num_boxes, 4]
        containing corners of the groundtruth boxes.
      match_list: a list of matcher.Match objects encoding the match between
        anchors and groundtruth boxes for each image of the batch,
        with rows of the Match objects corresponding to groundtruth boxes
        and columns corresponding to anchors.
    """
    avg_num_gt_boxes = tf.reduce_mean(
      tf.cast(
        tf.stack([tf.shape(x)[0] for x in groundtruth_boxes_list]),
        dtype=tf.float32))
    avg_num_matched_gt_boxes = tf.reduce_mean(
      tf.cast(
        tf.stack([match.num_matched_rows() for match in match_list]),
        dtype=tf.float32))
    avg_pos_anchors = tf.reduce_mean(
      tf.cast(
        tf.stack([match.num_matched_columns() for match in match_list]),
        dtype=tf.float32))
    avg_neg_anchors = tf.reduce_mean(
      tf.cast(
        tf.stack([match.num_unmatched_columns() for match in match_list]),
        dtype=tf.float32))
    avg_ignored_anchors = tf.reduce_mean(
      tf.cast(
        tf.stack([match.num_ignored_columns() for match in match_list]),
        dtype=tf.float32))
    # TODO(rathodv): Add a test for these summaries.
    try:
      # TODO(kaftan): Integrate these summaries into the v2 style loops
      with tf.compat.v2.init_scope():
        if tf.compat.v2.executing_eagerly():
          return
    except AttributeError:
      pass

    tf.summary.scalar('AvgNumGroundtruthBoxesPerImage',
                      avg_num_gt_boxes,
                      family='TargetAssignment')
    tf.summary.scalar('AvgNumGroundtruthBoxesMatchedPerImage',
                      avg_num_matched_gt_boxes,
                      family='TargetAssignment')
    tf.summary.scalar('AvgNumPositiveAnchorsPerImage',
                      avg_pos_anchors,
                      family='TargetAssignment')
    tf.summary.scalar('AvgNumNegativeAnchorsPerImage',
                      avg_neg_anchors,
                      family='TargetAssignment')
    tf.summary.scalar('AvgNumIgnoredAnchorsPerImage',
                      avg_ignored_anchors,
                      family='TargetAssignment')

  def _summarize_classwise_target_assignment(self, groundtruth_boxes_list, groundtruth_classes_list,
                                             match_list, categroy_index):
    """Creates tensorflow summaries for the input boxes and anchors.

    This function creates four summaries corresponding to the average
    number (over images in a batch) of (1) groundtruth boxes, (2) anchors
    marked as positive, (3) anchors marked as negative, and (4) anchors marked
    as ignored.

    Args:
      groundtruth_boxes_list: a list of 2-D tensors of shape [num_boxes, 4]
        containing corners of the groundtruth boxes.
      groundtruth_classes_list: a list of 2-D one-hot (or k-hot) tensors of
        shape [num_boxes, num_classes] containing the class targets with the 0th
        index assumed to map to the first non-background class.
      match_list: a list of matcher.Match objects encoding the match between
        anchors and groundtruth boxes for each image of the batch,
        with rows of the Match objects corresponding to groundtruth boxes
        and columns corresponding to anchors.
    """
    avg_num_gt_boxes = tf.reduce_mean(
      tf.cast(
        tf.stack([tf.shape(x)[0] for x in groundtruth_boxes_list]),
        dtype=tf.float32))
    avg_num_matched_gt_boxes = tf.reduce_mean(
      tf.cast(
        tf.stack([match.num_matched_rows() for match in match_list]),
        dtype=tf.float32))
    avg_pos_anchors = tf.reduce_mean(
      tf.cast(
        tf.stack([match.num_matched_columns() for match in match_list]),
        dtype=tf.float32))
    avg_neg_anchors = tf.reduce_mean(
      tf.cast(
        tf.stack([match.num_unmatched_columns() for match in match_list]),
        dtype=tf.float32))
    avg_ignored_anchors = tf.reduce_mean(
      tf.cast(
        tf.stack([match.num_ignored_columns() for match in match_list]),
        dtype=tf.float32))

    summary_number_dict = dict()
    for key, value in categroy_index.items():
      summary_number_dict[value['name']] = []
    summary_dict = dict()

    for match, groundtruth_classes in zip(match_list,
                                          groundtruth_classes_list):
      matched_row_indices = match.matched_row_indices()
      result = tf.gather(groundtruth_classes, matched_row_indices, axis=0)
      result = tf.argmax(result, axis=1)

      for key, value in categroy_index.items():
        number = tf.reduce_sum(tf.cast(tf.equal(result, key - 1), tf.int32))
        summary_number_dict[value['name']].append(number)

    for key, value in summary_number_dict.items():
      summary_dict[key] = tf.stack(value)

    tf.summary.scalar('AvgNumGroundtruthBoxesPerImage',
                      avg_num_gt_boxes,
                      family='TargetAssignment')
    tf.summary.scalar('AvgNumGroundtruthBoxesMatchedPerImage',
                      avg_num_matched_gt_boxes,
                      family='TargetAssignment')
    tf.summary.scalar('AvgNumPositiveAnchorsPerImage',
                      avg_pos_anchors,
                      family='TargetAssignment')
    tf.summary.scalar('AvgNumNegativeAnchorsPerImage',
                      avg_neg_anchors,
                      family='TargetAssignment')
    tf.summary.scalar('AvgNumIgnoredAnchorsPerImage',
                      avg_ignored_anchors,
                      family='TargetAssignment')

    for key, value in summary_dict.items():
      tf.summary.scalar('AvgNumPositive_' + key + '_AnchorsPerImage',
                        tf.reduce_mean(tf.to_float(value)),
                        family='TargetAssignmentPerClass')

  def _add_histogram_summaries(self):
    histogram = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
    for variable in slim.get_model_variables():
      histogram.add(tf.summary.histogram(variable.op.name, variable))

  def _apply_hard_mining(self, location_losses, cls_losses, prediction_dict,
                         match_list):
    """Applies hard mining to anchorwise losses.

    Args:
      location_losses: Float tensor of shape [batch_size, num_anchors]
        representing anchorwise location losses.
      cls_losses: Float tensor of shape [batch_size, num_anchors]
        representing anchorwise classification losses.
      prediction_dict: p a dictionary holding prediction tensors with
        1) box_encodings: 3-D float tensor of shape [batch_size, num_anchors,
          box_code_dimension] containing predicted boxes.
        2) class_predictions_with_background: 3-D float tensor of shape
          [batch_size, num_anchors, num_classes+1] containing class predictions
          (logits) for each of the anchors.  Note that this tensor *includes*
          background class predictions.
        3) anchors: (optional) 2-D float tensor of shape [num_anchors, 4].
      match_list: a list of matcher.Match objects encoding the match between
        anchors and groundtruth boxes for each image of the batch,
        with rows of the Match objects corresponding to groundtruth boxes
        and columns corresponding to anchors.

    Returns:
      mined_location_loss: a float scalar with sum of localization losses from
        selected hard examples.
      mined_cls_loss: a float scalar with sum of classification losses from
        selected hard examples.
    """
    class_predictions = prediction_dict['class_predictions_with_background']
    if self._add_background_class:
      class_predictions = tf.slice(class_predictions, [0, 0, 1], [-1, -1, -1])

    if 'anchors' not in prediction_dict:
      prediction_dict['anchors'] = self.anchors.get()
    decoded_boxes, _ = self._batch_decode(prediction_dict['box_encodings'],
                                          prediction_dict['anchors'])
    decoded_box_tensors_list = tf.unstack(decoded_boxes)
    class_prediction_list = tf.unstack(class_predictions)
    decoded_boxlist_list = []
    for box_location, box_score in zip(decoded_box_tensors_list,
                                       class_prediction_list):
      decoded_boxlist = box_list.BoxList(box_location)
      decoded_boxlist.add_field('scores', box_score)
      decoded_boxlist_list.append(decoded_boxlist)
    return self._hard_example_miner(
      location_losses=location_losses,
      cls_losses=cls_losses,
      decoded_boxlist_list=decoded_boxlist_list,
      match_list=match_list)

  def _batch_decode(self, box_encodings, anchors):
    """Decodes a batch of box encodings with respect to the anchors.

    Args:
      box_encodings: A float32 tensor of shape
        [batch_size, num_anchors, box_code_size] containing box encodings.
      anchors: A tensor of shape [num_anchors, 4].

    Returns:
      decoded_boxes: A float32 tensor of shape
        [batch_size, num_anchors, 4] containing the decoded boxes.
      decoded_keypoints: A float32 tensor of shape
        [batch_size, num_anchors, num_keypoints, 2] containing the decoded
        keypoints if present in the input `box_encodings`, None otherwise.
    """
    combined_shape = shape_utils.combined_static_and_dynamic_shape(
      box_encodings)
    batch_size = combined_shape[0]
    tiled_anchor_boxes = tf.tile(tf.expand_dims(anchors, 0), [batch_size, 1, 1])
    tiled_anchors_boxlist = box_list.BoxList(
      tf.reshape(tiled_anchor_boxes, [-1, 4]))
    decoded_boxes = self._box_coder.decode(
      tf.reshape(box_encodings, [-1, self._box_coder.code_size]),
      tiled_anchors_boxlist)

    decoded_boxes = tf.reshape(decoded_boxes.get(), tf.stack(
      [combined_shape[0], combined_shape[1], 4]))
    return decoded_boxes

  def _batch_decode_3d(self, box_encodings, anchors):  # todo sep24
    """Decodes a batch of box encodings with respect to the anchors.

    Args:
      box_encodings: A float32 tensor of shape
        [batch_size, num_anchors, box_code_size] containing box encodings.

    Returns:
      decoded_boxes: A float32 tensor of shape
        [batch_size, num_anchors, 4] containing the decoded boxes.
      decoded_keypoints: A float32 tensor of shape
        [batch_size, num_anchors, num_keypoints, 2] containing the decoded
        keypoints if present in the input `box_encodings`, None otherwise.
    """
    combined_shape = shape_utils.combined_static_and_dynamic_shape(
      box_encodings)
    batch_size = combined_shape[0]
    tiled_anchor_boxes = tf.tile(
      tf.expand_dims(anchors, 0), [batch_size, 1, 1])
    tiled_anchors_boxlist = box_list.BoxList(
      tf.reshape(tiled_anchor_boxes, [-1, 4]))
    decoded_boxes = self._box_coder.decode_3d(
      tf.reshape(box_encodings, [-1, self._box_coder.code_size_3d]),
      tiled_anchors_boxlist)
    decoded_boxes = tf.reshape(decoded_boxes.get(), tf.stack(
      [combined_shape[0], combined_shape[1], 6]))
    return decoded_boxes

  def regularization_losses(self):
    """Returns a list of regularization losses for this model.

    Returns a list of regularization losses for this model that the estimator
    needs to use during training/optimization.

    Returns:
      A list of regularization loss tensors.
    """
    losses = []
    slim_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    # Copy the slim losses to avoid modifying the collection
    if slim_losses:
      losses.extend(slim_losses)
    if self._box_predictor.is_keras_model:
      losses.extend(self._box_predictor.losses)
    if self._feature_extractor.is_keras_model:
      losses.extend(self._feature_extractor.losses)
    return losses

  def restore_map(self,
                  fine_tune_checkpoint_type='detection',
                  load_all_detection_checkpoint_vars=False):
    """Returns a map of variables to load from a foreign checkpoint.

    See parent class for details.

    Args:
      fine_tune_checkpoint_type: whether to restore from a full detection
        checkpoint (with compatible variable names) or to restore from a
        classification checkpoint for initialization prior to training.
        Valid values: `detection`, `classification`. Default 'detection'.
      load_all_detection_checkpoint_vars: whether to load all variables (when
         `fine_tune_checkpoint_type='detection'`). If False, only variables
         within the appropriate scopes are included. Default False.

    Returns:
      A dict mapping variable names (to load from a checkpoint) to variables in
      the model graph.
    Raises:
      ValueError: if fine_tune_checkpoint_type is neither `classification`
        nor `detection`.
    """
    if fine_tune_checkpoint_type not in ['detection', 'classification']:
      raise ValueError('Not supported fine_tune_checkpoint_type: {}'.format(
        fine_tune_checkpoint_type))

    if fine_tune_checkpoint_type == 'classification':
      return self._feature_extractor.restore_from_classification_checkpoint_fn(
        self._extract_features_scope)

    if fine_tune_checkpoint_type == 'detection':
      variables_to_restore = {}
      for variable in tf.global_variables():
        var_name = variable.op.name
        if load_all_detection_checkpoint_vars:
          variables_to_restore[var_name] = variable
        else:
          if var_name.startswith(self._extract_features_scope):
            variables_to_restore[var_name] = variable

    return variables_to_restore

  def updates(self):
    """Returns a list of update operators for this model.

    Returns a list of update operators for this model that must be executed at
    each training step. The estimator's train op needs to have a control
    dependency on these updates.

    Returns:
      A list of update operators.
    """
    update_ops = []
    slim_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # Copy the slim ops to avoid modifying the collection
    if slim_update_ops:
      update_ops.extend(slim_update_ops)
    if self._box_predictor.is_keras_model:
      update_ops.extend(self._box_predictor.get_updates_for(None))
      update_ops.extend(self._box_predictor.get_updates_for(
        self._box_predictor.inputs))
    if self._feature_extractor.is_keras_model:
      update_ops.extend(self._feature_extractor.get_updates_for(None))
      update_ops.extend(self._feature_extractor.get_updates_for(
        self._feature_extractor.inputs))
    return update_ops
