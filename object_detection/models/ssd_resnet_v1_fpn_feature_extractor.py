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
"""SSD Feature Pyramid Network (FPN) feature extractors based on Resnet v1.

See https://arxiv.org/abs/1708.02002 for details.
"""

import tensorflow as tf

from object_detection.meta_architectures import ssd_meta_arch
from object_detection.models import feature_map_generators
from object_detection.utils import context_manager
from object_detection.utils import ops
from object_detection.utils import shape_utils
from nets import resnet_v1

slim = tf.contrib.slim


class SSDResnetV1FpnFeatureExtractor(ssd_meta_arch.SSDFeatureExtractor):
  """SSD FPN feature extractor based on Resnet v1 architecture."""

  def __init__(self,
               is_training,
               depth_multiplier,
               min_depth,
               pad_to_multiple,
               conv_hyperparams_fn,
               resnet_base_fn,
               resnet_scope_name,
               fpn_scope_name,
               sparsity_type,
               fpn_min_level=3,
               fpn_max_level=7,
               additional_layer_depth=256,
               num_input_channels=3,
               sparse_dense_branch=False,
               channel_means=None,
               # include_root_block=True, # todo sep24
               depthwise_convolution=False,
               use_full_feature_extractor=False,
               use_deconvolution=False,
               max_pool_subsample=False,
               root_downsampling_rate=2,
               recompute_grad=False,
               # min_base_depth=8,
               reuse_weights=None,
               use_explicit_padding=False,
               use_depthwise=False,
               override_base_feature_extractor_hyperparams=False,
               store_non_strided_activations=True):
    """SSD FPN feature extractor based on Resnet v1 architecture.

    Args:
      is_training: whether the network is in training mode.
      depth_multiplier: float depth multiplier for feature extractor.
      min_depth: minimum feature extractor depth.
      pad_to_multiple: the nearest multiple to zero pad the input height and
        width dimensions to.
      conv_hyperparams_fn: A function to construct tf slim arg_scope for conv2d
        and separable_conv2d ops in the layers that are added on top of the
        base feature extractor.
      resnet_base_fn: base resnet network to use.
      resnet_scope_name: scope name under which to construct resnet
      fpn_scope_name: scope name under which to construct the feature pyramid
        network.
      fpn_min_level: the highest resolution feature map to use in FPN. The valid
        values are {2, 3, 4, 5} which map to Resnet blocks {1, 2, 3, 4}
        respectively.
      fpn_max_level: the smallest resolution feature map to construct or use in
        FPN. FPN constructions uses features maps starting from fpn_min_level
        upto the fpn_max_level. In the case that there are not enough feature
        maps in the backbone network, additional feature maps are created by
        applying stride 2 convolutions until we get the desired number of fpn
        levels.
      additional_layer_depth: additional feature map layer channel depth.
      reuse_weights: Whether to reuse variables. Default is None.
      use_explicit_padding: Whether to use explicit padding when extracting
        features. Default is False. UNUSED currently.
      use_depthwise: Whether to use depthwise convolutions. UNUSED currently.
      override_base_feature_extractor_hyperparams: Whether to override
        hyperparameters of the base feature extractor with the one from
        `conv_hyperparams_fn`.

    Raises:
      ValueError: On supplying invalid arguments for unused arguments.
    """
    super(SSDResnetV1FpnFeatureExtractor, self).__init__(
        is_training=is_training,
        depth_multiplier=depth_multiplier,
        min_depth=min_depth,
        pad_to_multiple=pad_to_multiple,
        conv_hyperparams_fn=conv_hyperparams_fn,
        reuse_weights=reuse_weights,
        use_explicit_padding=use_explicit_padding,
        use_depthwise=use_depthwise,
        override_base_feature_extractor_hyperparams=
        override_base_feature_extractor_hyperparams)
    if self._use_explicit_padding is True:
      raise ValueError('Explicit padding is not a valid option.')
    self._resnet_base_fn = resnet_base_fn
    self._resnet_scope_name = resnet_scope_name
    self._fpn_scope_name = fpn_scope_name
    self._fpn_min_level = fpn_min_level
    self._fpn_max_level = fpn_max_level
    self._additional_layer_depth = additional_layer_depth
    if channel_means is None:
        channel_means = [123.68, 116.779, 103.939]
    self._sparsity_type = sparsity_type
    self._sparse_dense_branch = sparse_dense_branch
    self._channel_means = channel_means
    # self._include_root_block = include_root_block
    self._depthwise_convolution = depthwise_convolution
    self._use_full_feature_extractor = use_full_feature_extractor
    self._use_deconvolution = use_deconvolution
    self._max_pool_subsample = max_pool_subsample
    self._root_downsampling_rate = root_downsampling_rate
    self._store_non_strided_activations = store_non_strided_activations
    self._recompute_grad = recompute_grad

  def preprocess(self, resized_inputs):
    """SSD preprocessing.

    VGG style channel mean subtraction as described here:
    https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-mdnge.
    Note that if the number of channels is not equal to 3, the mean subtraction
    will be skipped and the original resized_inputs will be returned.

    Args:
      resized_inputs: a [batch, height, width, channels] float tensor
        representing a batch of images.

    Returns:
      preprocessed_inputs: a [batch, height, width, channels] float tensor
        representing a batch of images.
    """
    return resized_inputs - [[self._channel_means[:]]]

  def _filter_features(self, image_features):
    # TODO(rathodv): Change resnet endpoint to strip scope prefixes instead
    # of munging the scope here.
    filtered_image_features = dict({})
    for key, feature in image_features.items():
      feature_name = key.split('/')[-1]
      if feature_name in ['block1', 'block2', 'block3', 'block4']:
        filtered_image_features[feature_name] = feature
    return filtered_image_features

  def extract_features(self, preprocessed_inputs):
    """Extract features from preprocessed inputs.

    Args:
      preprocessed_inputs: a [batch, height, width, channels] float tensor
        representing a batch of images.

    Returns:
      feature_maps: a list of tensors where the ith tensor has shape
        [batch, height_i, width_i, depth_i]
    """
    preprocessed_inputs = shape_utils.check_min_image_dim(
        129, preprocessed_inputs)

    with tf.variable_scope(
        self._resnet_scope_name, reuse=self._reuse_weights) as scope:
      with slim.arg_scope(resnet_v1.resnet_arg_scope()):
        with (slim.arg_scope(self._conv_hyperparams_fn())
              if self._override_base_feature_extractor_hyperparams else
              context_manager.IdentityContextManager()):
          _, image_features = self._resnet_base_fn(
              inputs=ops.pad_to_multiple(preprocessed_inputs,
                                         self._pad_to_multiple),
              sparsity_type=self._sparsity_type,
              sparse_dense_branch=self._sparse_dense_branch,
              num_classes=None,
              is_training=None,
              global_pool=False,
              output_stride=None,
              # include_root_block=self._include_root_block,
              depthwise_convolution=self._depthwise_convolution,
              max_pool_subsample=self._max_pool_subsample,
              root_downsampling_rate=self._root_downsampling_rate,
              store_non_strided_activations=self._store_non_strided_activations,
              min_base_depth=self._min_depth,
              depth_multiplier=self._depth_multiplier,
              recompute_grad=self._recompute_grad,
              scope=scope)
          image_features = self._filter_features(image_features)
      depth_fn = lambda d: max(int(d * self._depth_multiplier), self._min_depth)
      with slim.arg_scope(self._conv_hyperparams_fn()):
        with tf.variable_scope(self._fpn_scope_name,
                               reuse=self._reuse_weights):

          # base_fpn_max_level = min(self._fpn_max_level, 5)
          base_fpn_max_level = min(self._fpn_max_level, self._fpn_min_level + 3)
          fpn_level_resnet_block_offset = self._fpn_min_level - 1

          feature_block_list = []
          for level in range(self._fpn_min_level, base_fpn_max_level + 1):
            feature_block_list.append('block{}'.format(level - fpn_level_resnet_block_offset))
          if self._use_full_feature_extractor:
            deeper_feature_block_list = []
            for level in range(base_fpn_max_level + 1, 6):
              deeper_feature_block_list.append('block{}'.format(level - fpn_level_resnet_block_offset))
            fpn_features = feature_map_generators.full_fpn_top_down_feature_maps(
                [(key, image_features[key]) for key in feature_block_list],
                [(key, image_features[key]) for key in deeper_feature_block_list],
                depth=self._additional_layer_depth,
                use_deconvolution=self._use_deconvolution)
          else:
            fpn_features = feature_map_generators.fpn_top_down_feature_maps(
                [(key, image_features[key]) for key in feature_block_list],
                depth=depth_fn(self._additional_layer_depth),
                use_deconvolution=self._use_deconvolution)
          feature_maps = []
          for level in range(self._fpn_min_level, base_fpn_max_level + 1):
            feature_maps.append(
                fpn_features['top_down_block{}'.format(level - fpn_level_resnet_block_offset)])
          last_feature_map = fpn_features['top_down_block{}'.format(
              base_fpn_max_level - fpn_level_resnet_block_offset)]
          # Construct coarse features
          for i in range(base_fpn_max_level, self._fpn_max_level):
            last_feature_map = slim.conv2d(
                last_feature_map,
                num_outputs=depth_fn(self._additional_layer_depth),
                kernel_size=[3, 3],
                stride=2,
                padding='SAME',
                scope='bottom_up_block{}'.format(i))
            feature_maps.append(last_feature_map)
    return feature_maps

  def extract_features_shared_encoder_for_augmentation(self, preprocessed_inputs):
      """Extract features from preprocessed inputs.

      Args:
        preprocessed_inputs: a [batch, height, width, channels] float tensor
          representing a batch of images.

      Returns:
        feature_maps: a list of tensors where the ith tensor has shape
          [batch, height_i, width_i, depth_i]
      """
      preprocessed_inputs = shape_utils.check_min_image_dim(
          129, preprocessed_inputs)

      with tf.variable_scope(
              self._resnet_scope_name, reuse=self._reuse_weights) as scope:

          with slim.arg_scope(resnet_v1.resnet_arg_scope()):
              with (slim.arg_scope(self._conv_hyperparams_fn())
              if self._override_base_feature_extractor_hyperparams else
              context_manager.IdentityContextManager()):
                  _, image_features = self._resnet_base_fn(
                      inputs=ops.pad_to_multiple(preprocessed_inputs,
                                                 self._pad_to_multiple),
                      sparsity_type=self._sparsity_type,
                      sparse_dense_branch=self._sparse_dense_branch,
                      num_classes=None,
                      is_training=None,
                      global_pool=False,
                      output_stride=None,
                      # include_root_block=self._include_root_block,
                      depthwise_convolution=self._depthwise_convolution,
                      max_pool_subsample=self._max_pool_subsample,
                      root_downsampling_rate=self._root_downsampling_rate,
                      store_non_strided_activations=self._store_non_strided_activations,
                      min_base_depth=self._min_depth,
                      depth_multiplier=self._depth_multiplier,
                      recompute_grad=self._recompute_grad,
                      scope=scope)
                  image_features = self._filter_features(image_features)
          depth_fn = lambda d: max(int(d * self._depth_multiplier), self._min_depth)


          with slim.arg_scope(self._conv_hyperparams_fn()):
              with tf.variable_scope(self._fpn_scope_name,
                                     reuse=self._reuse_weights):

                  # base_fpn_max_level = min(self._fpn_max_level, 5)
                  base_fpn_max_level = min(self._fpn_max_level, self._fpn_min_level + 3)
                  fpn_level_resnet_block_offset = self._fpn_min_level - 1

                  feature_block_list = []
                  for level in range(self._fpn_min_level, base_fpn_max_level + 1):
                      feature_block_list.append('block{}'.format(level - fpn_level_resnet_block_offset))

                  if self._use_full_feature_extractor:
                      deeper_feature_block_list = []
                      for level in range(base_fpn_max_level + 1, 6):
                          deeper_feature_block_list.append('block{}'.format(level - fpn_level_resnet_block_offset))
                      fpn_features = feature_map_generators.full_fpn_top_down_feature_maps(
                          [(key, image_features[key]) for key in feature_block_list],
                          [(key, image_features[key]) for key in deeper_feature_block_list],
                          depth=self._additional_layer_depth,
                          use_deconvolution=self._use_deconvolution)
                  else:
                      fpn_features, fpn_features_augm = feature_map_generators.fpn_top_down_feature_maps_augmentation_v1(  #   fpn_top_down_feature_maps_augmentation
                          [(key, image_features[key]) for key in feature_block_list],
                          depth=depth_fn(self._additional_layer_depth),
                          use_deconvolution=self._use_deconvolution)
                  feature_maps = []
                  feature_maps_augm = []
                  for level in range(self._fpn_min_level, base_fpn_max_level + 1):
                      feature_maps.append(
                          fpn_features['top_down_block{}'.format(level - fpn_level_resnet_block_offset)])
                      feature_maps_augm.append(
                          fpn_features_augm['top_down_augm_block{}'.format(level - fpn_level_resnet_block_offset)])
                  last_feature_map = fpn_features['top_down_block{}'.format( # not used in ye's
                      base_fpn_max_level - fpn_level_resnet_block_offset)]
                  # Construct coarse features
                  for i in range(base_fpn_max_level, self._fpn_max_level):
                      last_feature_map = slim.conv2d(
                          last_feature_map,
                          num_outputs=depth_fn(self._additional_layer_depth),
                          kernel_size=[3, 3],
                          stride=2,
                          padding='SAME',
                          scope='bottom_up_augm_block{}'.format(i))
                      feature_maps.append(last_feature_map)
      return feature_maps, feature_maps_augm


  def extract_features_shared_encoder_for_beliefs(self, preprocessed_inputs):
        """Extract features from preprocessed inputs.

        Args:
          preprocessed_inputs: a [batch, height, width, channels] float tensor
            representing a batch of images.

        Returns:
          feature_maps: a list of tensors where the ith tensor has shape
            [batch, height_i, width_i, depth_i]
        """
        preprocessed_inputs = shape_utils.check_min_image_dim(
            129, preprocessed_inputs)

        with tf.variable_scope(
                self._resnet_scope_name, reuse=self._reuse_weights) as scope:

            with slim.arg_scope(resnet_v1.resnet_arg_scope()):
                with (slim.arg_scope(self._conv_hyperparams_fn())
                if self._override_base_feature_extractor_hyperparams else
                context_manager.IdentityContextManager()):
                    _, image_features = self._resnet_base_fn(
                        inputs=ops.pad_to_multiple(preprocessed_inputs,
                                                   self._pad_to_multiple),
                        sparsity_type=self._sparsity_type,
                        sparse_dense_branch=self._sparse_dense_branch,
                        num_classes=None,
                        is_training=None,
                        global_pool=False,
                        output_stride=None,
                        # include_root_block=self._include_root_block,
                        depthwise_convolution=self._depthwise_convolution,
                        max_pool_subsample=self._max_pool_subsample,
                        root_downsampling_rate=self._root_downsampling_rate,
                        store_non_strided_activations=self._store_non_strided_activations,
                        min_base_depth=self._min_depth,
                        depth_multiplier=self._depth_multiplier,
                        recompute_grad=self._recompute_grad,
                        scope=scope)
                    image_features = self._filter_features(image_features)
            depth_fn = lambda d: max(int(d * self._depth_multiplier), self._min_depth)

            with slim.arg_scope(self._conv_hyperparams_fn()):
                with tf.variable_scope(self._fpn_scope_name,
                                       reuse=self._reuse_weights):

                    # base_fpn_max_level = min(self._fpn_max_level, 5)
                    base_fpn_max_level = min(self._fpn_max_level, self._fpn_min_level + 3)
                    fpn_level_resnet_block_offset = self._fpn_min_level - 1

                    feature_block_list = []
                    for level in range(self._fpn_min_level, base_fpn_max_level + 1):
                        feature_block_list.append('block{}'.format(level - fpn_level_resnet_block_offset))

                    if self._use_full_feature_extractor:
                        deeper_feature_block_list = []
                        for level in range(base_fpn_max_level + 1, 6):
                            deeper_feature_block_list.append('block{}'.format(level - fpn_level_resnet_block_offset))
                        fpn_features = feature_map_generators.full_fpn_top_down_feature_maps(
                            [(key, image_features[key]) for key in feature_block_list],
                            [(key, image_features[key]) for key in deeper_feature_block_list],
                            depth=self._additional_layer_depth,
                            use_deconvolution=self._use_deconvolution)
                    else:
                        fpn_features, fpn_features_augm = feature_map_generators.fpn_top_down_feature_maps_augmentation_htc(
                            # fpn_top_down_feature_maps_augmentation
                            [(key, image_features[key]) for key in feature_block_list],
                            depth=depth_fn(self._additional_layer_depth),
                            use_deconvolution=self._use_deconvolution)
                    feature_maps = []
                    feature_maps_augm = []
                    for level in range(self._fpn_min_level, base_fpn_max_level + 1):
                        feature_maps.append(
                            fpn_features['top_down_block{}'.format(level - fpn_level_resnet_block_offset)])
                        feature_maps_augm.append(
                            fpn_features_augm['top_down_augm_block{}'.format(level - fpn_level_resnet_block_offset)])
                    last_feature_map = fpn_features['top_down_block{}'.format(  # not used in ye's
                        base_fpn_max_level - fpn_level_resnet_block_offset)]
                    # Construct coarse features
                    for i in range(base_fpn_max_level, self._fpn_max_level):
                        last_feature_map = slim.conv2d(
                            last_feature_map,
                            num_outputs=depth_fn(self._additional_layer_depth),
                            kernel_size=[3, 3],
                            stride=2,
                            padding='SAME',
                            scope='bottom_up_augm_block{}'.format(i))
                        feature_maps.append(last_feature_map)
        return feature_maps, feature_maps_augm


class SSDResnet18V1FpnFeatureExtractor(SSDResnetV1FpnFeatureExtractor):
  """SSD Resnet18 V1 FPN feature extractor."""

  def __init__(self,
               is_training,
               depth_multiplier,
               min_depth,
               pad_to_multiple,
               conv_hyperparams_fn,
               sparsity_type,
               fpn_min_level=3,
               fpn_max_level=7,
               additional_layer_depth=256,
               num_input_channels=3,
               sparse_dense_branch=False,
               channel_means=None,
               # include_root_block=True,
               depthwise_convolution=False,
               use_full_feature_extractor=False,
               max_pool_subsample=False,
               use_deconvolution=False,
               root_downsampling_rate=2,
               store_non_strided_activations=True,
               recompute_grad=False,
               reuse_weights=None,
               use_explicit_padding=False,
               use_depthwise=False,
               override_base_feature_extractor_hyperparams=False):
    """SSD Resnet34 V1 FPN feature extractor based on Resnet v1 architecture.

    Args:
      is_training: whether the network is in training mode.
      depth_multiplier: float depth multiplier for feature extractor.
        UNUSED currently.
      min_depth: minimum feature extractor depth. UNUSED Currently.
      pad_to_multiple: the nearest multiple to zero pad the input height and
        width dimensions to.
      conv_hyperparams_fn: A function to construct tf slim arg_scope for conv2d
        and separable_conv2d ops in the layers that are added on top of the
        base feature extractor.
      fpn_min_level: the minimum level in feature pyramid networks.
      fpn_max_level: the maximum level in feature pyramid networks.
      additional_layer_depth: additional feature map layer channel depth.
      reuse_weights: Whether to reuse variables. Default is None.
      use_explicit_padding: Whether to use explicit padding when extracting
        features. Default is False. UNUSED currently.
      use_depthwise: Whether to use depthwise convolutions. UNUSED currently.
      override_base_feature_extractor_hyperparams: Whether to override
        hyperparameters of the base feature extractor with the one from
        `conv_hyperparams_fn`.
    """
    super(SSDResnet18V1FpnFeatureExtractor, self).__init__(
        is_training,
        depth_multiplier,
        min_depth,
        pad_to_multiple,
        conv_hyperparams_fn,
        resnet_v1.resnet_v1_18,
        'resnet_v1_18',
        'fpn',
        sparsity_type,
        fpn_min_level,
        fpn_max_level,
        additional_layer_depth,
        sparse_dense_branch=sparse_dense_branch,
        num_input_channels=num_input_channels,
        channel_means=channel_means,
        # include_root_block=include_root_block,
        depthwise_convolution=depthwise_convolution,
        use_full_feature_extractor=use_full_feature_extractor,
        use_deconvolution=use_deconvolution,
        max_pool_subsample=max_pool_subsample,
        store_non_strided_activations=store_non_strided_activations,
        root_downsampling_rate=root_downsampling_rate,
        recompute_grad=recompute_grad,
        reuse_weights=reuse_weights,
        use_explicit_padding=use_explicit_padding,
        use_depthwise=use_depthwise,
        override_base_feature_extractor_hyperparams=
        override_base_feature_extractor_hyperparams)

class SSDResnet34V1FpnFeatureExtractor(SSDResnetV1FpnFeatureExtractor):
  """SSD Resnet34 V1 FPN feature extractor."""

  def __init__(self,
               is_training,
               depth_multiplier,
               min_depth,
               pad_to_multiple,
               conv_hyperparams_fn,
               sparsity_type,
               fpn_min_level=3,
               fpn_max_level=7,
               additional_layer_depth=256,
               num_input_channels=3,
               sparse_dense_branch=False,
               channel_means=None,
               include_root_block=True,
               depthwise_convolution=False,
               use_full_feature_extractor=False,
               use_deconvolution=False,
               max_pool_subsample=False,
               root_downsampling_rate=2,
               store_non_strided_activations=True,
               recompute_grad=False,
               reuse_weights=None,
               use_explicit_padding=False,
               use_depthwise=False,
               override_base_feature_extractor_hyperparams=False):
    """SSD Resnet34 V1 FPN feature extractor based on Resnet v1 architecture.

    Args:
      is_training: whether the network is in training mode.
      depth_multiplier: float depth multiplier for feature extractor.
        UNUSED currently.
      min_depth: minimum feature extractor depth. UNUSED Currently.
      pad_to_multiple: the nearest multiple to zero pad the input height and
        width dimensions to.
      conv_hyperparams_fn: A function to construct tf slim arg_scope for conv2d
        and separable_conv2d ops in the layers that are added on top of the
        base feature extractor.
      fpn_min_level: the minimum level in feature pyramid networks.
      fpn_max_level: the maximum level in feature pyramid networks.
      additional_layer_depth: additional feature map layer channel depth.
      reuse_weights: Whether to reuse variables. Default is None.
      use_explicit_padding: Whether to use explicit padding when extracting
        features. Default is False. UNUSED currently.
      use_depthwise: Whether to use depthwise convolutions. UNUSED currently.
      override_base_feature_extractor_hyperparams: Whether to override
        hyperparameters of the base feature extractor with the one from
        `conv_hyperparams_fn`.
    """
    super(SSDResnet34V1FpnFeatureExtractor, self).__init__(
        is_training,
        depth_multiplier,
        min_depth,
        pad_to_multiple,
        conv_hyperparams_fn,
        resnet_v1.resnet_v1_34,
        'resnet_v1_34',
        'fpn',
        sparsity_type,
        fpn_min_level,
        fpn_max_level,
        additional_layer_depth,
        sparse_dense_branch=sparse_dense_branch,
        num_input_channels=num_input_channels,
        channel_means=channel_means,
        # include_root_block=include_root_block,
        depthwise_convolution=depthwise_convolution,
        use_full_feature_extractor=use_full_feature_extractor,
        use_deconvolution=use_deconvolution,
        max_pool_subsample=max_pool_subsample,
        store_non_strided_activations=store_non_strided_activations,
        root_downsampling_rate=root_downsampling_rate,
        recompute_grad=recompute_grad,
        reuse_weights=reuse_weights,
        use_explicit_padding=use_explicit_padding,
        use_depthwise=use_depthwise,
        override_base_feature_extractor_hyperparams=
        override_base_feature_extractor_hyperparams)

class SSDResnet50V1FpnFeatureExtractor(SSDResnetV1FpnFeatureExtractor):
  """SSD Resnet50 V1 FPN feature extractor."""

  def __init__(self,
               is_training,
               depth_multiplier,
               min_depth,
               pad_to_multiple,
               conv_hyperparams_fn,
               sparsity_type,
               fpn_min_level=3,
               fpn_max_level=7,
               additional_layer_depth=256,
               num_input_channels=3,
               sparse_dense_branch=False,
               channel_means=None,
               # include_root_block=True,
               depthwise_convolution=False,
               use_full_feature_extractor=False,
               use_deconvolution=False,
               max_pool_subsample=False,
               root_downsampling_rate=2,
               store_non_strided_activations=True,
               recompute_grad=False,
               reuse_weights=None,
               use_explicit_padding=False,
               use_depthwise=False,
               override_base_feature_extractor_hyperparams=False):
    """SSD Resnet50 V1 FPN feature extractor based on Resnet v1 architecture.

    Args:
      is_training: whether the network is in training mode.
      depth_multiplier: float depth multiplier for feature extractor.
      min_depth: minimum feature extractor depth.
      pad_to_multiple: the nearest multiple to zero pad the input height and
        width dimensions to.
      conv_hyperparams_fn: A function to construct tf slim arg_scope for conv2d
        and separable_conv2d ops in the layers that are added on top of the
        base feature extractor.
      fpn_min_level: the minimum level in feature pyramid networks.
      fpn_max_level: the maximum level in feature pyramid networks.
      additional_layer_depth: additional feature map layer channel depth.
      reuse_weights: Whether to reuse variables. Default is None.
      use_explicit_padding: Whether to use explicit padding when extracting
        features. Default is False. UNUSED currently.
      use_depthwise: Whether to use depthwise convolutions. UNUSED currently.
      override_base_feature_extractor_hyperparams: Whether to override
        hyperparameters of the base feature extractor with the one from
        `conv_hyperparams_fn`.
    """
    super(SSDResnet50V1FpnFeatureExtractor, self).__init__(
        is_training,
        depth_multiplier,
        min_depth,
        pad_to_multiple,
        conv_hyperparams_fn,
        resnet_v1.resnet_v1_50,
        'resnet_v1_50',
        'fpn',
        sparsity_type,
        fpn_min_level,
        fpn_max_level,
        additional_layer_depth,
        sparse_dense_branch=sparse_dense_branch,
        num_input_channels=num_input_channels,
        channel_means=channel_means,
        # include_root_block=include_root_block,
        depthwise_convolution=depthwise_convolution,
        use_full_feature_extractor=use_full_feature_extractor,
        use_deconvolution=use_deconvolution,
        max_pool_subsample=max_pool_subsample,
        store_non_strided_activations=store_non_strided_activations,
        root_downsampling_rate=root_downsampling_rate,
        recompute_grad=recompute_grad,
        reuse_weights=reuse_weights,
        use_explicit_padding=use_explicit_padding,
        use_depthwise=use_depthwise,
        override_base_feature_extractor_hyperparams=
        override_base_feature_extractor_hyperparams)

class SSDResnet50V1LightweightFpnFeatureExtractor(SSDResnetV1FpnFeatureExtractor):   # THIS ONE IS THE USED ONE
  """SSD Resnet50 V1 FPN feature extractor."""

  def __init__(self,
               is_training,
               depth_multiplier,
               min_depth,
               pad_to_multiple,
               conv_hyperparams_fn,
               sparsity_type,
               fpn_min_level=3,
               fpn_max_level=7,
               additional_layer_depth=256,
               num_input_channels=3,
               sparse_dense_branch=False,
               channel_means=None,
               # include_root_block=True,
               depthwise_convolution=False,
               use_full_feature_extractor=False,
               use_deconvolution=False,
               max_pool_subsample=False,
               root_downsampling_rate=2,
               store_non_strided_activations=True,
               recompute_grad=False,
               reuse_weights=None,
               use_explicit_padding=False,
               use_depthwise=False,
               override_base_feature_extractor_hyperparams=False):
    """SSD Resnet50 V1 FPN feature extractor based on Resnet v1 architecture.

    Args:
      is_training: whether the network is in training mode.
      depth_multiplier: float depth multiplier for feature extractor.
        UNUSED currently.
      min_depth: minimum feature extractor depth. UNUSED Currently.
      pad_to_multiple: the nearest multiple to zero pad the input height and
        width dimensions to.
      conv_hyperparams_fn: A function to construct tf slim arg_scope for conv2d
        and separable_conv2d ops in the layers that are added on top of the
        base feature extractor.
      fpn_min_level: the minimum level in feature pyramid networks.
      fpn_max_level: the maximum level in feature pyramid networks.
      additional_layer_depth: additional feature map layer channel depth.
      reuse_weights: Whether to reuse variables. Default is None.
      use_explicit_padding: Whether to use explicit padding when extracting
        features. Default is False. UNUSED currently.
      use_depthwise: Whether to use depthwise convolutions. UNUSED currently.
      override_base_feature_extractor_hyperparams: Whether to override
        hyperparameters of the base feature extractor with the one from
        `conv_hyperparams_fn`.
    """
    super(SSDResnet50V1LightweightFpnFeatureExtractor, self).__init__(
        is_training,
        depth_multiplier,
        min_depth,
        pad_to_multiple,
        conv_hyperparams_fn,
        resnet_v1.resnet_v1_50_lightweight,
        'resnet_v1_50_lightweight',
        'fpn',
        sparsity_type,
        fpn_min_level,
        fpn_max_level,
        additional_layer_depth,
        sparse_dense_branch=sparse_dense_branch,
        num_input_channels=num_input_channels,
        channel_means=channel_means,
        # include_root_block=include_root_block,
        depthwise_convolution=depthwise_convolution,
        use_full_feature_extractor=use_full_feature_extractor,
        use_deconvolution=use_deconvolution,
        max_pool_subsample=max_pool_subsample,
        store_non_strided_activations=store_non_strided_activations,
        root_downsampling_rate=root_downsampling_rate,
        recompute_grad=recompute_grad,
        reuse_weights=reuse_weights,
        use_explicit_padding=use_explicit_padding,
        use_depthwise=use_depthwise,
        override_base_feature_extractor_hyperparams=
        override_base_feature_extractor_hyperparams)

# class SSDResnet50V1LateDownsampleFpnFeatureExtractor(SSDResnetV1FpnFeatureExtractor):
#   """SSD Resnet50 V1 FPN feature extractor."""
#
#   def __init__(self,
#                is_training,
#                depth_multiplier,
#                min_depth,
#                pad_to_multiple,
#                conv_hyperparams_fn,
#                fpn_min_level=3,
#                fpn_max_level=7,
#                additional_layer_depth=256,
#                num_input_channels=3,
#                channel_means=None,
#                include_root_block=True,
#                depthwise_convolution=False,
#                use_full_feature_extractor=False,
#                use_deconvolution=False,
#                max_pool_subsample=False,
#                root_downsampling_rate=2,
#                reuse_weights=None,
#                use_explicit_padding=False,
#                use_depthwise=False,
#                override_base_feature_extractor_hyperparams=False):
#     """SSD Resnet50 V1 FPN feature extractor based on Resnet v1 architecture.
#
#     Args:
#       is_training: whether the network is in training mode.
#       depth_multiplier: float depth multiplier for feature extractor.
#         UNUSED currently.
#       min_depth: minimum feature extractor depth. UNUSED Currently.
#       pad_to_multiple: the nearest multiple to zero pad the input height and
#         width dimensions to.
#       conv_hyperparams_fn: A function to construct tf slim arg_scope for conv2d
#         and separable_conv2d ops in the layers that are added on top of the
#         base feature extractor.
#       fpn_min_level: the minimum level in feature pyramid networks.
#       fpn_max_level: the maximum level in feature pyramid networks.
#       additional_layer_depth: additional feature map layer channel depth.
#       reuse_weights: Whether to reuse variables. Default is None.
#       use_explicit_padding: Whether to use explicit padding when extracting
#         features. Default is False. UNUSED currently.
#       use_depthwise: Whether to use depthwise convolutions. UNUSED currently.
#       override_base_feature_extractor_hyperparams: Whether to override
#         hyperparameters of the base feature extractor with the one from
#         `conv_hyperparams_fn`.
#     """
#     super(SSDResnet50V1LateDownsampleFpnFeatureExtractor, self).__init__(
#         is_training,
#         depth_multiplier,
#         min_depth,
#         pad_to_multiple,
#         conv_hyperparams_fn,
#         resnet_v1.resnet_v1_50_late_downsample,
#         'resnet_v1_50_late_downsample',
#         'fpn',
#         fpn_min_level,
#         fpn_max_level,
#         additional_layer_depth,
#         num_input_channels=num_input_channels,
#         channel_means=channel_means,
#         # include_root_block=include_root_block,
#         depthwise_convolution=depthwise_convolution,
#         use_full_feature_extractor=use_full_feature_extractor,
#         use_deconvolution=use_deconvolution,
#         max_pool_subsample=max_pool_subsample,
#         root_downsampling_rate=root_downsampling_rate,
#         reuse_weights=reuse_weights,
#         use_explicit_padding=use_explicit_padding,
#         use_depthwise=use_depthwise,
#         override_base_feature_extractor_hyperparams=
#         override_base_feature_extractor_hyperparams)

class SSDResnet101V1FpnFeatureExtractor(SSDResnetV1FpnFeatureExtractor):
  """SSD Resnet101 V1 FPN feature extractor."""

  def __init__(self,
               is_training,
               depth_multiplier,
               min_depth,
               pad_to_multiple,
               conv_hyperparams_fn,
               sparsity_type,
               fpn_min_level=3,
               fpn_max_level=7,
               additional_layer_depth=256,
               num_input_channels=3,
               sparse_dense_branch=False,
               channel_means=None,
               # include_root_block=True,
               depthwise_convolution=False,
               use_full_feature_extractor=False,
               use_deconvolution=False,
               max_pool_subsample=False,
               root_downsampling_rate=2,
               store_non_strided_activations=True,
               recompute_grad=False,
               reuse_weights=None,
               use_explicit_padding=False,
               use_depthwise=False,
               override_base_feature_extractor_hyperparams=False):
    """SSD Resnet101 V1 FPN feature extractor based on Resnet v1 architecture.

    Args:
      is_training: whether the network is in training mode.
      depth_multiplier: float depth multiplier for feature extractor.
      min_depth: minimum feature extractor depth.
      pad_to_multiple: the nearest multiple to zero pad the input height and
        width dimensions to.
      conv_hyperparams_fn: A function to construct tf slim arg_scope for conv2d
        and separable_conv2d ops in the layers that are added on top of the
        base feature extractor.
      fpn_min_level: the minimum level in feature pyramid networks.
      fpn_max_level: the maximum level in feature pyramid networks.
      additional_layer_depth: additional feature map layer channel depth.
      reuse_weights: Whether to reuse variables. Default is None.
      use_explicit_padding: Whether to use explicit padding when extracting
        features. Default is False. UNUSED currently.
      use_depthwise: Whether to use depthwise convolutions. UNUSED currently.
      override_base_feature_extractor_hyperparams: Whether to override
        hyperparameters of the base feature extractor with the one from
        `conv_hyperparams_fn`.
    """
    super(SSDResnet101V1FpnFeatureExtractor, self).__init__(
        is_training,
        depth_multiplier,
        min_depth,
        pad_to_multiple,
        conv_hyperparams_fn,
        resnet_v1.resnet_v1_101,
        'resnet_v1_101',
        'fpn',
        sparsity_type,
        fpn_min_level,
        fpn_max_level,
        additional_layer_depth,
        sparse_dense_branch=sparse_dense_branch,
        num_input_channels=num_input_channels,
        channel_means=channel_means,
        # include_root_block=include_root_block,
        depthwise_convolution=depthwise_convolution,
        use_full_feature_extractor=use_full_feature_extractor,
        use_deconvolution=use_deconvolution,
        max_pool_subsample=max_pool_subsample,
        store_non_strided_activations=store_non_strided_activations,
        root_downsampling_rate=root_downsampling_rate,
        recompute_grad=recompute_grad,
        reuse_weights=reuse_weights,
        use_explicit_padding=use_explicit_padding,
        use_depthwise=use_depthwise,
        override_base_feature_extractor_hyperparams=
        override_base_feature_extractor_hyperparams)


class SSDResnet152V1FpnFeatureExtractor(SSDResnetV1FpnFeatureExtractor):
  """SSD Resnet152 V1 FPN feature extractor."""

  def __init__(self,
               is_training,
               depth_multiplier,
               min_depth,
               pad_to_multiple,
               conv_hyperparams_fn,
               sparsity_type,
               fpn_min_level=3,
               fpn_max_level=7,
               additional_layer_depth=256,
               num_input_channels=3,
               sparse_dense_branch=False,
               channel_means=None,
               # include_root_block=True,
               depthwise_convolution=False,
               use_full_feature_extractor=False,
               use_deconvolution=False,
               max_pool_subsample=False,
               root_downsampling_rate=2,
               store_non_strided_activations=True,
               recompute_grad=False,
               reuse_weights=None,
               use_explicit_padding=False,
               use_depthwise=False,
               override_base_feature_extractor_hyperparams=False):
    """SSD Resnet152 V1 FPN feature extractor based on Resnet v1 architecture.

    Args:
      is_training: whether the network is in training mode.
      depth_multiplier: float depth multiplier for feature extractor.
      min_depth: minimum feature extractor depth.
      pad_to_multiple: the nearest multiple to zero pad the input height and
        width dimensions to.
      conv_hyperparams_fn: A function to construct tf slim arg_scope for conv2d
        and separable_conv2d ops in the layers that are added on top of the
        base feature extractor.
      fpn_min_level: the minimum level in feature pyramid networks.
      fpn_max_level: the maximum level in feature pyramid networks.
      additional_layer_depth: additional feature map layer channel depth.
      reuse_weights: Whether to reuse variables. Default is None.
      use_explicit_padding: Whether to use explicit padding when extracting
        features. Default is False. UNUSED currently.
      use_depthwise: Whether to use depthwise convolutions. UNUSED currently.
      override_base_feature_extractor_hyperparams: Whether to override
        hyperparameters of the base feature extractor with the one from
        `conv_hyperparams_fn`.
    """
    super(SSDResnet152V1FpnFeatureExtractor, self).__init__(
        is_training,
        depth_multiplier,
        min_depth,
        pad_to_multiple,
        conv_hyperparams_fn,
        resnet_v1.resnet_v1_152,
        'resnet_v1_152',
        'fpn',
        sparsity_type,
        fpn_min_level,
        fpn_max_level,
        additional_layer_depth,
        sparse_dense_branch=sparse_dense_branch,
        num_input_channels=num_input_channels,
        channel_means=channel_means,
        # include_root_block=include_root_block,
        depthwise_convolution=depthwise_convolution,
        use_full_feature_extractor=use_full_feature_extractor,
        use_deconvolution=use_deconvolution,
        max_pool_subsample=max_pool_subsample,
        store_non_strided_activations=store_non_strided_activations,
        root_downsampling_rate=root_downsampling_rate,
        recompute_grad=recompute_grad,
        reuse_weights=reuse_weights,
        use_explicit_padding=use_explicit_padding,
        use_depthwise=use_depthwise,
        override_base_feature_extractor_hyperparams=
        override_base_feature_extractor_hyperparams)