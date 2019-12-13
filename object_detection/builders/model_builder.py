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

"""A function to build a DetectionModel from configuration."""

import functools
import tensorflow as tf
from object_detection.builders import anchor_generator_builder
from object_detection.builders import box_coder_builder
from object_detection.builders import box_predictor_builder
from object_detection.builders import hyperparams_builder
from object_detection.builders import image_resizer_builder
from object_detection.builders import losses_builder
from object_detection.builders import matcher_builder
from object_detection.builders import post_processing_builder
from object_detection.builders import region_similarity_calculator_builder as sim_calc
from object_detection.core import balanced_positive_negative_sampler as sampler
from object_detection.core import post_processing
from object_detection.core import target_assigner
from object_detection.meta_architectures import ssd_meta_arch
from object_detection.meta_architectures import ssd_augmentation_meta_arch
from object_detection.meta_architectures import ssd_augmentation_reuse_meta_arch
from object_detection.meta_architectures import ssd_augmentation_sequential_meta_arch
from object_detection.meta_architectures import ssd_augmentation_sharedEncoder_meta_arch
from object_detection.meta_architectures import ssd_augmentation_hybridSeq_meta_arch
from object_detection.models import ssd_resnet_v1_fpn_feature_extractor as ssd_resnet_v1_fpn
from object_detection.models import ssd_resnet_v1_multiscale_feature_extractor as ssd_resnet_multiscale
from object_detection.models import ssd_resnet_v1_ppn_feature_extractor as ssd_resnet_v1_ppn
from object_detection.models.embedded_ssd_mobilenet_v1_feature_extractor import EmbeddedSSDMobileNetV1FeatureExtractor
from object_detection.models.ssd_inception_v2_feature_extractor import SSDInceptionV2FeatureExtractor
from object_detection.models.ssd_inception_v3_feature_extractor import SSDInceptionV3FeatureExtractor
from object_detection.models.ssd_mobilenet_v1_feature_extractor import SSDMobileNetV1FeatureExtractor
from object_detection.models.ssd_mobilenet_v1_fpn_feature_extractor import SSDMobileNetV1FpnFeatureExtractor
from object_detection.models.ssd_mobilenet_v1_fpn_keras_feature_extractor import SSDMobileNetV1FpnKerasFeatureExtractor
from object_detection.models.ssd_mobilenet_v1_keras_feature_extractor import SSDMobileNetV1KerasFeatureExtractor
from object_detection.models.ssd_mobilenet_v1_ppn_feature_extractor import SSDMobileNetV1PpnFeatureExtractor
from object_detection.models.ssd_mobilenet_v2_feature_extractor import SSDMobileNetV2FeatureExtractor
from object_detection.models.ssd_mobilenet_v2_fpn_feature_extractor import SSDMobileNetV2FpnFeatureExtractor
from object_detection.models.ssd_mobilenet_v2_fpn_keras_feature_extractor import SSDMobileNetV2FpnKerasFeatureExtractor
from object_detection.models.ssd_mobilenet_v2_keras_feature_extractor import SSDMobileNetV2KerasFeatureExtractor
from object_detection.models.ssd_pnasnet_feature_extractor import SSDPNASNetFeatureExtractor
from object_detection.predictors import u_net_predictor
from object_detection.predictors import upsampling_predictor
from object_detection.predictors import ht_predictor
from object_detection.predictors import u_net_predictor_2branches_softmax_relu
from object_detection.predictors import sequential_2branches
from object_detection.predictors import shared_encoder_predictor
from object_detection.protos import model_pb2
from object_detection.utils import ops

# A map of names to SSD feature extractors.
SSD_FEATURE_EXTRACTOR_CLASS_MAP = {
    'ssd_inception_v2': SSDInceptionV2FeatureExtractor,
    'ssd_inception_v3': SSDInceptionV3FeatureExtractor,
    'ssd_mobilenet_v1': SSDMobileNetV1FeatureExtractor,
    'ssd_mobilenet_v1_fpn': SSDMobileNetV1FpnFeatureExtractor,
    'ssd_mobilenet_v1_ppn': SSDMobileNetV1PpnFeatureExtractor,
    'ssd_mobilenet_v2': SSDMobileNetV2FeatureExtractor,
    'ssd_mobilenet_v2_fpn': SSDMobileNetV2FpnFeatureExtractor,
    'ssd_resnet18_v1_fpn': ssd_resnet_v1_fpn.SSDResnet18V1FpnFeatureExtractor,
    'ssd_resnet34_v1_fpn': ssd_resnet_v1_fpn.SSDResnet34V1FpnFeatureExtractor,
    'ssd_resnet50_v1_fpn_lightweight': ssd_resnet_v1_fpn.SSDResnet50V1LightweightFpnFeatureExtractor,
    'ssd_resnet50_v1_multiscale': ssd_resnet_multiscale.SSDResnet50V1MultiscaleFeatureExtractor,
    'ssd_resnet50_v1_lightweight_multiscale': ssd_resnet_multiscale.SSDResnet50V1LightweightMultiscaleFeatureExtractor,
    'ssd_resnet50_v1_fpn': ssd_resnet_v1_fpn.SSDResnet50V1FpnFeatureExtractor,
    'ssd_resnet101_v1_fpn': ssd_resnet_v1_fpn.SSDResnet101V1FpnFeatureExtractor,
    'ssd_resnet152_v1_fpn': ssd_resnet_v1_fpn.SSDResnet152V1FpnFeatureExtractor,
    'ssd_resnet50_v1_ppn': ssd_resnet_v1_ppn.SSDResnet50V1PpnFeatureExtractor,
    'ssd_resnet101_v1_ppn':
        ssd_resnet_v1_ppn.SSDResnet101V1PpnFeatureExtractor,
    'ssd_resnet152_v1_ppn':
        ssd_resnet_v1_ppn.SSDResnet152V1PpnFeatureExtractor,
    'embedded_ssd_mobilenet_v1': EmbeddedSSDMobileNetV1FeatureExtractor,
    'ssd_pnasnet': SSDPNASNetFeatureExtractor,
}

SSD_KERAS_FEATURE_EXTRACTOR_CLASS_MAP = {
    'ssd_mobilenet_v1_keras': SSDMobileNetV1KerasFeatureExtractor,
    'ssd_mobilenet_v1_fpn_keras': SSDMobileNetV1FpnKerasFeatureExtractor,
    'ssd_mobilenet_v2_keras': SSDMobileNetV2KerasFeatureExtractor,
    'ssd_mobilenet_v2_fpn_keras': SSDMobileNetV2FpnKerasFeatureExtractor,
}


def build(model_config, is_training, add_summaries=True):
    """Builds a DetectionModel based on the model config.

    Args:
      model_config: A model.proto object containing the config for the desired
        DetectionModel.
      is_training: True if this model is being built for training purposes.
      add_summaries: Whether to add tensorflow summaries in the model graph.
    Returns:
      DetectionModel based on the config.

    Raises:
      ValueError: On invalid meta architecture or model.
    """
    if not isinstance(model_config, model_pb2.DetectionModel):
        raise ValueError('model_config not of type model_pb2.DetectionModel.')
    meta_architecture = model_config.WhichOneof('model')
    input_features = model_config.input_features
    if meta_architecture == 'ssd':
        return _build_ssd_model(model_config.ssd, is_training, add_summaries, sum(model_config.input_channels),
                                input_features)
    if meta_architecture == 'ssd_augmentation':
        return _build_ssd_augmentation_model(model_config.ssd_augmentation, is_training, add_summaries,
                                             sum(model_config.input_channels),
                                             input_features)
    if meta_architecture == 'ssd_augmentation_reuse':
        return _build_ssd_augmentation_reuse_model(model_config.ssd_augmentation_reuse, is_training, add_summaries,
                                             sum(model_config.input_channels),
                                             input_features)
    if meta_architecture == 'ssd_augmentation_sequential':
        return _build_ssd_augmentation_sequential_model(model_config.ssd_augmentation_sequential, is_training, add_summaries,
                                             sum(model_config.input_channels),
                                             input_features)
    if meta_architecture == 'ssd_augmentation_shared_encoder':
        return _build_ssd_augmentation_sharedEncoder_model(model_config.ssd_augmentation_shared_encoder, is_training, add_summaries,
                                             sum(model_config.input_channels),
                                             input_features)
    if meta_architecture == 'ssd_augmentation_hybrid_seq':
        return _build_ssd_augmentation_hybridSeq_model(model_config.ssd_augmentation_hybrid_seq, is_training, add_summaries,
                                             sum(model_config.input_channels),
                                             input_features)
    raise ValueError('Unknown meta architecture: {}'.format(meta_architecture))


def _build_ssd_feature_extractor(feature_extractor_config,
                                 is_training,
                                 input_features,
                                 freeze_batchnorm,
                                 num_input_channels=3,
                                 reuse_weights=tf.AUTO_REUSE): # instead of none  todo sep24
    """Builds a ssd_meta_arch.SSDFeatureExtractor based on config.

    Args:
      feature_extractor_config: A SSDFeatureExtractor proto config from ssd.proto.
      is_training: True if this feature extractor is being built for training.
      freeze_batchnorm: Whether to freeze batch norm parameters during
        training or not. When training with a small batch size (e.g. 1), it is
        desirable to freeze batch norm update and use pretrained batch norm
        params.
      reuse_weights: if the feature extractor should reuse weights.

    Returns:
      ssd_meta_arch.SSDFeatureExtractor based on config.

    Raises:
      ValueError: On invalid feature extractor type.
    """
    feature_type = feature_extractor_config.type
    sparsity_type = [feature in ['detections', 'decay_rate', 'intensity', 'zmax', 'zmin'] for feature in input_features]
    sparse_dense_branch = feature_extractor_config.sparse_dense_branch
    is_keras_extractor = feature_type in SSD_KERAS_FEATURE_EXTRACTOR_CLASS_MAP
    depth_multiplier = feature_extractor_config.depth_multiplier
    min_depth = feature_extractor_config.min_depth
    pad_to_multiple = feature_extractor_config.pad_to_multiple
    use_explicit_padding = feature_extractor_config.use_explicit_padding
    use_depthwise = feature_extractor_config.use_depthwise
    channel_means = feature_extractor_config.channel_means
    # include_root_block = feature_extractor_config.include_root_block
    depthwise_convolution = feature_extractor_config.depthwise_convolution
    max_pool_subsample = feature_extractor_config.max_pool_subsample
    root_downsampling_rate = feature_extractor_config.root_downsampling_rate
    store_non_strided_activations = feature_extractor_config.store_non_strided_activations
    recompute_grad = feature_extractor_config.recompute_grad

    # Check validity of fpn levels
    fpn_min_level = feature_extractor_config.fpn.min_level
    # root_downsampling_rate = feature_extractor_config.root_downsampling_rate
    # store_non_strided_activations = feature_extractor_config.store_non_strided_activations
    if root_downsampling_rate not in [1, 2]:
        raise ValueError('Root downampling rate must be 1 or 2.')
    if fpn_min_level == 0:
        if root_downsampling_rate != 1 or not store_non_strided_activations:
            raise ValueError('Configuration of FPN levels is invalid')
    elif fpn_min_level == 1:
        if (root_downsampling_rate == 2 and not store_non_strided_activations) or (root_downsampling_rate == 1 and
                                                                                   store_non_strided_activations):
            raise ValueError('Configuration of FPN levels is invalid')
    elif fpn_min_level == 2:
        if root_downsampling_rate != 2 or store_non_strided_activations:
            raise ValueError('Configuration of FPN levels is invalid')

    if is_keras_extractor:
        conv_hyperparams = hyperparams_builder.KerasLayerHyperparams(
            feature_extractor_config.conv_hyperparams)
    else:
        conv_hyperparams = hyperparams_builder.build(
            feature_extractor_config.conv_hyperparams, is_training)
    override_base_feature_extractor_hyperparams = (
        feature_extractor_config.override_base_feature_extractor_hyperparams)

    if (feature_type not in SSD_FEATURE_EXTRACTOR_CLASS_MAP) and (
            not is_keras_extractor):

        print("------------feature_type-------------")
        print(feature_type)

        raise ValueError('Unknown ssd feature_extractor: {}'.format(feature_type))

    if is_keras_extractor:
        feature_extractor_class = SSD_KERAS_FEATURE_EXTRACTOR_CLASS_MAP[
            feature_type]
    else:
        feature_extractor_class = SSD_FEATURE_EXTRACTOR_CLASS_MAP[feature_type]
    kwargs = {
        'is_training':
            is_training,
        'depth_multiplier':
            depth_multiplier,
        'min_depth':
            min_depth,
        'pad_to_multiple':
            pad_to_multiple,
        'sparsity_type':
            sparsity_type,
        'sparse_dense_branch':
            sparse_dense_branch,
        'num_input_channels':
            num_input_channels,
        'channel_means':
            channel_means,
        'use_explicit_padding':
            use_explicit_padding,
        'use_depthwise':
            use_depthwise,
        'override_base_feature_extractor_hyperparams':
            override_base_feature_extractor_hyperparams,
        # 'include_root_block':
        #     include_root_block,
        'depthwise_convolution':
            depthwise_convolution,
        'max_pool_subsample':
            max_pool_subsample,
        'root_downsampling_rate':
            root_downsampling_rate,
        'store_non_strided_activations':
            store_non_strided_activations,
        'recompute_grad':
            recompute_grad,
    }

    if feature_extractor_config.HasField('replace_preprocessor_with_placeholder'):
        kwargs.update({
            'replace_preprocessor_with_placeholder':
                feature_extractor_config.replace_preprocessor_with_placeholder
        })

    if is_keras_extractor:
        kwargs.update({
            'conv_hyperparams': conv_hyperparams,
            'inplace_batchnorm_update': False,
            'freeze_batchnorm': freeze_batchnorm
        })
    else:
        kwargs.update({
            'conv_hyperparams_fn': conv_hyperparams,
            'reuse_weights': reuse_weights,
        })

    if feature_extractor_config.HasField('fpn'):
        kwargs.update({
            'fpn_min_level':
                feature_extractor_config.fpn.min_level,
            'fpn_max_level':
                feature_extractor_config.fpn.max_level,
            'additional_layer_depth':
                feature_extractor_config.fpn.additional_layer_depth,
            'use_deconvolution':
                feature_extractor_config.fpn.use_deconvolution,
            'use_full_feature_extractor':
                feature_extractor_config.fpn.use_full_feature_extractor,
        })

    return feature_extractor_class(**kwargs)


def _build_ssd_model(ssd_config, is_training, add_summaries, num_input_channels,
                     input_features):
    """Builds an SSD detection model based on the model config.

    Args:
      ssd_config: A ssd.proto object containing the config for the desired
        SSDMetaArch.
      is_training: True if this model is being built for training purposes.
      add_summaries: Whether to add tf summaries in the model.
    Returns:
      SSDMetaArch based on the config.

    Raises:
      ValueError: If ssd_config.type is not recognized (i.e. not registered in
        model_class_map).
    """
    num_classes = ssd_config.num_classes

    # Feature extractor
    feature_extractor = _build_ssd_feature_extractor(
        feature_extractor_config=ssd_config.feature_extractor,
        freeze_batchnorm=ssd_config.freeze_batchnorm,
        is_training=is_training,
        input_features=input_features,
        num_input_channels=num_input_channels)

    box_coder = box_coder_builder.build(ssd_config.box_coder)
    matcher = matcher_builder.build(ssd_config.matcher)
    region_similarity_calculator = sim_calc.build(
        ssd_config.similarity_calculator)
    encode_background_as_zeros = ssd_config.encode_background_as_zeros
    negative_class_weight = ssd_config.negative_class_weight
    anchor_generator = anchor_generator_builder.build(
        ssd_config.anchor_generator)
    # , ssd_config.feature_extractor.include_root_block,
    # ssd_config.feature_extractor.root_downsampling_rate, ssd_config.feature_extractor.type,
    # ssd_config.feature_extractor.store_non_strided_activations
    if feature_extractor.is_keras_model:
        ssd_box_predictor = box_predictor_builder.build_keras(
            hyperparams_fn=hyperparams_builder.KerasLayerHyperparams,
            freeze_batchnorm=ssd_config.freeze_batchnorm,
            inplace_batchnorm_update=False,
            num_predictions_per_location_list=anchor_generator
                .num_anchors_per_location(),
            box_predictor_config=ssd_config.box_predictor,
            is_training=is_training,
            num_classes=num_classes,
            add_background_class=ssd_config.add_background_class)
    else:
        ssd_box_predictor = box_predictor_builder.build(
            hyperparams_builder.build, ssd_config.box_predictor, is_training,
            num_classes, ssd_config.add_background_class)
    image_resizer_fn = image_resizer_builder.build(ssd_config.image_resizer)
    non_max_suppression_fn, score_conversion_fn = post_processing_builder.build(
        ssd_config.post_processing)
    (classification_loss, localization_loss, classification_weight,
     localization_weight, hard_example_miner, random_example_sampler,
     expected_loss_weights_fn) = losses_builder.build(ssd_config.loss)
    normalize_loss_by_num_matches = ssd_config.normalize_loss_by_num_matches
    normalize_loc_loss_by_codesize = ssd_config.normalize_loc_loss_by_codesize
    # specific_threshold = ssd_config.specific_threshold
    # threshold_offset = ssd_config.threshold_offset
    # increse_small_object_size = ssd_config.increse_small_object_size

    equalization_loss_config = ops.EqualizationLossConfig(
        weight=ssd_config.loss.equalization_loss.weight,
        exclude_prefixes=ssd_config.loss.equalization_loss.exclude_prefixes)

    target_assigner_instance = target_assigner.TargetAssigner(
        region_similarity_calculator,
        matcher,
        box_coder,
        negative_class_weight=negative_class_weight
        # increse_small_object_size=increse_small_object_size,
        # specific_threshold=specific_threshold,
        # threshold_offset=threshold_offset
    )

    ssd_meta_arch_fn = ssd_meta_arch.SSDMetaArch
    kwargs = {}

    return ssd_meta_arch_fn(
        is_training=is_training,
        anchor_generator=anchor_generator,
        box_predictor=ssd_box_predictor,
        box_coder=box_coder,
        feature_extractor=feature_extractor,
        encode_background_as_zeros=encode_background_as_zeros,
        image_resizer_fn=image_resizer_fn,
        non_max_suppression_fn=non_max_suppression_fn,
        score_conversion_fn=score_conversion_fn,
        classification_loss=classification_loss,
        localization_loss=localization_loss,
        classification_loss_weight=classification_weight,
        localization_loss_weight=localization_weight,
        normalize_loss_by_num_matches=normalize_loss_by_num_matches,
        hard_example_miner=hard_example_miner,
        target_assigner_instance=target_assigner_instance,
        add_summaries=add_summaries,
        normalize_loc_loss_by_codesize=normalize_loc_loss_by_codesize,
        freeze_batchnorm=ssd_config.freeze_batchnorm,
        inplace_batchnorm_update=ssd_config.inplace_batchnorm_update,
        add_background_class=ssd_config.add_background_class,
        explicit_background_class=ssd_config.explicit_background_class,
        random_example_sampler=random_example_sampler,
        expected_loss_weights_fn=expected_loss_weights_fn,
        use_confidences_as_targets=ssd_config.use_confidences_as_targets,
        implicit_example_weight=ssd_config.implicit_example_weight,
        equalization_loss_config=equalization_loss_config,
        **kwargs)


def _build_ssd_augmentation_model(ssd_augm_config, is_training, add_summaries, num_input_channels,
                                  input_features):
    """Builds an SSD detection model based on the model config.

    Args:
      ssd_augmentation_config: A ssd.proto object containing the config for the desired
        SSDMetaArch.
      is_training: True if this model is being built for training purposes.
      add_summaries: Whether to add tf summaries in the model.
    Returns:
      SSDMetaArch based on the config.

    Raises:
      ValueError: If ssd_augm_config.type is not recognized (i.e. not registered in
        model_class_map).
    """
    num_classes = ssd_augm_config.num_classes

    # Feature extractor
    ssd_augm_config.feature_extractor.fpn.use_full_feature_extractor = False
    feature_extractor = _build_ssd_feature_extractor(
        feature_extractor_config=ssd_augm_config.feature_extractor,
        freeze_batchnorm=ssd_augm_config.freeze_batchnorm,
        is_training=is_training,
        input_features=input_features,
        num_input_channels=num_input_channels)

    box_coder = box_coder_builder.build(ssd_augm_config.box_coder)
    matcher = matcher_builder.build(ssd_augm_config.matcher)
    region_similarity_calculator = sim_calc.build(
        ssd_augm_config.similarity_calculator)
    encode_background_as_zeros = ssd_augm_config.encode_background_as_zeros
    negative_class_weight = ssd_augm_config.negative_class_weight
    anchor_generator = anchor_generator_builder.build(
        ssd_augm_config.anchor_generator

        # ssd_augm_config.feature_extractor.include_root_block,
        # ssd_augm_config.feature_extractor.root_downsampling_rate, ssd_augm_config.feature_extractor.type,
        # ssd_augm_config.feature_extractor.store_non_strided_activations
    )

    if feature_extractor.is_keras_model:
        ssd_box_predictor = box_predictor_builder.build_keras(
            hyperparams_fn=hyperparams_builder.KerasLayerHyperparams,
            freeze_batchnorm=ssd_augm_config.freeze_batchnorm,
            inplace_batchnorm_update=False,
            num_predictions_per_location_list=anchor_generator
                .num_anchors_per_location(),
            box_predictor_config=ssd_augm_config.box_predictor,
            is_training=is_training,
            num_classes=num_classes,
            add_background_class=ssd_augm_config.add_background_class)
    else:
        ssd_box_predictor = box_predictor_builder.build(
            hyperparams_builder.build, ssd_augm_config.box_predictor, is_training,
            num_classes, ssd_augm_config.add_background_class)

    ## Add augmentation network
    if ssd_augm_config.beliefs_predictor.predictor == 'u_net':
        ssd_augmentation_predictor = u_net_predictor.UNetPredictor(
            is_training=is_training,
            layer_norm=ssd_augm_config.beliefs_predictor.layer_norm,
            stack_size=ssd_augm_config.beliefs_predictor.stack_size,
            kernel_size=ssd_augm_config.beliefs_predictor.kernel_size,
            filters=ssd_augm_config.beliefs_predictor.filters)
    elif ssd_augm_config.beliefs_predictor.predictor == 'upsampling':
        ssd_augmentation_predictor = upsampling_predictor.UpsamplingPredictor(
            is_training=is_training,
            layer_norm=ssd_augm_config.beliefs_predictor.layer_norm,
            stack_size=ssd_augm_config.beliefs_predictor.stack_size,
            kernel_size=ssd_augm_config.beliefs_predictor.kernel_size,
            filters=ssd_augm_config.beliefs_predictor.filters)
    elif ssd_augm_config.beliefs_predictor.predictor == 'hybrid_task_cascade':
        ssd_augmentation_predictor = ht_predictor.HTPredictor(
            is_training=is_training,
            layer_norm=ssd_augm_config.beliefs_predictor.layer_norm,
            stack_size=ssd_augm_config.beliefs_predictor.stack_size,
            kernel_size=ssd_augm_config.beliefs_predictor.kernel_size,
            filters=ssd_augm_config.beliefs_predictor.filters)
    elif ssd_augm_config.beliefs_predictor.predictor == 'u_net_2branches':
        ssd_augmentation_predictor = u_net_predictor_2branches_softmax_relu.UNet2branchesPredictor(
            is_training=is_training,
            layer_norm=ssd_augm_config.beliefs_predictor.layer_norm,
            stack_size=ssd_augm_config.beliefs_predictor.stack_size,
            kernel_size=ssd_augm_config.beliefs_predictor.kernel_size,
            filters=ssd_augm_config.beliefs_predictor.filters)
    elif ssd_augm_config.beliefs_predictor.predictor == 'sequential':
        ssd_augmentation_predictor = sequential_2branches.Sequential2branchesPredictor(
            is_training=is_training,
            layer_norm=ssd_augm_config.beliefs_predictor.layer_norm,
            stack_size=ssd_augm_config.beliefs_predictor.stack_size,
            kernel_size=ssd_augm_config.beliefs_predictor.kernel_size,
            filters=ssd_augm_config.beliefs_predictor.filters)
    elif ssd_augm_config.beliefs_predictor.predictor == 'shared_encoder':
        ssd_augmentation_predictor = shared_encoder_predictor.SharedEncoderPredictor(
            is_training=is_training,
            layer_norm=ssd_augm_config.beliefs_predictor.layer_norm,
            stack_size=ssd_augm_config.beliefs_predictor.stack_size,
            kernel_size=ssd_augm_config.beliefs_predictor.kernel_size,
            filters=ssd_augm_config.beliefs_predictor.filters)
    else:
        raise RuntimeError('unknown predictor %s for augmentation branch' % ssd_augm_config.beliefs_predictor.predictor)

    image_resizer_fn = image_resizer_builder.build(ssd_augm_config.image_resizer)
    non_max_suppression_fn, score_conversion_fn = post_processing_builder.build(
        ssd_augm_config.post_processing)
    (classification_loss, localization_loss, classification_weight,
     localization_weight, hard_example_miner, random_example_sampler,
     expected_loss_weights_fn) = losses_builder.build(ssd_augm_config.loss)
    normalize_loss_by_num_matches = ssd_augm_config.normalize_loss_by_num_matches
    normalize_loc_loss_by_codesize = ssd_augm_config.normalize_loc_loss_by_codesize
    # specific_threshold = ssd_augm_config.specific_threshold
    # threshold_offset = ssd_augm_config.threshold_offset
    # increse_small_object_size = ssd_augm_config.increse_small_object_size

    equalization_loss_config = ops.EqualizationLossConfig(
        weight=ssd_augm_config.loss.equalization_loss.weight,
        exclude_prefixes=ssd_augm_config.loss.equalization_loss.exclude_prefixes)

    target_assigner_instance = target_assigner.TargetAssigner(
        region_similarity_calculator,
        matcher,
        box_coder,
        negative_class_weight=negative_class_weight
        # increse_small_object_size=increse_small_object_size,
        # specific_threshold=specific_threshold,
        # threshold_offset=threshold_offset
    )

    ssd_augm_meta_arch_fn = ssd_augmentation_meta_arch.SSDAugmentationMetaArch
    kwargs = {}

    return ssd_augm_meta_arch_fn(
        is_training=is_training,
        anchor_generator=anchor_generator,
        box_predictor=ssd_box_predictor,
        augmentation_predictor=ssd_augmentation_predictor,
        factor_loss_fused_bel_O=ssd_augm_config.augmentation_branch.factor_loss_fused_bel_O,
        factor_loss_fused_bel_F=ssd_augm_config.augmentation_branch.factor_loss_fused_bel_F,
        factor_loss_fused_zmax_det=ssd_augm_config.augmentation_branch.factor_loss_fused_zmax_det,
        factor_loss_fused_obs_zmin=ssd_augm_config.augmentation_branch.factor_loss_fused_obs_zmin,
        factor_loss_augm=ssd_augm_config.augmentation_branch.factor_loss_augm,
        box_coder=box_coder,
        feature_extractor=feature_extractor,
        encode_background_as_zeros=encode_background_as_zeros,
        image_resizer_fn=image_resizer_fn,
        non_max_suppression_fn=non_max_suppression_fn,
        score_conversion_fn=score_conversion_fn,
        use_uncertainty_weighting_loss=ssd_augm_config.loss.use_uncertainty_weighting_loss,
        classification_loss=classification_loss,
        localization_loss=localization_loss,
        classification_loss_weight=classification_weight,
        localization_loss_weight=localization_weight,
        normalize_loss_by_num_matches=normalize_loss_by_num_matches,
        hard_example_miner=hard_example_miner,
        target_assigner_instance=target_assigner_instance,
        add_summaries=add_summaries,
        normalize_loc_loss_by_codesize=normalize_loc_loss_by_codesize,
        freeze_batchnorm=ssd_augm_config.freeze_batchnorm,
        inplace_batchnorm_update=ssd_augm_config.inplace_batchnorm_update,
        add_background_class=ssd_augm_config.add_background_class,
        explicit_background_class=ssd_augm_config.explicit_background_class,
        random_example_sampler=random_example_sampler,
        expected_loss_weights_fn=expected_loss_weights_fn,
        use_confidences_as_targets=ssd_augm_config.use_confidences_as_targets,
        implicit_example_weight=ssd_augm_config.implicit_example_weight,
        equalization_loss_config=equalization_loss_config,
        **kwargs)


def _build_ssd_augmentation_reuse_model(ssd_augm_config, is_training, add_summaries, num_input_channels,
                                  input_features):
    """Builds an SSD detection model based on the model config.

    Args:
      ssd_augmentation_config: A ssd.proto object containing the config for the desired
        SSDMetaArch.
      is_training: True if this model is being built for training purposes.
      add_summaries: Whether to add tf summaries in the model.
    Returns:
      SSDMetaArch based on the config.

    Raises:
      ValueError: If ssd_augm_config.type is not recognized (i.e. not registered in
        model_class_map).
    """
    num_classes = ssd_augm_config.num_classes

    # Feature extractor
    ssd_augm_config.feature_extractor.fpn.use_full_feature_extractor = False
    feature_extractor = _build_ssd_feature_extractor(
        feature_extractor_config=ssd_augm_config.feature_extractor,
        freeze_batchnorm=ssd_augm_config.freeze_batchnorm,
        is_training=is_training,
        input_features=input_features,
        num_input_channels=num_input_channels)

    box_coder = box_coder_builder.build(ssd_augm_config.box_coder)
    matcher = matcher_builder.build(ssd_augm_config.matcher)
    region_similarity_calculator = sim_calc.build(
        ssd_augm_config.similarity_calculator)
    encode_background_as_zeros = ssd_augm_config.encode_background_as_zeros
    negative_class_weight = ssd_augm_config.negative_class_weight
    anchor_generator = anchor_generator_builder.build(
        ssd_augm_config.anchor_generator

        # ssd_augm_config.feature_extractor.include_root_block,
        # ssd_augm_config.feature_extractor.root_downsampling_rate, ssd_augm_config.feature_extractor.type,
        # ssd_augm_config.feature_extractor.store_non_strided_activations
    )

    if feature_extractor.is_keras_model:
        ssd_box_predictor = box_predictor_builder.build_keras(
            hyperparams_fn=hyperparams_builder.KerasLayerHyperparams,
            freeze_batchnorm=ssd_augm_config.freeze_batchnorm,
            inplace_batchnorm_update=False,
            num_predictions_per_location_list=anchor_generator
                .num_anchors_per_location(),
            box_predictor_config=ssd_augm_config.box_predictor,
            is_training=is_training,
            num_classes=num_classes,
            add_background_class=ssd_augm_config.add_background_class)
    else:
        ssd_box_predictor = box_predictor_builder.build(
            hyperparams_builder.build, ssd_augm_config.box_predictor, is_training,
            num_classes, ssd_augm_config.add_background_class)

    ## Add augmentation network
    if ssd_augm_config.beliefs_predictor.predictor == 'u_net':
        ssd_augmentation_predictor = u_net_predictor.UNetPredictor(
            is_training=is_training,
            layer_norm=ssd_augm_config.beliefs_predictor.layer_norm,
            stack_size=ssd_augm_config.beliefs_predictor.stack_size,
            kernel_size=ssd_augm_config.beliefs_predictor.kernel_size,
            filters=ssd_augm_config.beliefs_predictor.filters)
    elif ssd_augm_config.beliefs_predictor.predictor == 'upsampling':
        ssd_augmentation_predictor = upsampling_predictor.UpsamplingPredictor(
            is_training=is_training,
            layer_norm=ssd_augm_config.beliefs_predictor.layer_norm,
            stack_size=ssd_augm_config.beliefs_predictor.stack_size,
            kernel_size=ssd_augm_config.beliefs_predictor.kernel_size,
            filters=ssd_augm_config.beliefs_predictor.filters)
    elif ssd_augm_config.beliefs_predictor.predictor == 'hybrid_task_cascade':
        ssd_augmentation_predictor = ht_predictor.HTPredictor(
            is_training=is_training,
            layer_norm=ssd_augm_config.beliefs_predictor.layer_norm,
            stack_size=ssd_augm_config.beliefs_predictor.stack_size,
            kernel_size=ssd_augm_config.beliefs_predictor.kernel_size,
            filters=ssd_augm_config.beliefs_predictor.filters)
    elif ssd_augm_config.beliefs_predictor.predictor == 'u_net_2branches':
        ssd_augmentation_predictor = u_net_predictor_2branches_softmax_relu.UNet2branchesPredictor(
            is_training=is_training,
            layer_norm=ssd_augm_config.beliefs_predictor.layer_norm,
            stack_size=ssd_augm_config.beliefs_predictor.stack_size,
            kernel_size=ssd_augm_config.beliefs_predictor.kernel_size,
            filters=ssd_augm_config.beliefs_predictor.filters)
    elif ssd_augm_config.beliefs_predictor.predictor == 'sequential':
        ssd_augmentation_predictor = sequential_2branches.Sequential2branchesPredictor(
            is_training=is_training,
            layer_norm=ssd_augm_config.beliefs_predictor.layer_norm,
            stack_size=ssd_augm_config.beliefs_predictor.stack_size,
            kernel_size=ssd_augm_config.beliefs_predictor.kernel_size,
            filters=ssd_augm_config.beliefs_predictor.filters)
    elif ssd_augm_config.beliefs_predictor.predictor == 'shared_encoder':
        ssd_augmentation_predictor = shared_encoder_predictor.SharedEncoderPredictor(
            is_training=is_training,
            layer_norm=ssd_augm_config.beliefs_predictor.layer_norm,
            stack_size=ssd_augm_config.beliefs_predictor.stack_size,
            kernel_size=ssd_augm_config.beliefs_predictor.kernel_size,
            filters=ssd_augm_config.beliefs_predictor.filters)
    else:
        raise RuntimeError('unknown predictor %s for augmentation branch' % ssd_augm_config.beliefs_predictor.predictor)

    image_resizer_fn = image_resizer_builder.build(ssd_augm_config.image_resizer)
    non_max_suppression_fn, score_conversion_fn = post_processing_builder.build(
        ssd_augm_config.post_processing)
    (classification_loss, localization_loss, classification_weight,
     localization_weight, hard_example_miner, random_example_sampler,
     expected_loss_weights_fn) = losses_builder.build(ssd_augm_config.loss)
    normalize_loss_by_num_matches = ssd_augm_config.normalize_loss_by_num_matches
    normalize_loc_loss_by_codesize = ssd_augm_config.normalize_loc_loss_by_codesize
    # specific_threshold = ssd_augm_config.specific_threshold
    # threshold_offset = ssd_augm_config.threshold_offset
    # increse_small_object_size = ssd_augm_config.increse_small_object_size

    equalization_loss_config = ops.EqualizationLossConfig(
        weight=ssd_augm_config.loss.equalization_loss.weight,
        exclude_prefixes=ssd_augm_config.loss.equalization_loss.exclude_prefixes)

    target_assigner_instance = target_assigner.TargetAssigner(
        region_similarity_calculator,
        matcher,
        box_coder,
        negative_class_weight=negative_class_weight
        # increse_small_object_size=increse_small_object_size,
        # specific_threshold=specific_threshold,
        # threshold_offset=threshold_offset
    )

    ssd_augm_meta_arch_fn = ssd_augmentation_reuse_meta_arch.SSDAugmentationReuseMetaArch
    kwargs = {}

    return ssd_augm_meta_arch_fn(
        is_training=is_training,
        anchor_generator=anchor_generator,
        box_predictor=ssd_box_predictor,
        augmentation_predictor=ssd_augmentation_predictor,
        factor_loss_fused_bel_O=ssd_augm_config.augmentation_branch.factor_loss_fused_bel_O,
        factor_loss_fused_bel_F=ssd_augm_config.augmentation_branch.factor_loss_fused_bel_F,
        factor_loss_fused_zmax_det=ssd_augm_config.augmentation_branch.factor_loss_fused_zmax_det,
        factor_loss_fused_obs_zmin=ssd_augm_config.augmentation_branch.factor_loss_fused_obs_zmin,
        factor_loss_augm=ssd_augm_config.augmentation_branch.factor_loss_augm,
        box_coder=box_coder,
        feature_extractor=feature_extractor,
        encode_background_as_zeros=encode_background_as_zeros,
        image_resizer_fn=image_resizer_fn,
        non_max_suppression_fn=non_max_suppression_fn,
        score_conversion_fn=score_conversion_fn,
        use_uncertainty_weighting_loss=ssd_augm_config.loss.use_uncertainty_weighting_loss,
        classification_loss=classification_loss,
        localization_loss=localization_loss,
        classification_loss_weight=classification_weight,
        localization_loss_weight=localization_weight,
        normalize_loss_by_num_matches=normalize_loss_by_num_matches,
        hard_example_miner=hard_example_miner,
        target_assigner_instance=target_assigner_instance,
        add_summaries=add_summaries,
        normalize_loc_loss_by_codesize=normalize_loc_loss_by_codesize,
        freeze_batchnorm=ssd_augm_config.freeze_batchnorm,
        inplace_batchnorm_update=ssd_augm_config.inplace_batchnorm_update,
        add_background_class=ssd_augm_config.add_background_class,
        explicit_background_class=ssd_augm_config.explicit_background_class,
        random_example_sampler=random_example_sampler,
        expected_loss_weights_fn=expected_loss_weights_fn,
        use_confidences_as_targets=ssd_augm_config.use_confidences_as_targets,
        implicit_example_weight=ssd_augm_config.implicit_example_weight,
        equalization_loss_config=equalization_loss_config,
        **kwargs)



def _build_ssd_augmentation_sequential_model(ssd_augm_config, is_training, add_summaries, num_input_channels,
                                  input_features):
    """Builds an SSD detection model based on the model config.

    Args:
      ssd_augmentation_config: A ssd.proto object containing the config for the desired
        SSDMetaArch.
      is_training: True if this model is being built for training purposes.
      add_summaries: Whether to add tf summaries in the model.
    Returns:
      SSDMetaArch based on the config.

    Raises:
      ValueError: If ssd_augm_config.type is not recognized (i.e. not registered in
        model_class_map).
    """
    num_classes = ssd_augm_config.num_classes

    # Feature extractor
    ssd_augm_config.feature_extractor.fpn.use_full_feature_extractor = False
    feature_extractor = _build_ssd_feature_extractor(
        feature_extractor_config=ssd_augm_config.feature_extractor,
        freeze_batchnorm=ssd_augm_config.freeze_batchnorm,
        is_training=is_training,
        input_features=input_features,
        num_input_channels=num_input_channels)

    box_coder = box_coder_builder.build(ssd_augm_config.box_coder)
    matcher = matcher_builder.build(ssd_augm_config.matcher)
    region_similarity_calculator = sim_calc.build(
        ssd_augm_config.similarity_calculator)
    encode_background_as_zeros = ssd_augm_config.encode_background_as_zeros
    negative_class_weight = ssd_augm_config.negative_class_weight
    anchor_generator = anchor_generator_builder.build(
        ssd_augm_config.anchor_generator

        # ssd_augm_config.feature_extractor.include_root_block,
        # ssd_augm_config.feature_extractor.root_downsampling_rate, ssd_augm_config.feature_extractor.type,
        # ssd_augm_config.feature_extractor.store_non_strided_activations
    )

    if feature_extractor.is_keras_model:
        ssd_box_predictor = box_predictor_builder.build_keras(
            hyperparams_fn=hyperparams_builder.KerasLayerHyperparams,
            freeze_batchnorm=ssd_augm_config.freeze_batchnorm,
            inplace_batchnorm_update=False,
            num_predictions_per_location_list=anchor_generator
                .num_anchors_per_location(),
            box_predictor_config=ssd_augm_config.box_predictor,
            is_training=is_training,
            num_classes=num_classes,
            add_background_class=ssd_augm_config.add_background_class)
    else:
        ssd_box_predictor = box_predictor_builder.build(
            hyperparams_builder.build, ssd_augm_config.box_predictor, is_training,
            num_classes, ssd_augm_config.add_background_class)

    ## Add augmentation network
    if ssd_augm_config.beliefs_predictor.predictor == 'u_net':
        ssd_augmentation_predictor = u_net_predictor.UNetPredictor(
            is_training=is_training,
            layer_norm=ssd_augm_config.beliefs_predictor.layer_norm,
            stack_size=ssd_augm_config.beliefs_predictor.stack_size,
            kernel_size=ssd_augm_config.beliefs_predictor.kernel_size,
            filters=ssd_augm_config.beliefs_predictor.filters)
    elif ssd_augm_config.beliefs_predictor.predictor == 'upsampling':
        ssd_augmentation_predictor = upsampling_predictor.UpsamplingPredictor(
            is_training=is_training,
            layer_norm=ssd_augm_config.beliefs_predictor.layer_norm,
            stack_size=ssd_augm_config.beliefs_predictor.stack_size,
            kernel_size=ssd_augm_config.beliefs_predictor.kernel_size,
            filters=ssd_augm_config.beliefs_predictor.filters)
    elif ssd_augm_config.beliefs_predictor.predictor == 'hybrid_task_cascade':
        ssd_augmentation_predictor = ht_predictor.HTPredictor(
            is_training=is_training,
            layer_norm=ssd_augm_config.beliefs_predictor.layer_norm,
            stack_size=ssd_augm_config.beliefs_predictor.stack_size,
            kernel_size=ssd_augm_config.beliefs_predictor.kernel_size,
            filters=ssd_augm_config.beliefs_predictor.filters)
    elif ssd_augm_config.beliefs_predictor.predictor == 'u_net_2branches':
        ssd_augmentation_predictor = u_net_predictor_2branches_softmax_relu.UNet2branchesPredictor(
            is_training=is_training,
            layer_norm=ssd_augm_config.beliefs_predictor.layer_norm,
            stack_size=ssd_augm_config.beliefs_predictor.stack_size,
            kernel_size=ssd_augm_config.beliefs_predictor.kernel_size,
            filters=ssd_augm_config.beliefs_predictor.filters)
    elif ssd_augm_config.beliefs_predictor.predictor == 'sequential':
        ssd_augmentation_predictor = sequential_2branches.Sequential2branchesPredictor(
            is_training=is_training,
            layer_norm=ssd_augm_config.beliefs_predictor.layer_norm,
            stack_size=ssd_augm_config.beliefs_predictor.stack_size,
            kernel_size=ssd_augm_config.beliefs_predictor.kernel_size,
            filters=ssd_augm_config.beliefs_predictor.filters)
    else:
        raise RuntimeError('unknown predictor %s for augmentation branch' % ssd_augm_config.beliefs_predictor.predictor)

    image_resizer_fn = image_resizer_builder.build(ssd_augm_config.image_resizer)
    non_max_suppression_fn, score_conversion_fn = post_processing_builder.build(
        ssd_augm_config.post_processing)
    (classification_loss, localization_loss, classification_weight,
     localization_weight, hard_example_miner, random_example_sampler,
     expected_loss_weights_fn) = losses_builder.build(ssd_augm_config.loss)
    normalize_loss_by_num_matches = ssd_augm_config.normalize_loss_by_num_matches
    normalize_loc_loss_by_codesize = ssd_augm_config.normalize_loc_loss_by_codesize
    # specific_threshold = ssd_augm_config.specific_threshold
    # threshold_offset = ssd_augm_config.threshold_offset
    # increse_small_object_size = ssd_augm_config.increse_small_object_size

    equalization_loss_config = ops.EqualizationLossConfig(
        weight=ssd_augm_config.loss.equalization_loss.weight,
        exclude_prefixes=ssd_augm_config.loss.equalization_loss.exclude_prefixes)

    target_assigner_instance = target_assigner.TargetAssigner(
        region_similarity_calculator,
        matcher,
        box_coder,
        negative_class_weight=negative_class_weight
        # increse_small_object_size=increse_small_object_size,
        # specific_threshold=specific_threshold,
        # threshold_offset=threshold_offset
    )

    ssd_augm_meta_arch_fn = ssd_augmentation_sequential_meta_arch.SSDAugmentationSequentialMetaArch
    kwargs = {}

    return ssd_augm_meta_arch_fn(
        is_training=is_training,
        anchor_generator=anchor_generator,
        box_predictor=ssd_box_predictor,
        augmentation_predictor=ssd_augmentation_predictor,
        factor_loss_fused_bel_O=ssd_augm_config.augmentation_branch.factor_loss_fused_bel_O,
        factor_loss_fused_bel_F=ssd_augm_config.augmentation_branch.factor_loss_fused_bel_F,
        factor_loss_fused_zmax_det=ssd_augm_config.augmentation_branch.factor_loss_fused_zmax_det,
        factor_loss_fused_obs_zmin=ssd_augm_config.augmentation_branch.factor_loss_fused_obs_zmin,
        factor_loss_augm=ssd_augm_config.augmentation_branch.factor_loss_augm,
        box_coder=box_coder,
        feature_extractor=feature_extractor,
        encode_background_as_zeros=encode_background_as_zeros,
        image_resizer_fn=image_resizer_fn,
        non_max_suppression_fn=non_max_suppression_fn,
        score_conversion_fn=score_conversion_fn,
        use_uncertainty_weighting_loss=ssd_augm_config.loss.use_uncertainty_weighting_loss,
        classification_loss=classification_loss,
        localization_loss=localization_loss,
        classification_loss_weight=classification_weight,
        localization_loss_weight=localization_weight,
        normalize_loss_by_num_matches=normalize_loss_by_num_matches,
        hard_example_miner=hard_example_miner,
        target_assigner_instance=target_assigner_instance,
        add_summaries=add_summaries,
        normalize_loc_loss_by_codesize=normalize_loc_loss_by_codesize,
        freeze_batchnorm=ssd_augm_config.freeze_batchnorm,
        inplace_batchnorm_update=ssd_augm_config.inplace_batchnorm_update,
        add_background_class=ssd_augm_config.add_background_class,
        explicit_background_class=ssd_augm_config.explicit_background_class,
        random_example_sampler=random_example_sampler,
        expected_loss_weights_fn=expected_loss_weights_fn,
        use_confidences_as_targets=ssd_augm_config.use_confidences_as_targets,
        implicit_example_weight=ssd_augm_config.implicit_example_weight,
        equalization_loss_config=equalization_loss_config,
        **kwargs)


def _build_ssd_augmentation_sharedEncoder_model(ssd_augm_config, is_training, add_summaries, num_input_channels,
                                  input_features):
    """Builds an SSD detection model based on the model config.

    Args:
      ssd_augmentation_config: A ssd.proto object containing the config for the desired
        SSDMetaArch.
      is_training: True if this model is being built for training purposes.
      add_summaries: Whether to add tf summaries in the model.
    Returns:
      SSDMetaArch based on the config.

    Raises:
      ValueError: If ssd_augm_config.type is not recognized (i.e. not registered in
        model_class_map).
    """
    num_classes = ssd_augm_config.num_classes

    # Feature extractor
    ssd_augm_config.feature_extractor.fpn.use_full_feature_extractor = False
    feature_extractor = _build_ssd_feature_extractor(
        feature_extractor_config=ssd_augm_config.feature_extractor,
        freeze_batchnorm=ssd_augm_config.freeze_batchnorm,
        is_training=is_training,
        input_features=input_features,
        num_input_channels=num_input_channels)

    box_coder = box_coder_builder.build(ssd_augm_config.box_coder)
    matcher = matcher_builder.build(ssd_augm_config.matcher)
    region_similarity_calculator = sim_calc.build(
        ssd_augm_config.similarity_calculator)
    encode_background_as_zeros = ssd_augm_config.encode_background_as_zeros
    negative_class_weight = ssd_augm_config.negative_class_weight
    anchor_generator = anchor_generator_builder.build(
        ssd_augm_config.anchor_generator

        # ssd_augm_config.feature_extractor.include_root_block,
        # ssd_augm_config.feature_extractor.root_downsampling_rate, ssd_augm_config.feature_extractor.type,
        # ssd_augm_config.feature_extractor.store_non_strided_activations
    )

    if feature_extractor.is_keras_model:
        ssd_box_predictor = box_predictor_builder.build_keras(
            hyperparams_fn=hyperparams_builder.KerasLayerHyperparams,
            freeze_batchnorm=ssd_augm_config.freeze_batchnorm,
            inplace_batchnorm_update=False,
            num_predictions_per_location_list=anchor_generator
                .num_anchors_per_location(),
            box_predictor_config=ssd_augm_config.box_predictor,
            is_training=is_training,
            num_classes=num_classes,
            add_background_class=ssd_augm_config.add_background_class)
    else:
        ssd_box_predictor = box_predictor_builder.build(
            hyperparams_builder.build, ssd_augm_config.box_predictor, is_training,
            num_classes, ssd_augm_config.add_background_class)

    ## Add augmentation network
    if ssd_augm_config.beliefs_predictor.predictor == 'u_net':
        ssd_augmentation_predictor = u_net_predictor.UNetPredictor(
            is_training=is_training,
            layer_norm=ssd_augm_config.beliefs_predictor.layer_norm,
            stack_size=ssd_augm_config.beliefs_predictor.stack_size,
            kernel_size=ssd_augm_config.beliefs_predictor.kernel_size,
            filters=ssd_augm_config.beliefs_predictor.filters)
    elif ssd_augm_config.beliefs_predictor.predictor == 'upsampling':
        ssd_augmentation_predictor = upsampling_predictor.UpsamplingPredictor(
            is_training=is_training,
            layer_norm=ssd_augm_config.beliefs_predictor.layer_norm,
            stack_size=ssd_augm_config.beliefs_predictor.stack_size,
            kernel_size=ssd_augm_config.beliefs_predictor.kernel_size,
            filters=ssd_augm_config.beliefs_predictor.filters)
    elif ssd_augm_config.beliefs_predictor.predictor == 'hybrid_task_cascade':
        ssd_augmentation_predictor = ht_predictor.HTPredictor(
            is_training=is_training,
            layer_norm=ssd_augm_config.beliefs_predictor.layer_norm,
            stack_size=ssd_augm_config.beliefs_predictor.stack_size,
            kernel_size=ssd_augm_config.beliefs_predictor.kernel_size,
            filters=ssd_augm_config.beliefs_predictor.filters)
    elif ssd_augm_config.beliefs_predictor.predictor == 'u_net_2branches':
        ssd_augmentation_predictor = u_net_predictor_2branches_softmax_relu.UNet2branchesPredictor(
            is_training=is_training,
            layer_norm=ssd_augm_config.beliefs_predictor.layer_norm,
            stack_size=ssd_augm_config.beliefs_predictor.stack_size,
            kernel_size=ssd_augm_config.beliefs_predictor.kernel_size,
            filters=ssd_augm_config.beliefs_predictor.filters)
    elif ssd_augm_config.beliefs_predictor.predictor == 'sequential':
        ssd_augmentation_predictor = sequential_2branches.Sequential2branchesPredictor(
            is_training=is_training,
            layer_norm=ssd_augm_config.beliefs_predictor.layer_norm,
            stack_size=ssd_augm_config.beliefs_predictor.stack_size,
            kernel_size=ssd_augm_config.beliefs_predictor.kernel_size,
            filters=ssd_augm_config.beliefs_predictor.filters)
    elif ssd_augm_config.beliefs_predictor.predictor == 'shared_encoder':
        ssd_augmentation_predictor = shared_encoder_predictor.SharedEncoderPredictor(
            is_training=is_training,
            layer_norm=ssd_augm_config.beliefs_predictor.layer_norm,
            stack_size=ssd_augm_config.beliefs_predictor.stack_size,
            kernel_size=ssd_augm_config.beliefs_predictor.kernel_size,
            filters=ssd_augm_config.beliefs_predictor.filters)
    else:
        raise RuntimeError('unknown predictor %s for augmentation branch' % ssd_augm_config.beliefs_predictor.predictor)

    image_resizer_fn = image_resizer_builder.build(ssd_augm_config.image_resizer)
    non_max_suppression_fn, score_conversion_fn = post_processing_builder.build(
        ssd_augm_config.post_processing)
    (classification_loss, localization_loss, classification_weight,
     localization_weight, hard_example_miner, random_example_sampler,
     expected_loss_weights_fn) = losses_builder.build(ssd_augm_config.loss)
    normalize_loss_by_num_matches = ssd_augm_config.normalize_loss_by_num_matches
    normalize_loc_loss_by_codesize = ssd_augm_config.normalize_loc_loss_by_codesize
    # specific_threshold = ssd_augm_config.specific_threshold
    # threshold_offset = ssd_augm_config.threshold_offset
    # increse_small_object_size = ssd_augm_config.increse_small_object_size

    equalization_loss_config = ops.EqualizationLossConfig(
        weight=ssd_augm_config.loss.equalization_loss.weight,
        exclude_prefixes=ssd_augm_config.loss.equalization_loss.exclude_prefixes)

    target_assigner_instance = target_assigner.TargetAssigner(
        region_similarity_calculator,
        matcher,
        box_coder,
        negative_class_weight=negative_class_weight
        # increse_small_object_size=increse_small_object_size,
        # specific_threshold=specific_threshold,
        # threshold_offset=threshold_offset
    )

    ssd_augm_meta_arch_fn = ssd_augmentation_sharedEncoder_meta_arch.SSDAugmentationSharedEncoderMetaArch
    kwargs = {}

    return ssd_augm_meta_arch_fn(
        is_training=is_training,
        anchor_generator=anchor_generator,
        box_predictor=ssd_box_predictor,
        augmentation_predictor=ssd_augmentation_predictor,
        factor_loss_fused_bel_O=ssd_augm_config.augmentation_branch.factor_loss_fused_bel_O,
        factor_loss_fused_bel_F=ssd_augm_config.augmentation_branch.factor_loss_fused_bel_F,
        factor_loss_fused_zmax_det=ssd_augm_config.augmentation_branch.factor_loss_fused_zmax_det,
        factor_loss_fused_obs_zmin=ssd_augm_config.augmentation_branch.factor_loss_fused_obs_zmin,
        factor_loss_augm=ssd_augm_config.augmentation_branch.factor_loss_augm,
        box_coder=box_coder,
        feature_extractor=feature_extractor,
        encode_background_as_zeros=encode_background_as_zeros,
        image_resizer_fn=image_resizer_fn,
        non_max_suppression_fn=non_max_suppression_fn,
        score_conversion_fn=score_conversion_fn,
        use_uncertainty_weighting_loss=ssd_augm_config.loss.use_uncertainty_weighting_loss,
        classification_loss=classification_loss,
        localization_loss=localization_loss,
        classification_loss_weight=classification_weight,
        localization_loss_weight=localization_weight,
        normalize_loss_by_num_matches=normalize_loss_by_num_matches,
        hard_example_miner=hard_example_miner,
        target_assigner_instance=target_assigner_instance,
        add_summaries=add_summaries,
        normalize_loc_loss_by_codesize=normalize_loc_loss_by_codesize,
        freeze_batchnorm=ssd_augm_config.freeze_batchnorm,
        inplace_batchnorm_update=ssd_augm_config.inplace_batchnorm_update,
        add_background_class=ssd_augm_config.add_background_class,
        explicit_background_class=ssd_augm_config.explicit_background_class,
        random_example_sampler=random_example_sampler,
        expected_loss_weights_fn=expected_loss_weights_fn,
        use_confidences_as_targets=ssd_augm_config.use_confidences_as_targets,
        implicit_example_weight=ssd_augm_config.implicit_example_weight,
        equalization_loss_config=equalization_loss_config,
        **kwargs)

def _build_ssd_augmentation_hybridSeq_model(ssd_augm_config, is_training, add_summaries, num_input_channels,
                                  input_features):
    """Builds an SSD detection model based on the model config.

    Args:
      ssd_augmentation_config: A ssd.proto object containing the config for the desired
        SSDMetaArch.
      is_training: True if this model is being built for training purposes.
      add_summaries: Whether to add tf summaries in the model.
    Returns:
      SSDMetaArch based on the config.

    Raises:
      ValueError: If ssd_augm_config.type is not recognized (i.e. not registered in
        model_class_map).
    """
    num_classes = ssd_augm_config.num_classes

    # Feature extractor
    ssd_augm_config.feature_extractor.fpn.use_full_feature_extractor = False
    feature_extractor = _build_ssd_feature_extractor(
        feature_extractor_config=ssd_augm_config.feature_extractor,
        freeze_batchnorm=ssd_augm_config.freeze_batchnorm,
        is_training=is_training,
        input_features=input_features,
        num_input_channels=num_input_channels)

    box_coder = box_coder_builder.build(ssd_augm_config.box_coder)
    matcher = matcher_builder.build(ssd_augm_config.matcher)
    region_similarity_calculator = sim_calc.build(
        ssd_augm_config.similarity_calculator)
    encode_background_as_zeros = ssd_augm_config.encode_background_as_zeros
    negative_class_weight = ssd_augm_config.negative_class_weight
    anchor_generator = anchor_generator_builder.build(
        ssd_augm_config.anchor_generator

        # ssd_augm_config.feature_extractor.include_root_block,
        # ssd_augm_config.feature_extractor.root_downsampling_rate, ssd_augm_config.feature_extractor.type,
        # ssd_augm_config.feature_extractor.store_non_strided_activations
    )

    if feature_extractor.is_keras_model:
        ssd_box_predictor = box_predictor_builder.build_keras(
            hyperparams_fn=hyperparams_builder.KerasLayerHyperparams,
            freeze_batchnorm=ssd_augm_config.freeze_batchnorm,
            inplace_batchnorm_update=False,
            num_predictions_per_location_list=anchor_generator
                .num_anchors_per_location(),
            box_predictor_config=ssd_augm_config.box_predictor,
            is_training=is_training,
            num_classes=num_classes,
            add_background_class=ssd_augm_config.add_background_class)
    else:
        ssd_box_predictor = box_predictor_builder.build(
            hyperparams_builder.build, ssd_augm_config.box_predictor, is_training,
            num_classes, ssd_augm_config.add_background_class)

    ## Add augmentation network
    if ssd_augm_config.beliefs_predictor.predictor == 'u_net':
        ssd_augmentation_predictor = u_net_predictor.UNetPredictor(
            is_training=is_training,
            layer_norm=ssd_augm_config.beliefs_predictor.layer_norm,
            stack_size=ssd_augm_config.beliefs_predictor.stack_size,
            kernel_size=ssd_augm_config.beliefs_predictor.kernel_size,
            filters=ssd_augm_config.beliefs_predictor.filters)
    elif ssd_augm_config.beliefs_predictor.predictor == 'upsampling':
        ssd_augmentation_predictor = upsampling_predictor.UpsamplingPredictor(
            is_training=is_training,
            layer_norm=ssd_augm_config.beliefs_predictor.layer_norm,
            stack_size=ssd_augm_config.beliefs_predictor.stack_size,
            kernel_size=ssd_augm_config.beliefs_predictor.kernel_size,
            filters=ssd_augm_config.beliefs_predictor.filters)
    elif ssd_augm_config.beliefs_predictor.predictor == 'hybrid_task_cascade':
        ssd_augmentation_predictor = ht_predictor.HTPredictor(
            is_training=is_training,
            layer_norm=ssd_augm_config.beliefs_predictor.layer_norm,
            stack_size=ssd_augm_config.beliefs_predictor.stack_size,
            kernel_size=ssd_augm_config.beliefs_predictor.kernel_size,
            filters=ssd_augm_config.beliefs_predictor.filters)
    elif ssd_augm_config.beliefs_predictor.predictor == 'u_net_2branches':
        ssd_augmentation_predictor = u_net_predictor_2branches_softmax_relu.UNet2branchesPredictor(
            is_training=is_training,
            layer_norm=ssd_augm_config.beliefs_predictor.layer_norm,
            stack_size=ssd_augm_config.beliefs_predictor.stack_size,
            kernel_size=ssd_augm_config.beliefs_predictor.kernel_size,
            filters=ssd_augm_config.beliefs_predictor.filters)
    elif ssd_augm_config.beliefs_predictor.predictor == 'sequential':
        ssd_augmentation_predictor = sequential_2branches.Sequential2branchesPredictor(
            is_training=is_training,
            layer_norm=ssd_augm_config.beliefs_predictor.layer_norm,
            stack_size=ssd_augm_config.beliefs_predictor.stack_size,
            kernel_size=ssd_augm_config.beliefs_predictor.kernel_size,
            filters=ssd_augm_config.beliefs_predictor.filters)
    elif ssd_augm_config.beliefs_predictor.predictor == 'shared_encoder':
        ssd_augmentation_predictor = shared_encoder_predictor.SharedEncoderPredictor(
          is_training=is_training,
          layer_norm=ssd_augm_config.beliefs_predictor.layer_norm,
          stack_size=ssd_augm_config.beliefs_predictor.stack_size,
          kernel_size=ssd_augm_config.beliefs_predictor.kernel_size,
          filters=ssd_augm_config.beliefs_predictor.filters)
    else:
        raise RuntimeError('unknown predictor %s for augmentation branch' % ssd_augm_config.beliefs_predictor.predictor)

    image_resizer_fn = image_resizer_builder.build(ssd_augm_config.image_resizer)
    non_max_suppression_fn, score_conversion_fn = post_processing_builder.build(
        ssd_augm_config.post_processing)
    (classification_loss, localization_loss, classification_weight,
     localization_weight, hard_example_miner, random_example_sampler,
     expected_loss_weights_fn) = losses_builder.build(ssd_augm_config.loss)
    normalize_loss_by_num_matches = ssd_augm_config.normalize_loss_by_num_matches
    normalize_loc_loss_by_codesize = ssd_augm_config.normalize_loc_loss_by_codesize
    # specific_threshold = ssd_augm_config.specific_threshold
    # threshold_offset = ssd_augm_config.threshold_offset
    # increse_small_object_size = ssd_augm_config.increse_small_object_size

    equalization_loss_config = ops.EqualizationLossConfig(
        weight=ssd_augm_config.loss.equalization_loss.weight,
        exclude_prefixes=ssd_augm_config.loss.equalization_loss.exclude_prefixes)

    target_assigner_instance = target_assigner.TargetAssigner(
        region_similarity_calculator,
        matcher,
        box_coder,
        negative_class_weight=negative_class_weight
        # increse_small_object_size=increse_small_object_size,
        # specific_threshold=specific_threshold,
        # threshold_offset=threshold_offset
    )

    ssd_augm_meta_arch_fn = ssd_augmentation_hybridSeq_meta_arch.SSDAugmentationHybridSeqMetaArch
    kwargs = {}

    return ssd_augm_meta_arch_fn(
        is_training=is_training,
        anchor_generator=anchor_generator,
        box_predictor=ssd_box_predictor,
        augmentation_predictor=ssd_augmentation_predictor,
        factor_loss_fused_bel_O=ssd_augm_config.augmentation_branch.factor_loss_fused_bel_O,
        factor_loss_fused_bel_F=ssd_augm_config.augmentation_branch.factor_loss_fused_bel_F,
        factor_loss_fused_zmax_det=ssd_augm_config.augmentation_branch.factor_loss_fused_zmax_det,
        factor_loss_fused_obs_zmin=ssd_augm_config.augmentation_branch.factor_loss_fused_obs_zmin,
        factor_loss_augm=ssd_augm_config.augmentation_branch.factor_loss_augm,
        box_coder=box_coder,
        feature_extractor=feature_extractor,
        encode_background_as_zeros=encode_background_as_zeros,
        image_resizer_fn=image_resizer_fn,
        non_max_suppression_fn=non_max_suppression_fn,
        score_conversion_fn=score_conversion_fn,
        use_uncertainty_weighting_loss=ssd_augm_config.loss.use_uncertainty_weighting_loss,
        classification_loss=classification_loss,
        localization_loss=localization_loss,
        classification_loss_weight=classification_weight,
        localization_loss_weight=localization_weight,
        normalize_loss_by_num_matches=normalize_loss_by_num_matches,
        hard_example_miner=hard_example_miner,
        target_assigner_instance=target_assigner_instance,
        add_summaries=add_summaries,
        normalize_loc_loss_by_codesize=normalize_loc_loss_by_codesize,
        freeze_batchnorm=ssd_augm_config.freeze_batchnorm,
        inplace_batchnorm_update=ssd_augm_config.inplace_batchnorm_update,
        add_background_class=ssd_augm_config.add_background_class,
        explicit_background_class=ssd_augm_config.explicit_background_class,
        random_example_sampler=random_example_sampler,
        expected_loss_weights_fn=expected_loss_weights_fn,
        use_confidences_as_targets=ssd_augm_config.use_confidences_as_targets,
        implicit_example_weight=ssd_augm_config.implicit_example_weight,
        equalization_loss_config=equalization_loss_config,
        **kwargs)