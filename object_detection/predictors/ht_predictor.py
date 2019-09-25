"""Beliefs Estimation using U-Net."""

import tensorflow as tf

import tensorflow.contrib.layers as clayer

from object_detection.core import beliefs_predictor
from object_detection.utils import shape_utils

BELIEF_O_PREDICTION = beliefs_predictor.BELIEF_O_PREDICTION
BELIEF_F_PREDICTION = beliefs_predictor.BELIEF_F_PREDICTION
Z_MAX_DETECTIONS_PREDICTION = beliefs_predictor.Z_MAX_DETECTIONS_PREDICTION
Z_MIN_OBSERVATIONS_PREDICTION = beliefs_predictor.Z_MIN_OBSERVATIONS_PREDICTION


class HTPredictor(beliefs_predictor.BeliefPredictor):  #
    """U Net Predictor with weight sharing."""

    def __init__(self,
                 is_training,
                 layer_norm,
                 stack_size,
                 kernel_size,
                 filters):
        super(HTPredictor, self).__init__(is_training)
        self._layer_norm = layer_norm
        self._stack_size = stack_size
        self._kernel_size = kernel_size
        self._filters = filters
        # self._depth = depth

    def _conv_bn_relu(self, x, filters, ksize, stride):
        x = tf.layers.conv2d(x, filters=filters, kernel_size=ksize, strides=stride, padding='same')
        if self._layer_norm:
            x = clayer.layer_norm(x, scale=False)
        # else:
        #     x = tf.layers.batch_normalization(x)
        x = tf.nn.relu(x)

        return x

    def _conv_block(self, x, filters, training, ksize, stack_size, name):
        with tf.variable_scope("block_%s" % name):
            for i in range(stack_size):
                with tf.variable_scope('conv_%i' % i):
                    x = self._conv_bn_relu(x, filters=filters, ksize=ksize, stride=1)
            return x

    def _conv3x3_net(self, x, ksize, stackSize, filter_num_start, outputs_channels):
        x = self._conv_block(x, filters=filter_num_start, stack_size=stackSize, ksize=ksize,
                             training=self._is_training, name='conv3x3_1')
        x = self._conv_block(x, filters=filter_num_start / 2, stack_size=stackSize, ksize=ksize,
                             training=self._is_training, name='conv3x3_2')
        x = self._conv_block(x, filters=filter_num_start / 4, stack_size=stackSize, ksize=ksize,
                             training=self._is_training, name='conv3x3_3')
        x = self._conv_block(x, filters=filter_num_start / 8, stack_size=stackSize, ksize=ksize,
                             training=self._is_training, name='conv3x3_4')
        x = self._conv_block(x, filters=filter_num_start / 16, stack_size=stackSize, ksize=ksize,
                             training=self._is_training, name='conv3x3_5')
        # x = tf.Print(x, [x], 'after_4x_conv_3x3', summarize=15)
        #
        # End
        with tf.variable_scope("end"):
            x = tf.layers.conv2d(x, filters=outputs_channels, kernel_size=1, strides=1, padding='same')
            x = tf.nn.relu(x)

        return x

    def _create_upsampling_net(self, x, total_upsampling_level):  # outputs_channels

        x = tf.layers.conv2d_transpose(x, filters=int(self._filters / pow(2, total_upsampling_level)), kernel_size=3,
                                       strides=pow(2, total_upsampling_level),
                                       padding='same', name='aug_level%d_transpose' % total_upsampling_level)
        x = tf.nn.relu(x)

        # with tf.variable_scope("end"):
        #     x = tf.layers.conv2d(x, filters=outputs_channels, kernel_size=1, strides=1, padding='same')
        # x = tf.nn.relu(x)
        return x

    def _predict(self, image_features, preprocessed_input, scope=None):
        upsampled_level1 = self._create_upsampling_net(image_features[0], 1)
        # print('shape_utils.combined_static_and_dynamic_shape(upsampled_level1)')
        # print(shape_utils.combined_static_and_dynamic_shape(upsampled_level1))
        upsampled_level2 = self._create_upsampling_net(image_features[1], 2)
        upsampled_level3 = self._create_upsampling_net(image_features[2], 3)
        upsampled_level4 = self._create_upsampling_net(image_features[3], 4)
        # print('shape_utils.combined_static_and_dynamic_shape(upsampled_level4)')
        # print(shape_utils.combined_static_and_dynamic_shape(upsampled_level4))

        concatenated = tf.concat([preprocessed_input, upsampled_level1, upsampled_level2, upsampled_level3, upsampled_level4], 3,
                                 name='concatenated')
        print('shape_utils.combined_static_and_dynamic_shape(concatenated)')
        print(shape_utils.combined_static_and_dynamic_shape(concatenated))

        output = self._conv3x3_net(concatenated, ksize=3, stackSize=3, filter_num_start=128, outputs_channels=4)

        # print('shape_utils.combined_static_and_dynamic_shape(output)')
        # print(shape_utils.combined_static_and_dynamic_shape(output))

        pred_bel_F = tf.expand_dims(output[:, :, :, 0], axis=3)
        pred_bel_O = tf.expand_dims(output[:, :, :, 1], axis=3)

        # print('shape_utils.combined_static_and_dynamic_shape(pred_bel_O)')
        # print(shape_utils.combined_static_and_dynamic_shape(pred_bel_O))

        pred_z_max_detections = tf.expand_dims(output[:, :, :, 2], axis=3)
        pred_z_min_observations = tf.expand_dims(output[:, :, :, 3], axis=3)

        # tf.summary.image('predicted_bel_O_before_clamping', pred_bel_O)
        # tf.summary.image('predicted_bel_F_before_clamping', pred_bel_F)

        # with tf.name_scope("clamping"):
        #     pred_bel_F_clamped = tf.maximum(tf.minimum(pred_bel_F, 1), 0)
        #     pred_bel_O_clamped = tf.maximum(tf.minimum(pred_bel_O, 1), 0)
        # pred_bel_F_clamped = tf.clamp
        # pred_bel_F, 1), 0)
        # pred_bel_O_clamped = tf.maximum(tf.minimum(pred_bel_O, 1), 0)

        predictions = {
            BELIEF_O_PREDICTION: [],
            BELIEF_F_PREDICTION: [],
            Z_MAX_DETECTIONS_PREDICTION: [],
            Z_MIN_OBSERVATIONS_PREDICTION: []
        }
        # predictions[BELIEF_O_PREDICTION] = pred_bel_O_clamped
        # predictions[BELIEF_F_PREDICTION] = pred_bel_F_clamped
        predictions[BELIEF_O_PREDICTION] = pred_bel_O
        predictions[BELIEF_F_PREDICTION] = pred_bel_F
        predictions[Z_MAX_DETECTIONS_PREDICTION] = pred_z_max_detections
        predictions[Z_MIN_OBSERVATIONS_PREDICTION] = pred_z_min_observations

        return predictions
