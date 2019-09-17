"""Beliefs Estimation using U-Net."""

import tensorflow as tf

import tensorflow.contrib.layers as clayer

from object_detection.core import beliefs_predictor

BELIEF_O_PREDICTION = beliefs_predictor.BELIEF_O_PREDICTION
BELIEF_F_PREDICTION = beliefs_predictor.BELIEF_F_PREDICTION
Z_MAX_DETECTIONS_PREDICTION = beliefs_predictor.Z_MAX_DETECTIONS_PREDICTION
Z_MIN_OBSERVATIONS_PREDICTION = beliefs_predictor.Z_MIN_OBSERVATIONS_PREDICTION


class UNetPredictor(beliefs_predictor.BeliefPredictor):
    """U Net Predictor with weight sharing."""

    def __init__(self,
                 is_training,
                 layer_norm,
                 stack_size,
                 kernel_size,
                 filters):
        super(UNetPredictor, self).__init__(is_training)
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
        # x = tf.layers.BatchNormalization(x)
        x = tf.nn.relu(x)

        return x

    def _conv_block(self, x, filters, stack_size, ksize, name):
        with tf.variable_scope(name):
            for i in range(stack_size):
                with tf.variable_scope('conv_%i' % i):
                    x = self._conv_bn_relu(x, filters=filters, ksize=ksize, stride=1)
            return x

    def _create_net(self, x, short_cut, outputs_channels):
        x = tf.Print(x, [x], 'img_features[0]:', summarize=15)

        x = self._conv_block(x, filters=int(self._filters / 2), stack_size=1, ksize=1,
                             name="augm_conv_relu_before_transpose")

        x = tf.layers.conv2d_transpose(x, filters=int(self._filters / 4), kernel_size=self._kernel_size, strides=2,
                                       padding='same', name='aug_transpose1')


        x = tf.concat([x, short_cut], 3, name='concate_original_dim')

        x = self._conv_block(x, filters=int(self._filters / 4), stack_size=self._stack_size, ksize=self._kernel_size,
                             name="augm_conv_block")

        with tf.variable_scope("end"):
            x = tf.layers.conv2d(x, filters=outputs_channels, kernel_size=self._kernel_size, strides=1, padding='same',
                                 name='augm_conv_end')
            x = tf.nn.relu(x)

        return x

    def _predict(self, image_features, preprocessed_input, scope=None):

        input = image_features[0]

        # Create Unet
        output = self._create_net(input, preprocessed_input, outputs_channels=4)

        pred_bel_F = tf.expand_dims(output[:, :, :, 0], axis=3)
        pred_bel_O = tf.expand_dims(output[:, :, :, 1], axis=3)
        pred_z_max_detections = tf.expand_dims(output[:, :, :, 2], axis=3)
        pred_z_min_observations = tf.expand_dims(output[:, :, :, 3], axis=3)

        predictions = {
            BELIEF_O_PREDICTION: [],
            BELIEF_F_PREDICTION: [],
            Z_MAX_DETECTIONS_PREDICTION: [],
            Z_MIN_OBSERVATIONS_PREDICTION: []
        }

        predictions[BELIEF_O_PREDICTION] = pred_bel_O
        predictions[BELIEF_F_PREDICTION] = pred_bel_F
        predictions[Z_MAX_DETECTIONS_PREDICTION] = pred_z_max_detections
        predictions[Z_MIN_OBSERVATIONS_PREDICTION] = pred_z_min_observations


        return predictions
