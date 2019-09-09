"""Beliefs Estimation using U-Net."""

import tensorflow as tf

import tensorflow.contrib.layers as clayer

from object_detection.core import beliefs_predictor

BELIEF_O_PREDICTION = beliefs_predictor.BELIEF_O_PREDICTION
BELIEF_F_PREDICTION = beliefs_predictor.BELIEF_F_PREDICTION


class UpsamplingPredictor(beliefs_predictor.BeliefPredictor):  #
    """U Net Predictor with weight sharing."""

    def __init__(self,
                 is_training,
                 layer_norm,
                 stack_size,
                 kernel_size,
                 filters):
        super(UpsamplingPredictor, self).__init__(is_training)
        self._layer_norm = layer_norm
        self._stack_size = stack_size
        self._kernel_size = kernel_size
        self._filters = filters
        # self._depth = depth

    def _conv_bn_relu(self, x, filters, ksize, stride):

        x = tf.layers.conv2d(x, filters=filters, kernel_size=ksize, strides=stride, padding='same')
        if self._layer_norm:  # todo fragen where to pass
            x = clayer.layer_norm(x, scale=False)
        else:
            x = tf.layers.BatchNormalization(x)
        x = tf.nn.relu(x)

        return x

    def _unet_block(self, x, filter_num, filters, training, ksize, stack_size, name):
        with tf.variable_scope(name):
            for i in range(stack_size):
                with tf.variable_scope('conv_%i' % i):
                    x = self._conv_bn_relu(x, filters=filters, ksize=ksize, stride=1)
            return x

    def _create_net(self, x, ksize, filter_num, stack_size, outputs_channels):
        x = self._unet_block(x, filters=filter_num, stack_size=stack_size, ksize=ksize,
                             training=self._is_training, name='4x_conv_3x3')
        x = tf.Print(x, [x], 'after_4x_conv_3x3', summarize=15)
        #
        End
        with tf.variable_scope("end"):
            x = tf.layers.conv2d(x, filters=outputs_channels, kernel_size=self._kernel_size, strides=1, padding='same',
                                 name='conv')
            x = tf.nn.relu(x)

        return x

    def _create_upsampling_net(self, x, total_upsampling_level, outputs_channels):
        # x = tf.Print(x, [x], 'img_features[0]:', summarize=15)
        for level in range(total_upsampling_level):
            x = tf.layers.conv2d_transpose(x, filters=int(self._filters / pow(2, level + 2)), kernel_size=1, strides=2,
                                           padding='same', name='aug_transpose%d' % level)
            # x = tf.Print(x, [x], 'after_1_conv2dTranspose:', summarize=15)
            # x = tf.layers.conv2d(x, filters=int(self._filters / 4), kernel_size=1, training=self._is_training, name='aug_1x1_1')
            x = self._unet_block(x, filters=int(self._filters / pow(2, level + 2)), stack_size=1, ksize=3,
                                 training=self._is_training, name="dec_1")
            # x = tf.Print(x, [x], 'after_1_%dstacked_%dx%dconv:first10:' % (self._stack_size, self._filters, self._filters)
            #              , summarize=15)

        # End
        # with tf.variable_scope("end"):
        #     x = tf.layers.conv2d(x, filters=outputs_channels, kernel_size=self._kernel_size, strides=1, padding='same',
        #                          name='conv')
        x = tf.nn.relu(x)

        return x

    def _predict(self, image_features):
        upsampled_level1 = image_features[0]
        upsampled_level2 = self._create_upsampling_net(image_features[1], 1)
        upsampled_level3 = self._create_upsampling_net(image_features[2], 2)
        upsampled_level4 = self._create_upsampling_net(image_features[3], 3)
        concatenated = tf.concat([upsampled_level1, upsampled_level2, upsampled_level3, upsampled_level4], 3,
                                 name='concatenated')

        output = self._create_net(concatenated, ksize=3, filter_num=8, stack_size=4, outputs_channels=2)

        # pred_bel_F = tf.slice(output, begin=[0, 0, 0, 0], size=[-1, -1, -1, 1])
        # pred_bel_O = tf.slice(output, begin=[0, 0, 0, 1], size=[-1, -1, -1, 1])
        pred_bel_F = tf.expand_dims(output[:, :, :, 0], axis=3)
        pred_bel_O = tf.expand_dims(output[:, :, :, 1], axis=3)

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
            BELIEF_F_PREDICTION: []
        }
        # predictions[BELIEF_O_PREDICTION] = pred_bel_O_clamped
        # predictions[BELIEF_F_PREDICTION] = pred_bel_F_clamped
        predictions[BELIEF_O_PREDICTION] = pred_bel_O
        predictions[BELIEF_F_PREDICTION] = pred_bel_F
        # predictions[Z_XX] = # todo

        return predictions
