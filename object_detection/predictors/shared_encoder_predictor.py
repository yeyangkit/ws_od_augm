import tensorflow as tf

import tensorflow.contrib.layers as clayer

from object_detection.core import beliefs_predictor


BELIEF_O_PREDICTION = beliefs_predictor.BELIEF_O_PREDICTION
BELIEF_F_PREDICTION = beliefs_predictor.BELIEF_F_PREDICTION
Z_MAX_DETECTIONS_PREDICTION = beliefs_predictor.Z_MAX_DETECTIONS_PREDICTION
Z_MIN_OBSERVATIONS_PREDICTION = beliefs_predictor.Z_MIN_OBSERVATIONS_PREDICTION
BELIEF_U_PREDICTION = beliefs_predictor.BELIEF_U_PREDICTION
Z_MIN_DETECTIONS_PREDICTION = beliefs_predictor.Z_MIN_DETECTIONS_PREDICTION
DETECTIONS_DRIVINGCORRIDOR_PREDICTION = beliefs_predictor.DETECTIONS_DRIVINGCORRIDOR_PREDICTION
INTENSITY_PREDICTION = beliefs_predictor.INTENSITY_PREDICTION

class SharedEncoderPredictor(beliefs_predictor.BeliefPredictor):  #
    """U Net Predictor with weight sharing."""
    def __init__(self,
                 is_training,
                 layer_norm,
                 stack_size,
                 kernel_size,
                 filters):
        super(SharedEncoderPredictor, self).__init__(is_training)
        self._layer_norm = layer_norm
        self._stack_size = stack_size
        self._final_kernel_size = kernel_size
        self._detectionFM_filters = filters
        # self._depth = depth


    def _multiResUnet_block(self, x, depth, name, stack_size=1):
        """
        reference https://arxiv.org/abs/1902.04049
        """
        for i in range(stack_size):
            x = tf.layers.conv2d(x, depth, [1, 1], name="multiRes_Block_{}_{}_bottleneckIn".format(name, i))
            x1 = tf.layers.conv2d(x, depth, [3, 3], name="multiRes_Block_{}_{}_inception3x3conv".format(name, i), padding='same')
            x2 = tf.layers.conv2d(x1, depth, [3, 3], name="multiRes_Block_{}_{}_inception5x5conv".format(name, i), padding='same')
            x3 = tf.layers.conv2d(x2, depth, [3, 3], name="multiRes_Block_{}_{}_inception7x7conv".format(name, i), padding='same')
            x4 = tf.concat((x1, x2, x3), axis=3, name="multiRes_Block_{}_{}_concat".format(name, i))
            x4 = tf.layers.conv2d(x4, depth, [1, 1], name="multiRes_Block_{}_{}_bottleneckOut".format(name, i))
            x += x4
        return tf.nn.relu(x, name="multiRes_Block_{}_relu".format(name))


    def _multiResUnet_resPath(self, x, depth, name, stack_size=1):
      """
      reference https://arxiv.org/abs/1902.04049
      """
      for i in range(stack_size):
        x1 = tf.layers.conv2d(x, depth, [3, 3], name="multiRes_shortcut_{}_{}_3x3conv".format(name, i), padding='same')
        x2 = tf.layers.conv2d(x, depth, [1, 1], name="multiRes_shortcut_{}_{}_1x1conv".format(name, i))
        x = x1 + x2
      return x


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

    def _create_input_conv_net(self, preprocessed_input):
        x = self._multiResUnet_resPath(preprocessed_input, depth=int(self._detectionFM_filters), stack_size=1, name='preprocessed_input')
        return x    #   tf.nn.relu(x)

    def _create_unet_end(self, last_feature_maps_augm, short_cut, bels_outputs_channels, maps_outputs_channels):

        x = self._multiResUnet_block(last_feature_maps_augm, depth=int(self._detectionFM_filters/2), stack_size=self._stack_size, name='last_feature_maps_augm')

        x = tf.layers.conv2d_transpose(x, filters=int(self._detectionFM_filters / 2), kernel_size=3, strides=2,
                                       padding='same', name='augm_transpose')


        x = tf.concat([x, short_cut], 3, name='concate_original_dim')

        x1 = self._multiResUnet_block(x, depth=int(self._detectionFM_filters / 4), stack_size=self._stack_size,
                             name="augm_BELS_multiResUnet_block_after_transpose")


        with tf.variable_scope("augm_BELS_end"):
            x1 = tf.layers.conv2d(x1, filters=bels_outputs_channels, kernel_size=self._final_kernel_size, strides=1, padding='same',
                                 name='augm_BELS_conv_end_outputs')
            bels = tf.nn.softmax(x1)

        x2 = self._multiResUnet_block(x, depth=int(self._detectionFM_filters / 4), stack_size=self._stack_size,
                             name="augm_MAPS_multiResUnet_block_after_transpose")

        with tf.variable_scope("augm_MAPS_end"):
            x2 = tf.layers.conv2d(x2, filters=maps_outputs_channels, kernel_size=self._final_kernel_size, strides=1, padding='same',
                                 name='augm_MAPS_conv_end_outputs')
            maps = tf.nn.relu(x2)

        return bels, maps

    def _predict(self, image_features, preprocessed_input, scope=None):

        input = image_features[0]

        shortcut = self._create_input_conv_net(preprocessed_input)

        # Create Unet
        pred_bels, pred_maps = self._create_unet_end(input, shortcut, bels_outputs_channels=3, maps_outputs_channels=5)

        pred_bel_F = tf.expand_dims(pred_bels[:, :, :, 0], axis=3)
        pred_bel_O = tf.expand_dims(pred_bels[:, :, :, 1], axis=3)
        pred_bel_U = tf.expand_dims(pred_bels[:, :, :, 2], axis=3)

        pred_z_max_detections = tf.expand_dims(pred_maps[:, :, :, 0], axis=3)
        pred_z_min_observations = tf.expand_dims(pred_maps[:, :, :, 1], axis=3)
        pred_z_min_detections = tf.expand_dims(pred_maps[:, :, :, 2], axis=3)
        pred_detections_drivingCorridor = tf.expand_dims(pred_maps[:, :, :, 3], axis=3)
        pred_intensity = tf.expand_dims(pred_maps[:, :, :, 4], axis=3)

        predictions = {
            BELIEF_O_PREDICTION: [],
            BELIEF_F_PREDICTION: [],
            Z_MAX_DETECTIONS_PREDICTION: [],
            Z_MIN_OBSERVATIONS_PREDICTION: [],
            BELIEF_U_PREDICTION: [],
            Z_MIN_DETECTIONS_PREDICTION: [],
            DETECTIONS_DRIVINGCORRIDOR_PREDICTION: [],
            INTENSITY_PREDICTION: []
        }

        predictions[BELIEF_O_PREDICTION] = pred_bel_O
        predictions[BELIEF_F_PREDICTION] = pred_bel_F
        predictions[Z_MAX_DETECTIONS_PREDICTION] = pred_z_max_detections
        predictions[Z_MIN_OBSERVATIONS_PREDICTION] = pred_z_min_observations
        predictions[BELIEF_U_PREDICTION] = pred_bel_U
        predictions[Z_MIN_DETECTIONS_PREDICTION] = pred_z_min_detections
        predictions[DETECTIONS_DRIVINGCORRIDOR_PREDICTION] = pred_detections_drivingCorridor
        predictions[INTENSITY_PREDICTION] = pred_intensity


        return predictions

