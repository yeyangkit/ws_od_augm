# """Beliefs Estimation using U-Net."""
#
# import tensorflow as tf
#
# import tensorflow.contrib.layers as clayer
#
# from object_detection.core import beliefs_predictor
#
# BELIEF_O_PREDICTION = beliefs_predictor.BELIEF_O_PREDICTION
# BELIEF_F_PREDICTION = beliefs_predictor.BELIEF_F_PREDICTION
# Z_MAX_DETECTIONS_PREDICTION = beliefs_predictor.Z_MAX_DETECTIONS_PREDICTION
# Z_MIN_OBSERVATIONS_PREDICTION = beliefs_predictor.Z_MIN_OBSERVATIONS_PREDICTION
# BELIEF_U_PREDICTION = beliefs_predictor.BELIEF_U_PREDICTION
# Z_MIN_DETECTIONS_PREDICTION = beliefs_predictor.Z_MIN_DETECTIONS_PREDICTION
# DETECTIONS_DRIVINGCORRIDOR_PREDICTION = beliefs_predictor.DETECTIONS_DRIVINGCORRIDOR_PREDICTION
# INTENSITY_PREDICTION = beliefs_predictor.INTENSITY_PREDICTION
#
# class UNet2branchesPredictor(beliefs_predictor.BeliefPredictor):
#     """U Net Predictor with weight sharing."""
#
#     def __init__(self,
#                  is_training,
#                  layer_norm,
#                  stack_size,
#                  kernel_size,
#                  filters):
#         super(UNet2branchesPredictor, self).__init__(is_training)
#         self._layer_norm = layer_norm
#         self._stack_size = stack_size
#         self._final_kernel_size = kernel_size
#         self._detectionFM_filters = filters
#         # self._depth = depth
#
#     def _conv_bn_relu(self, x, filters, ksize, stride):
#
#         x = tf.layers.conv2d(x, filters=filters, kernel_size=ksize, strides=stride, padding='same')
#         if self._layer_norm:
#             x = clayer.layer_norm(x, scale=False)
#         # else:
#         # x = tf.layers.BatchNormalization(x)
#         x = tf.nn.relu(x)
#
#         return x
#
#     def _conv_block(self, x, filters, stack_size, ksize, name):
#         with tf.variable_scope(name):
#             for i in range(stack_size):
#                 with tf.variable_scope('conv_%i' % i):
#                     x = self._conv_bn_relu(x, filters=filters, ksize=ksize, stride=1)
#             return x
#
#     def _create_input_conv_net(self, preprocessed_input):
#         x = tf.layers.conv2d(preprocessed_input, filters=self._detectionFM_filters/2, kernel_size=1, strides=1,
#                              padding='same', name='preprocessed_input_start')
#
#         x = self._conv_block(x, filters=int(self._detectionFM_filters / 2), stack_size=self._stack_size, ksize=3,
#                              name="augm_input_conv_op_block")
#
#         with tf.variable_scope("shortcut"):
#             x = tf.layers.conv2d(x, filters=int(self._detectionFM_filters / 2), kernel_size=3, strides=1, padding='same',
#                                  name='augm_conv_end')
#             shortcut = tf.nn.relu(x)
#         return shortcut
#
#     def _create_unet_end(self, image_features, short_cut, bels_outputs_channels, maps_outputs_channels):
#         # x = tf.Print(x, [x], 'img_features[0]:', summarize=15)
#         x = tf.layers.conv2d(image_features, filters=self._detectionFM_filters, kernel_size=1, strides=1,padding='same',name='image_features_start')
#
#         x = self._conv_block(x, filters=int(self._detectionFM_filters), stack_size=self._stack_size, ksize=3,
#                              name="augm_conv_block1_before_transpose")
#
#         x = self._conv_block(x, filters=int(self._detectionFM_filters), stack_size=self._stack_size, ksize=3,
#                              name="augm_conv_block2_before_transpose")
#
#         x = tf.layers.conv2d_transpose(x, filters=int(self._detectionFM_filters / 2), kernel_size=3, strides=2,
#                                        padding='same', name='augm_transpose1')
#
#
#         x = tf.concat([x, short_cut], 3, name='concate_original_dim')
#
#         x1 = self._conv_block(x, filters=int(self._detectionFM_filters / 4), stack_size=self._stack_size, ksize=3,
#                              name="augm_BELS_conv_block1_after_transpose")
#
#         x1 = self._conv_block(x1, filters=int(self._detectionFM_filters / 4), stack_size=self._stack_size, ksize=3,
#                              name="augm_BELS_conv_block2_after_transpose")
#         with tf.variable_scope("augm_BELS_end"):
#             x1 = tf.layers.conv2d(x1, filters=bels_outputs_channels, kernel_size=self._final_kernel_size, strides=1, padding='same',
#                                  name='augm_BELS_conv_end_outputs')
#             bels = tf.nn.softmax(x1)
#
#         x2 = self._conv_block(x, filters=int(self._detectionFM_filters / 4), stack_size=self._stack_size, ksize=3,
#                              name="augm_MAPS_conv_block1_after_transpose")
#
#         x2 = self._conv_block(x2, filters=int(self._detectionFM_filters / 4), stack_size=self._stack_size, ksize=3,
#                              name="augm_MAPS_conv_block2_after_transpose")
#
#         with tf.variable_scope("augm_MAPS_end"):
#             x2 = tf.layers.conv2d(x2, filters=maps_outputs_channels, kernel_size=self._final_kernel_size, strides=1, padding='same',
#                                  name='augm_MAPS_conv_end_outputs')
#             maps = tf.nn.relu(x2)
#
#         return bels, maps
#
#     def _predict(self, image_features, preprocessed_input, scope=None):
#
#         input = image_features[0]
#
#         shortcut = self._create_input_conv_net(preprocessed_input)
#
#         # Create Unet
#         pred_bels, pred_maps = self._create_unet_end(input, shortcut, bels_outputs_channels=3, maps_outputs_channels=5)
#
#         pred_bel_F = tf.expand_dims(pred_bels[:, :, :, 0], axis=3)
#         pred_bel_O = tf.expand_dims(pred_bels[:, :, :, 1], axis=3)
#         pred_bel_U = tf.expand_dims(pred_bels[:, :, :, 2], axis=3)
#
#         pred_z_max_detections = tf.expand_dims(pred_maps[:, :, :, 0], axis=3)
#         pred_z_min_observations = tf.expand_dims(pred_maps[:, :, :, 1], axis=3)
#         pred_z_min_detections = tf.expand_dims(pred_maps[:, :, :, 2], axis=3)
#         pred_detections_drivingCorridor = tf.expand_dims(pred_maps[:, :, :, 3], axis=3)
#         pred_intensity = tf.expand_dims(pred_maps[:, :, :, 4], axis=3)
#
#         predictions = {
#             BELIEF_O_PREDICTION: [],
#             BELIEF_F_PREDICTION: [],
#             Z_MAX_DETECTIONS_PREDICTION: [],
#             Z_MIN_OBSERVATIONS_PREDICTION: [],
#             BELIEF_U_PREDICTION: [],
#             Z_MIN_DETECTIONS_PREDICTION: [],
#             DETECTIONS_DRIVINGCORRIDOR_PREDICTION: [],
#             INTENSITY_PREDICTION: []
#         }
#
#         predictions[BELIEF_O_PREDICTION] = pred_bel_O
#         predictions[BELIEF_F_PREDICTION] = pred_bel_F
#         predictions[Z_MAX_DETECTIONS_PREDICTION] = pred_z_max_detections
#         predictions[Z_MIN_OBSERVATIONS_PREDICTION] = pred_z_min_observations
#         predictions[BELIEF_U_PREDICTION] = pred_bel_U
#         predictions[Z_MIN_DETECTIONS_PREDICTION] = pred_z_min_detections
#         predictions[DETECTIONS_DRIVINGCORRIDOR_PREDICTION] = pred_detections_drivingCorridor
#         predictions[INTENSITY_PREDICTION] = pred_intensity
#
#
#         return predictions
"""Beliefs Estimation using U-Net."""

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

class UNet2branchesPredictor(beliefs_predictor.BeliefPredictor):
    """U Net Predictor with weight sharing."""

    def __init__(self,
                 is_training,
                 layer_norm,
                 stack_size,
                 kernel_size,
                 filters):
        super(UNet2branchesPredictor, self).__init__(is_training)
        self._layer_norm = layer_norm
        self._stack_size = stack_size
        self._final_kernel_size = kernel_size
        self._detectionFM_filters = filters
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

    def _create_input_conv_net(self, preprocessed_input):
        x = tf.layers.conv2d(preprocessed_input, filters=self._detectionFM_filters/2, kernel_size=1, strides=1,
                             padding='same', name='preprocessed_input_start')

        x = self._conv_block(x, filters=int(self._detectionFM_filters / 2), stack_size=self._stack_size, ksize=3,
                             name="augm_input_conv_op_block")

        with tf.variable_scope("shortcut"):
            x = tf.layers.conv2d(x, filters=int(self._detectionFM_filters / 2), kernel_size=3, strides=1, padding='same',
                                 name='augm_conv_end')
            shortcut = tf.nn.relu(x)
        return shortcut

    def _create_unet_end(self, image_features, short_cut, bels_outputs_channels, maps_outputs_channels):
        # x = tf.Print(x, [x], 'img_features[0]:', summarize=15)
        x = tf.layers.conv2d(image_features, filters=self._detectionFM_filters, kernel_size=1, strides=1,padding='same',name='image_features_start')

        x = self._conv_block(x, filters=int(self._detectionFM_filters), stack_size=self._stack_size, ksize=3,
                             name="augm_conv_block1_before_transpose")

        x = self._conv_block(x, filters=int(self._detectionFM_filters), stack_size=self._stack_size, ksize=3,
                             name="augm_conv_block2_before_transpose")

        x = tf.layers.conv2d_transpose(x, filters=int(self._detectionFM_filters / 2), kernel_size=3, strides=2,
                                       padding='same', name='augm_transpose1')


        x = tf.concat([x, short_cut], 3, name='concate_original_dim')

        x1 = self._conv_block(x, filters=int(self._detectionFM_filters / 4), stack_size=self._stack_size, ksize=3,
                             name="augm_BELS_conv_block1_after_transpose")

        x1 = self._conv_block(x1, filters=int(self._detectionFM_filters / 4), stack_size=self._stack_size, ksize=3,
                             name="augm_BELS_conv_block2_after_transpose")
        with tf.variable_scope("augm_BELS_end"):
            x1 = tf.layers.conv2d(x1, filters=bels_outputs_channels, kernel_size=self._final_kernel_size, strides=1, padding='same',
                                 name='augm_BELS_conv_end_outputs')
            bels = tf.nn.softmax(x1)

        x2 = self._conv_block(x, filters=int(self._detectionFM_filters / 4), stack_size=self._stack_size, ksize=3,
                             name="augm_MAPS_conv_block1_after_transpose")

        x2 = self._conv_block(x2, filters=int(self._detectionFM_filters / 4), stack_size=self._stack_size, ksize=3,
                             name="augm_MAPS_conv_block2_after_transpose")

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


        predictions[INTENSITY_PREDICTION] = pred_intensity

        predictions[BELIEF_O_PREDICTION] = pred_bel_O
        predictions[BELIEF_F_PREDICTION] = pred_bel_F
        predictions[Z_MAX_DETECTIONS_PREDICTION] = pred_z_max_detections
        predictions[Z_MIN_OBSERVATIONS_PREDICTION] = pred_z_min_observations
        predictions[BELIEF_U_PREDICTION] = pred_bel_U
        predictions[Z_MIN_DETECTIONS_PREDICTION] = pred_z_min_detections
        predictions[DETECTIONS_DRIVINGCORRIDOR_PREDICTION] = pred_detections_drivingCorridor


        return predictions



