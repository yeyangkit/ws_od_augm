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


class Sequential2branchesPredictor(beliefs_predictor.BeliefPredictor):
    """U Net Predictor with weight sharing."""

    def __init__(self,
                 is_training,
                 layer_norm,
                 stack_size,
                 kernel_size,
                 filters):
        super(Sequential2branchesPredictor, self).__init__(is_training)
        self._layer_norm = layer_norm
        self._stack_size = stack_size
        self._final_kernel_size = kernel_size
        self._min_filter = filters
        # self._depth = depth

    def _conv_bn_relu(self, x, filters, ksize, stride):
        x = tf.layers.conv2d(x, filters=filters, kernel_size=ksize, strides=stride, padding='same')
        x = tf.nn.relu(x)
        return x

    def _conv_block(self, x, filters, training, stack_size, ksize, name):
        with tf.variable_scope(name):
            for i in range(stack_size):
                with tf.variable_scope('augment_conv_%i' % i):
                    x = self._conv_bn_relu(x, filters=filters, ksize=ksize, stride=1)
            return x

    def _create_unet(self, x, f, kernel_size, stack_size, depth, training, maps_outputs_channels=4,
                     bels_outputs_channels=3):

        skips = []
        # level_outputs = []

        for i in range(depth):
            x = self._conv_block(x, filters=f * (2 ** i), training=training, stack_size=stack_size, ksize=3,
                                 name="augment_enc_%i" % i)
            skips.append(x)
            x = tf.layers.max_pooling2d(x, pool_size=2, strides=2, name='augment_pool_%i' % (i + 1))

        x = self._conv_block(x, filters=f * (2 ** (depth - 1)), training=training, stack_size=stack_size,
                             ksize=kernel_size,
                             name="augment_deep")
        # with tf.variable_scope("augment_deep_end"):
        #     x = tf.layers.conv2d(x, filters=output_channel, kernel_size=1, strides=1,
        #                          padding='same')
        # level_outputs.append(x)

        for i in reversed(range(depth)):
            with tf.variable_scope('augment_up_conv_%i' % (i + 1)):
                x = tf.layers.conv2d_transpose(x, filters=f * (2 ** i), kernel_size=kernel_size, strides=2,
                                               padding='same')
                # if FLAGS.layer_norm:
                #     x = clayer.layer_norm(x, scale=False)
                x = tf.nn.relu(x)
            x = tf.concat([skips.pop(), x], axis=3)

            x = self._conv_block(x, filters=f * (2 ** i), training=training, stack_size=stack_size, ksize=3,
                                 name="augment_dec_%i" % i)

            # with tf.variable_scope("augment_end_%i" % i):
            #     x = tf.layers.conv2d(x, filters=output_channel, kernel_size=1, strides=1,
            #                          padding='same')
            # level_outputs.append(x)
            # if i==0:
            #     tf.summary.image("augment_LARGEST_fMap", x, family="watcher")
        # print("level_outputs-------------------------------------------------------------------------------")
        # print(level_outputs)

        x1 = self._conv_block(x, filters=f, stack_size=self._stack_size, ksize=3,training=training,
                              name="augm_BELS_conv_block1_after_transpose")

        x1 = self._conv_block(x1, filters=f, stack_size=self._stack_size, ksize=3,training=training,
                              name="augm_BELS_conv_block2_after_transpose")
        with tf.variable_scope("augm_BELS_end"):
            x1 = tf.layers.conv2d(x1, filters=bels_outputs_channels, kernel_size=self._final_kernel_size, strides=1,
                                  padding='same',
                                  name='augm_BELS_conv_end_outputs')
            bels = tf.nn.softmax(x1)

        x2 = self._conv_block(x, filters=f, stack_size=self._stack_size, ksize=3,training=training,
                              name="augm_MAPS_conv_block1_after_transpose")

        x2 = self._conv_block(x2, filters=f, stack_size=self._stack_size, ksize=3,training=training,
                              name="augm_MAPS_conv_block2_after_transpose")

        with tf.variable_scope("augm_MAPS_end"):
            x2 = tf.layers.conv2d(x2, filters=maps_outputs_channels, kernel_size=self._final_kernel_size, strides=1,
                                  padding='same',
                                  name='augm_MAPS_conv_end_outputs')
            maps = tf.nn.relu(x2)

        return bels, maps

    def _predict(self, image_features, preprocessed_input, scope=None):

        # Create Unet
        pred_bels, pred_maps = self._create_unet(preprocessed_input, training=True, depth=5, f=self._min_filter, kernel_size=4,
                                                 stack_size=self._stack_size, maps_outputs_channels=4,
                                                 bels_outputs_channels=3)

        pred_bel_F = tf.expand_dims(pred_bels[:, :, :, 0], axis=3)
        pred_bel_O = tf.expand_dims(pred_bels[:, :, :, 1], axis=3)
        pred_bel_U = tf.expand_dims(pred_bels[:, :, :, 2], axis=3)

        pred_z_max_detections = tf.expand_dims(pred_maps[:, :, :, 0], axis=3)
        pred_z_min_observations = tf.expand_dims(pred_maps[:, :, :, 1], axis=3)
        pred_z_min_detections = tf.expand_dims(pred_maps[:, :, :, 2], axis=3)
        pred_detections_drivingCorridor = tf.expand_dims(pred_maps[:, :, :, 3], axis=3)

        predictions = {
            BELIEF_O_PREDICTION: [],
            BELIEF_F_PREDICTION: [],
            Z_MAX_DETECTIONS_PREDICTION: [],
            Z_MIN_OBSERVATIONS_PREDICTION: [],
            BELIEF_U_PREDICTION: [],
            Z_MIN_DETECTIONS_PREDICTION: [],
            DETECTIONS_DRIVINGCORRIDOR_PREDICTION: [],
        }

        predictions[BELIEF_O_PREDICTION] = pred_bel_O
        predictions[BELIEF_F_PREDICTION] = pred_bel_F
        predictions[Z_MAX_DETECTIONS_PREDICTION] = pred_z_max_detections
        predictions[Z_MIN_OBSERVATIONS_PREDICTION] = pred_z_min_observations
        predictions[BELIEF_U_PREDICTION] = pred_bel_U
        predictions[Z_MIN_DETECTIONS_PREDICTION] = pred_z_min_detections
        predictions[DETECTIONS_DRIVINGCORRIDOR_PREDICTION] = pred_detections_drivingCorridor

        return predictions
