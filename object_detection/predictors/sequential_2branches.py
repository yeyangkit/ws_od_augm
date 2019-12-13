
"""Beliefs Estimation using U-Net."""

import tensorflow as tf
import numpy as np
import tensorflow.contrib.layers as clayer
import matplotlib as plt

from object_detection.core import beliefs_predictor

BELIEF_O_PREDICTION = beliefs_predictor.BELIEF_O_PREDICTION
BELIEF_F_PREDICTION = beliefs_predictor.BELIEF_F_PREDICTION
Z_MAX_DETECTIONS_PREDICTION = beliefs_predictor.Z_MAX_DETECTIONS_PREDICTION
Z_MIN_OBSERVATIONS_PREDICTION = beliefs_predictor.Z_MIN_OBSERVATIONS_PREDICTION
BELIEF_U_PREDICTION = beliefs_predictor.BELIEF_U_PREDICTION
Z_MIN_DETECTIONS_PREDICTION = beliefs_predictor.Z_MIN_DETECTIONS_PREDICTION
DETECTIONS_DRIVINGCORRIDOR_PREDICTION = beliefs_predictor.DETECTIONS_DRIVINGCORRIDOR_PREDICTION
INTENSITY_PREDICTION = beliefs_predictor.INTENSITY_PREDICTION


class Sequential2branchesPredictor(beliefs_predictor.BeliefPredictor):
    """U Net Predictor with weight sharing."""

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
            x1 = tf.layers.conv2d(x, depth, [3, 3], name="multiRes_shortcut_{}_stack{}_conv3".format(name, i), padding='same')
            x1 = tf.nn.relu(x1, name="multiresPath_{}_3_relu".format(name))
            x2 = tf.layers.conv2d(x, depth, [1, 1], name="multiRes_shortcut_{}_stack{}_conv1".format(name, i), padding='same')
            x2 = tf.nn.relu(x2, name="multiresPath_{}_1_relu".format(name))
            x = x1 + x2
        return x

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


        preprocessed_input = x

        use_attention_map = True

        if use_attention_map is True:
            skips = []
            for i in range(depth):
                x = self._multiResUnet_block(x, depth=f * (2 ** i), name="augment_enc_%i" % i)
                skips.append(x)
                x = tf.layers.max_pooling2d(x, pool_size=2, strides=2, name='augment_pool_%i' % (i + 1))

            x = self._multiResUnet_block(x, depth=f * (2 ** (depth - 1)), name="augment_deep")

            for i in reversed(range(depth)):
                if i != 0:

                    with tf.variable_scope('augment_up_conv_%i' % (i + 1)):
                        x = tf.layers.conv2d_transpose(x, filters=f * (2 ** i), kernel_size=kernel_size, strides=2,
                                                       padding='same', name="augment_deconv_%i" % i)

                    x = tf.concat([skips.pop(), x], axis=3)

                    x = self._multiResUnet_block(x, depth=f * (2 ** i), name="augment_decoder_%i" % i)

                else:

                    short_cut = skips.pop()

                    use_separable_conv = True


                    attention_map = tf.concat((tf.layers.conv2d(short_cut, filters=f, kernel_size=4, strides=[2,2], name='attetion_skip_conv') \
                                        , tf.layers.conv2d(x, filters=f, kernel_size=1,name='attention_coarse_conv')), axis=3)
                    attention_map = tf.nn.relu(attention_map, name='attention_relu')
                    attention_map = tf.layers.conv2d(attention_map, filters=f, kernel_size=1, name='attention_psai_conv', padding='same')

                    attention_map = tf.nn.sigmoid(attention_map, name='attention_sigmoid')

                    # attention_map = tf.layers.conv2d_transpose(attention_map, filters=f, kernel_size=4, strides=[2,2], name='attention_map_interpolation', padding='same')
                    attention_map = tf.image.resize_images(attention_map, [short_cut.shape[1], short_cut.shape[2]], align_corners=True)

                    # visulize the attention maps
                    attention = tf.expand_dims(
                                tf.concat((preprocessed_input[0, :, :, 0], preprocessed_input[0, :, :, 2], preprocessed_input[0,:,:,5], tf.reduce_mean(255*attention_map[0,:,:,:], axis=2)), axis=1), 0)

                    for j in range(int(f/4)):
                        attention_row = tf.expand_dims(
                            tf.concat((255*attention_map[0, :, :, 4*j], 255*attention_map[0, :, :, 4*j+1], 255*attention_map[0, :, :, 4*j+2], 255*attention_map[0, :, :, 4*j+3]), axis=1), 0)
                        attention = tf.concat((attention, attention_row), axis=1)
                    attention = tf.expand_dims(attention, axis=-1)
                    tf.summary.image('attention_mask', attention, family='sequential_model_watcher')


                    short_cut_with_attention = attention_map * short_cut
                    attention = tf.expand_dims(
                                tf.concat((preprocessed_input[0, :, :, 0], preprocessed_input[0, :, :, 2], preprocessed_input[0,:,:,5], tf.reduce_mean(short_cut_with_attention[0,:,:,:], axis=2)), axis=1), 0)
                    for j in range(int(f/4)):

                        attention_row = tf.expand_dims(
                            tf.concat((short_cut_with_attention[0, :, :, 4*j], short_cut_with_attention[0, :, :, 4*j+1], short_cut_with_attention[0, :, :, 4*j+2], short_cut_with_attention[0, :, :, 4*j+3]), axis=1), 0)
                        attention = tf.concat((attention, attention_row), axis=1)
                    attention = tf.expand_dims(attention, axis=-1)
                    tf.summary.image('multiplied with_attention', attention, family='sequential_model_watcher')

                    with tf.variable_scope('augment_up_conv_%i' % (i + 1)):
                        x = tf.layers.conv2d_transpose(x, filters=f * (2 ** i), kernel_size=kernel_size, strides=2,
                                                       padding='same')


                    x = tf.concat((short_cut_with_attention, x), axis=3)

                    x = self._multiResUnet_block(x, depth=f * (2 ** i), name="augment_dec_%i" % i)
        else:
            skips = []

            for i in range(depth):
                x = self._multiResUnet_block(x, depth=f * (2 ** i), name="augment_enc_%i" % i)
                skips.append(x)
                x = tf.layers.max_pooling2d(x, pool_size=2, strides=2, name='augment_pool_%i' % (i + 1))

            x = self._multiResUnet_block(x, depth=f * (2 ** (depth - 1)), name="augment_deep")

            for i in reversed(range(depth)):
                with tf.variable_scope('augment_up_conv_%i' % (i + 1)):
                    x = tf.layers.conv2d_transpose(x, filters=f * (2 ** i), kernel_size=kernel_size, strides=2,
                                                   padding='same')

                x = tf.concat([skips.pop(), x], axis=3)

                x = self._multiResUnet_block(x, depth=f * (2 ** i), name="augment_dec_%i" % i)


        with tf.variable_scope("augm_BELS_end"):
            x1 = tf.layers.conv2d(x, filters=bels_outputs_channels, kernel_size=self._final_kernel_size, strides=1,
                                  padding='same', name='augm_BELS_conv_end_outputs')
            bels = tf.nn.softmax(x1, name='augm_BELS_softmax')



        with tf.variable_scope("augm_MAPS_end"):
            x2 = tf.layers.conv2d(x, filters=maps_outputs_channels, kernel_size=self._final_kernel_size, strides=1,
                                  padding='same',
                                  name='augm_MAPS_conv_end_outputs')
            maps = tf.nn.relu(x2, name='augm_MAPS_relu')

        return bels, maps

    def _predict(self, image_features, preprocessed_input, scope=None):

        # Create Unet
        pred_bels, pred_maps = self._create_unet(preprocessed_input, training=True, depth=4, f=self._min_filter, kernel_size=3,
                                                 stack_size=self._stack_size, maps_outputs_channels=5,
                                                 bels_outputs_channels=3)

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
                                                 stack_size=self._stack_size, maps_outputs_channels=5,
                                                 bels_outputs_channels=3)

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













