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
        # x = tf.Print(x, [x], 'img_features[0]:', summarize=15)

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

    # # Custom loss layer
    # class CustomMultiLossLayer(Layer):
    #     def __init__(self, nb_outputs=2, **kwargs):
    #         self.nb_outputs = nb_outputs
    #         self.is_placeholder = True
    #         super(CustomMultiLossLayer, self).__init__(**kwargs)
    #
    #     def build(self, input_shape=None):
    #         # initialise log_vars
    #         self.log_vars = []
    #         for i in range(self.nb_outputs):
    #             self.log_vars += [self.add_weight(name='log_var' + str(i), shape=(1,),
    #                                               initializer=Constant(0.), trainable=True)]
    #         super(CustomMultiLossLayer, self).build(input_shape)
    #
    #     def multi_loss(self, ys_true, ys_pred):
    #         assert len(ys_true) == self.nb_outputs and len(ys_pred) == self.nb_outputs
    #         loss = 0
    #         for y_true, y_pred, log_var in zip(ys_true, ys_pred, self.log_vars):
    #             precision = K.exp(-log_var[0])
    #             loss += K.sum(precision * (y_true - y_pred) ** 2. + log_var[0], -1)
    #         return K.mean(loss)
    #
    #     def call(self, inputs):
    #         ys_true = inputs[:self.nb_outputs]
    #         ys_pred = inputs[self.nb_outputs:]
    #         loss = self.multi_loss(ys_true, ys_pred)
    #         self.add_loss(loss, inputs=inputs)
    #         # We won't actually use the output.
    #         return K.concatenate(inputs, -1)
    #
    # def get_prediction_model():
    #     inp = Input(shape=(Q,), name='inp')
    #     x = Dense(nb_features, activation='relu')(inp)
    #     y1_pred = Dense(D1)(x)
    #     y2_pred = Dense(D2)(x)
    #     return Model(inp, [y1_pred, y2_pred])
    #
    # def get_trainable_model(prediction_model):
    #     inp = Input(shape=(Q,), name='inp')
    #     y1_pred, y2_pred = prediction_model(inp)
    #     y1_true = Input(shape=(D1,), name='y1_true')
    #     y2_true = Input(shape=(D2,), name='y2_true')
    #     out = CustomMultiLossLayer(nb_outputs=2)([y1_true, y2_true, y1_pred, y2_pred])
    #     return Model([inp, y1_true, y2_true], out)
    #
    # prediction_model = get_prediction_model()
    # trainable_model = get_trainable_model(prediction_model)
    # trainable_model.compile(optimizer='adam', loss=None)
    # assert len(trainable_model.layers[-1].trainable_weights) == 2  # two log_vars, one for each output
    # assert len(trainable_model.losses) == 1


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
