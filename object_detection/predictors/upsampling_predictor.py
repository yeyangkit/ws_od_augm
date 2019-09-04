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
               filters,
               depth):
    super(UpsamplingPredictor, self).__init__(is_training)
    self._layer_norm = layer_norm
    self._stack_size = stack_size
    self._kernel_size = kernel_size
    self._filters = filters
    self._depth = depth

  def _conv_bn_relu(self, x, filters, ksize, stride):

      x = tf.layers.conv2d(x, filters=filters, kernel_size=ksize, strides=stride, padding='same')
      if self._layer_norm:  # todo fragen where to pass
          x = clayer.layer_norm(x, scale=False)
      x = tf.nn.relu(x)

      return x

  def _unet_block(self, x, filters, training, name):
      with tf.variable_scope(name):
          for i in range(self._stack_size):
              with tf.variable_scope('conv_%i' % i):
                  x = self._conv_bn_relu(x, filters=filters, ksize=self._kernel_size, stride=1)
          return x

  def _create_net(self, x, outputs_channels):
      # tf.summary.image('input--image_features[0]', x)

      x = tf.layers.conv2d_transpose(x, filters=int(self._filters/4), kernel_size=self._kernel_size, strides=2, padding='same')

      x = self._unet_block(x, filters=int(self._filters/4), training=self._is_training, name="dec_1")

      x = tf.layers.conv2d_transpose(x, filters=int(self._filters/16), kernel_size=self._kernel_size, strides=2, padding='same')

      x = self._unet_block(x, filters=int(self._filters/16), training=self._is_training, name="dec_2")

      # End
      with tf.variable_scope("end"):
          x = tf.layers.conv2d(x, filters=outputs_channels, kernel_size=self._kernel_size, strides=1, padding='same',
                               name='conv')
      return x

  def _predict(self, image_features):
    input = image_features[0]
    output = self._create_net(input, outputs_channels=2)

    pred_bel_F = tf.slice(output, begin=[0, 0, 0, 0], size=[-1, -1, -1, 1])
    pred_bel_O = tf.slice(output, begin=[0, 0, 0, 1], size=[-1, -1, -1, 1])

    tf.summary.image('predicted_bel_O_before_clamping', pred_bel_O)
    tf.summary.image('predicted_bel_F_before_clamping', pred_bel_F)

    with tf.name_scope("clamping"):
        pred_bel_F_clamped = tf.maximum(tf.minimum(pred_bel_F, 1), 0)
        pred_bel_O_clamped = tf.maximum(tf.minimum(pred_bel_O, 1), 0)
        # pred_bel_F_clamped = tf.clamp
        # pred_bel_F, 1), 0)
        # pred_bel_O_clamped = tf.maximum(tf.minimum(pred_bel_O, 1), 0)

    predictions = {
        BELIEF_O_PREDICTION: [],
        BELIEF_F_PREDICTION: []
    }
    predictions[BELIEF_O_PREDICTION] = pred_bel_O_clamped
    predictions[BELIEF_F_PREDICTION] = pred_bel_F_clamped
    # predictions[Z_XX] = # todo

    return predictions