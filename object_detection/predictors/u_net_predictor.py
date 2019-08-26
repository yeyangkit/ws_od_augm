"""Beliefs Estimation using U-Net."""

import tensorflow as tf

import tensorflow.contrib.layers as clayer


from object_detection.core import beliefs_predictor

slim = tf.contrib.slim

BELIEF_O_PREDICTION  = beliefs_predictor.BELIEF_O_PREDICTION
BELIEF_F_PREDICTION  = belies_predictor.BELIEF_F_PREDICTION


class UNetPredictor(flow_predictor.FlowPredictor):  ## todo fragen relationship between od and augm where to see
  """U Net Predictor with weight sharing."""

  def __init__(self,
               is_training,
               fpn_config,
               search_range,
               use_dense,
               use_res):
    super(PWCNetPredictor, self).__init__(is_training)
    self._fpn_config = fpn_config
    self._search_range = search_range
    self._use_dense = use_dense
    self._use_res = use_res

  def _conv_bn_relu(self, x, filters, ksize, stride, training):

      x = tf.layers.conv2d(upfeat, filters=filters, kernel_size=ksize, strides=stride, padding='same')
      if FLAGS.layer_norm:  # todo fragen where to pass
          x = clayer.layer_norm(x, scale=False)
      x = tf.nn.relu(x)

      return x

  def _unet_block(self, x, filters, training, name):
      with tf.variable_scope(name):
          for i in range(FLAGS.stack_size):
              with tf.variable_scope('conv_%i' % i):
                  x = self._conv_bn_relu(x, filters=filters, ksize=FLAGS.kernel_size, stride=1, training=training)
          return x

  def _create_unet(self, x, outputs_channels, training):

      f = FLAGS.filters # todo fragen where to pass

      skips = []

      for i in range(FLAGS.depth):
          x = self._unet_block(x, filters=f * (2 ** i), training=training, name="enc_%i" % i)

          skips.append(x)
          x = tf.layers.max_pooling2d(x, pool_size=2, strides=2, name='pool_%i' % (i + 1))

      x = self._unet_block(x, filters=f * (2 ** (FLAGS.depth - 1)), training=training, name="deep")

      for i in reversed(range(FLAGS.depth+1)):

          with tf.variable_scope('up_conv_%i' % (i + 1)):
              x = tf.layers.conv2d_transpose(x, filters=f * (2 ** i), kernel_size=FLAGS.kernel_size, strides=2,
                                             padding='same')
              if FLAGS.layer_norm:
                  x = clayer.layer_norm(x, scale=False)
              x = tf.nn.relu(x)
          x = tf.concat([skips.pop(), x], axis=3)

          x = self._unet_block(x, filters=f * (2 ** i), training=training, name="dec_%i" % i)

      # End
      with tf.variable_scope("end"):
          x = tf.layers.conv2d(x, filters=outputs_channels, kernel_size=FLAGS.kernel_size, strides=1, padding='same',
                               name='conv')
      return x

  def _predict(self, image_features):
    predictions = {
        BELIEF_O_PREDICTION: [],
        BELIEF_F_PREDICTION: []
    }

    #TODO fragen
    input = image_features[self._fpn_config.max_level]

    # Create Unet
    output = self._create_unet(input, outputs_channels=2, training=training)

    pred_bel_F = tf.slice(output, begin=[0, 0, 0, 0], size=[-1, -1, -1, 1])
    pred_bel_O = tf.slice(output, begin=[0, 0, 0, 1], size=[-1, -1, -1, 1])

    with tf.name_scope("clamping"):
        pred_bel_F_clamped = tf.maximum(tf.minimum(pred_bel_F, 1), 0)
        pred_bel_O_clamped = tf.maximum(tf.minimum(pred_bel_O, 1), 0)


    predictions[BELIEF_O_PREDICTION] = pred_bel_O_clamped
    predictions[BELIEF_F_PREDICTION] = pred_bel_F_clamped
    # predictions[Z_XX] = # todo

    return predictions


def model_fn(features, labels, mode):

    training = (mode == tf.estimator.ModeKeys.TRAIN)

    with tf.variable_scope("unet_pool"):
        # Network input

        # feat_cnt = features['img_file_cnt_high'] + features['img_file_cnt_low']
        # feat_hits = features['img_file_hits_high'] + features['img_file_hits_low']
        # feat_int_tmp = features['img_file_int_high']*features['img_file_hits_high'] + features['img_file_int_low']*features['img_file_hits_low']
        # feat_int = feat_int_tmp / (tf.cast(tf.greater(0.5, feat_hits), dtype=tf.float32) + feat_hits)
        #
        # input = tf.concat([feat_cnt,
        #                    feat_hits*10,
        #                    feat_int], axis=3)

        input = tf.concat([features['img_file_cnt_high'],
                           features['img_file_cnt_low'],
                           features['img_file_hits_high']*10,
                           features['img_file_hits_low']*10,
                           features['img_file_int_high'],
                           features['img_file_int_low']], axis=3)
        # Create Unet
        output = create_unet(input, outputs_channels=2, training=training)

        pred_bel_F = tf.slice(output, begin=[0, 0, 0, 0], size=[-1, -1, -1, 1])
        pred_bel_O = tf.slice(output, begin=[0, 0, 0, 1], size=[-1, -1, -1, 1])

        with tf.name_scope("clamping"):
            pred_bel_F_clamped = tf.maximum(tf.minimum(pred_bel_F, 1), 0)
            pred_bel_O_clamped = tf.maximum(tf.minimum(pred_bel_O, 1), 0)

    # RETURN PREDICT #
    predictions = {
        'in_cnt_high': features['img_file_cnt_high'],
        'in_cnt_low': features['img_file_cnt_low'],
        'in_hits_high': features['img_file_hits_high'],
        'in_hits_low': features['img_file_hits_low'],
        'in_int_high': features['img_file_int_high'],
        'in_int_low': features['img_file_int_low'],
        'in_path': features['img_path_cnt_high'],
        'bel_F': pred_bel_F_clamped,
        'bel_O': pred_bel_O_clamped
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    label_bel_F = labels['img_file_bel_F']
    label_bel_O = labels['img_file_bel_O']

    # LOSSES #
    with tf.name_scope("losses_and_weights"):
        wLC10 = my_weights_label_cert(labels, 10.)

        L1 = my_loss_L1(pred_bel_F, label_bel_F, xBiggerY=10., weights=wLC10) + my_loss_L1(pred_bel_O, label_bel_O, xBiggerY=10., weights=wLC10)
        L1x2 = my_loss_L1(pred_bel_F, label_bel_F, xBiggerY=2.) + my_loss_L1(pred_bel_O, label_bel_O, xBiggerY=2.)
        L2 = my_loss_L2(pred_bel_F, label_bel_F) + my_loss_L2(pred_bel_O, label_bel_O)

        train_loss = L1

    # SUMMARIES #
    with tf.name_scope("summaries"):

        ema = tf.train.ExponentialMovingAverage(decay=0.9)
        ema_ops = ema.apply([train_loss])

        tf.summary.scalar("loss", ema.average(train_loss))

        my_image_sum(features, predictions, labels)

    # RETURN EVAL #
    if mode == tf.estimator.ModeKeys.EVAL:
        with tf.name_scope("metrics"):
            metric_ops = get_my_metric_dict(predictions, labels, train_loss)
        return tf.estimator.EstimatorSpec(mode=mode, eval_metric_ops=metric_ops, loss=train_loss)

    # RETURN TRAIN #
    if training:

        with tf.name_scope("unet_pool_train"):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                tvars = [var for var in tf.trainable_variables() if var.name.startswith("unet_pool")]
                optim = tf.train.AdamOptimizer(FLAGS.lr)
                grads_and_vars = optim.compute_gradients(train_loss, var_list=tvars)
                train = optim.apply_gradients(grads_and_vars)

        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=train_loss,
                                          train_op=tf.group(ema_ops, get_global_step_incr_op(), train))

    raise ValueError("Unkown Estimator Mode Key!")