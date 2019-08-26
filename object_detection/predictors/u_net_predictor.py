"""Beliefs Estimation using U-Net."""

import tensorflow as tf

import tensorflow.contrib.layers as clayer


from object_detection.core import augmentation_predictor
# from object_detection.core import flow_ops

slim = tf.contrib.slim

# FLOW_PREDICTION = flow_predictor.FLOW_PREDICTION
# FLOW_PYRAMID = flow_predictor.FLOW_PYRAMID

class UNetPredictor(flow_predictor.FlowPredictor):
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

  def conv_bn_relu(x, filters, ksize, stride, training):

      x = tf.layers.conv2d(upfeat, filters=filters, kernel_size=ksize, strides=stride, padding='same')
      if FLAGS.layer_norm:
          x = clayer.layer_norm(x, scale=False)
      x = tf.nn.relu(x)

      return x

  def unet_block(x, filters, training, name):
      with tf.variable_scope(name):
          for i in range(FLAGS.stack_size):
              with tf.variable_scope('conv_%i' % i):
                  x = conv_bn_relu(x, filters=filters, ksize=FLAGS.kernel_size, stride=1, training=training)
          return x

  def create_unet(x, outputs_channels, training):

      f = FLAGS.filters

      skips = []

      for i in range(FLAGS.depth):
          x = unet_block(x, filters=f * (2 ** i), training=training, name="enc_%i" % i)

          skips.append(x)
          x = tf.layers.max_pooling2d(x, pool_size=2, strides=2, name='pool_%i' % (i + 1))

      x = unet_block(x, filters=f * (2 ** (FLAGS.depth - 1)), training=training, name="deep")

      for i in reversed(range(FLAGS.depth)):

          with tf.variable_scope('up_conv_%i' % (i + 1)):
              x = tf.layers.conv2d_transpose(x, filters=f * (2 ** i), kernel_size=FLAGS.kernel_size, strides=2,
                                             padding='same')
              if FLAGS.layer_norm:
                  x = clayer.layer_norm(x, scale=False)
              x = tf.nn.relu(x)
          x = tf.concat([skips.pop(), x], axis=3)

          x = unet_block(x, filters=f * (2 ** i), training=training, name="dec_%i" % i)

      # End
      with tf.variable_scope("end"):
          x = tf.layers.conv2d(x, filters=outputs_channels, kernel_size=FLAGS.kernel_size, strides=1, padding='same',
                               name='conv')
      return x

  # def _refine_flow(self, upfeat, flow, idx):
  #   init = tf.keras.initializers.he_normal()
  #   with tf.variable_scope('flow_refinement{}'.format(idx)):
  #       x = tf.layers.conv2d(upfeat, 128, 3, 1, 'same', dilation_rate=1, kernel_initializer=init, name='conv1')
  #       x = tf.nn.leaky_relu(x, alpha=0.1)
  #       x = tf.layers.conv2d(x, 128, 3, 1, 'same', dilation_rate=2, kernel_initializer=init, name='conv2')
  #       x = tf.nn.leaky_relu(x, alpha=0.1)
  #       x = tf.layers.conv2d(x, 128, 3, 1, 'same', dilation_rate=4, kernel_initializer=init, name='conv3')
  #       x = tf.nn.leaky_relu(x, alpha=0.1)
  #       x = tf.layers.conv2d(x, 96, 3, 1, 'same', dilation_rate=8, kernel_initializer=init, name='conv4')
  #       x = tf.nn.leaky_relu(x, alpha=0.1)
  #       x = tf.layers.conv2d(x, 64, 3, 1, 'same', dilation_rate=16, kernel_initializer=init, name='conv5')
  #       x = tf.nn.leaky_relu(x, alpha=0.1)
  #       x = tf.layers.conv2d(x, 32, 3, 1, 'same', dilation_rate=1, kernel_initializer=init, name='conv6')
  #       x = tf.nn.leaky_relu(x, alpha=0.1)
  #       x = tf.layers.conv2d(x, 2, 3, 1, 'same', dilation_rate=1, kernel_initializer=init, name='conv7')
  #
  #       return tf.add(flow, x, name='refined_flow')

  # def _predict_flow(self, cost_volume, image_feature, up_flow, up_flow_feat, idx):
  #   init = tf.keras.initializers.he_normal()
  #   with tf.variable_scope('flow_prediction{}'.format(idx)):
  #       if image_feature is None and up_flow is None and up_flow_feat is None:
  #           x = cost_volume
  #       else:
  #           x = tf.concat([cost_volume, image_feature, up_flow, up_flow_feat], axis=3)
  #
  #       conv = tf.layers.conv2d(x, 128, 3, 1, 'same', kernel_initializer=init, name='conv1')
  #       act = tf.nn.leaky_relu(conv, alpha=0.1)
  #       x = tf.concat([act, x], axis=3) if self._use_dense else act
  #       conv = tf.layers.conv2d(x, 128, 3, 1, 'same', kernel_initializer=init, name='conv2')
  #       act = tf.nn.leaky_relu(conv, alpha=0.1)
  #       x = tf.concat([act, x], axis=3) if self._use_dense else act
  #       conv = tf.layers.conv2d(x, 96, 3, 1, 'same', kernel_initializer=init, name='conv3')
  #       act = tf.nn.leaky_relu(conv, alpha=0.1)
  #       x = tf.concat([act, x], axis=3) if self._use_dense else act
  #       conv = tf.layers.conv2d(x, 64, 3, 1, 'same', kernel_initializer=init, name='conv4')
  #       act = tf.nn.leaky_relu(conv, alpha=0.1)
  #       x = tf.concat([act, x], axis=3) if self._use_dense else act
  #       conv = tf.layers.conv2d(x, 32, 3, 1, 'same', kernel_initializer=init, name='conv5')
  #       act = tf.nn.leaky_relu(conv, alpha=0.1)
  #       up_feat = tf.concat([act, x], axis=3) if self._use_dense else act
  #
  #       flow = tf.layers.conv2d(up_feat, 2, 3, 1, 'same', name='flow')
  #
  #       return up_feat, flow

  def _predict(self, image_features):
    predictions = {
        BELIEF_O: [],
        BELIEF_F: []
    }

    # flow_feat, flow = self._predict_flow(cost_volume, None, None, None, feat_idx)

    # flow_pyr = []
    # for feat_idx in range(self._fpn_config.max_level, self._fpn_config.min_level-1,  -1):
    #     with tf.variable_scope('PWCNetPredictor', reuse=reuse):
    #         if feat_idx == self._fpn_config.max_level:
    #             cost_volume = flow_ops.cost_volume(image_features[feat_idx - self._fpn_config.min_level],
    #                                                image_prev_features[feat_idx - self._fpn_config.min_level],
    #                                                self._search_range,
    #                                                'correlation{}'.format(feat_idx))
    #             flow_feat, flow = self._predict_flow(cost_volume, None, None, None, feat_idx)
    #         else:
    #             scale = 20. / 2 ** feat_idx
    #             image_prev_feature_warped = tf.contrib.image.dense_image_warp(
    #                 image_prev_features[feat_idx - self._fpn_config.min_level],
    #                 up_flow * scale, 'warp{}'.format(feat_idx))
    #             cost_volume = flow_ops.cost_volume(image_features[feat_idx - self._fpn_config.min_level],
    #                                                image_prev_feature_warped,
    #                                                self._search_range,
    #                                                'correlation{}'.format(feat_idx))
    #             flow_feat, flow = self._predict_flow(cost_volume, image_features[feat_idx - self._fpn_config.min_level],
    #                                                  up_flow, up_flow_feat, feat_idx)
    #             _, feat_idx_height, feat_idx_width, _ = tf.unstack(tf.shape(
    #                 image_features[feat_idx - self._fpn_config.min_level]))
    #         if feat_idx != self._fpn_config.min_level:
    #             if self._use_res:
    #                 flow = self._refine_flow(flow_feat, flow, feat_idx)
    #             flow_pyr.insert(0, flow)
    #             up_flow = tf.layers.conv2d_transpose(flow, 2, 4, 2, 'same',
    #                                                  name='upsampled_flow{}'.format(feat_idx))
    #             up_flow_feat = tf.layers.conv2d_transpose(flow_feat, 2, 4, 2, 'same',
    #                                                       name='upsampled_flow_feature{}'.format(feat_idx))
    #         else:
    #             flow = self._refine_flow(flow_feat, flow, feat_idx)
    #             flow_pyr.insert(0, flow)
    #             scale = 2 ** self._fpn_config.min_level
    #             size = (feat_idx_height * scale, feat_idx_width * scale)
    #             #flow_pred = tf.image.resize_bilinear(flow, size, name="flow_pred") * scale
    #             flow_pred = tf.image.resize_bilinear(flow, size, name="flow_pred") * 20.0

    # Create Unet
    output = create_unet(input, outputs_channels=2, training=training)

    pred_bel_F = tf.slice(output, begin=[0, 0, 0, 0], size=[-1, -1, -1, 1])
    pred_bel_O = tf.slice(output, begin=[0, 0, 0, 1], size=[-1, -1, -1, 1])

    with tf.name_scope("clamping"):
        pred_bel_F_clamped = tf.maximum(tf.minimum(pred_bel_F, 1), 0)
        pred_bel_O_clamped = tf.maximum(tf.minimum(pred_bel_O, 1), 0)

    # predictions[Z_XX] = flow_pyr
    predictions[BELIEF_O] = bel_O_pred
    predictions[BELIEF_F] = bel_F_pred

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