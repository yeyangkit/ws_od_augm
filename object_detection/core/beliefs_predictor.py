from abc import abstractmethod
import tensorflow as tf

BELIEF_PREDICTION = 'belief_prediction'
# FLOW_PYRAMID = 'flow_pyramid'


class BeliefPredictor(object):
  """BeliefPredictor."""

  def __init__(self, is_training):
    self._is_training = is_training

  @property
  def is_keras_model(self):
    return False

  def predict(self, image_features, scope=None, **params):
    if scope is not None:
      with tf.variable_scope(scope):
        return self._predict(image_features, **params)
    return self._predict(image_features, **params)

  @abstractmethod
  def _predict(self, image_features, **params):
    pass
