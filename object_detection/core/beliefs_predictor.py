from abc import abstractmethod
import tensorflow as tf

BELIEF_O_PREDICTION = 'belief_O_prediction'
BELIEF_F_PREDICTION = 'belief_F_prediction'
Z_MAX_DETECTIONS_PREDICTION = 'z_max_detections_prediction'
Z_MIN_OBSERVATIONS_PREDICTION = 'z_min_observations_prediction'


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
