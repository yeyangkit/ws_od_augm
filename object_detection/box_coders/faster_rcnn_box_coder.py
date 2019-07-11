# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Faster RCNN box coder.

Faster RCNN box coder follows the coding schema described below:
  ty = (y - ya) / ha
  tx = (x - xa) / wa
  th = log(h / ha)
  tw = log(w / wa)
  where x, y, w, h denote the box's center coordinates, width and height
  respectively. Similarly, xa, ya, wa, ha denote the anchor's center
  coordinates, width and height. tx, ty, tw and th denote the anchor-encoded
  center, width and height respectively.

  See http://arxiv.org/abs/1506.01497 for details.
"""

import tensorflow as tf

from object_detection.core import box_coder
from object_detection.core import box_list

EPSILON = 1e-8


class FasterRcnnBoxCoder(box_coder.BoxCoder):
  """Faster RCNN box coder."""

  def __init__(self, scale_factors=None):
    """Constructor for FasterRcnnBoxCoder.

    Args:
      scale_factors: List of 4 positive scalars to scale ty, tx, th and tw.
        If set to None, does not perform scaling. For Faster RCNN,
        the open-source implementation recommends using [10.0, 10.0, 5.0, 5.0].
    """
    if scale_factors:
      assert len(scale_factors) == 6
      for scalar in scale_factors:
        assert scalar > 0
    self._scale_factors = scale_factors

  @property
  def code_size(self):
    return 4

  @property
  def code_size_3d(self):
    return 6

  def _encode(self, boxes, anchors):
    """Encode a box collection with respect to anchor collection.

    Args:
      boxes: BoxList holding N boxes to be encoded.
      anchors: BoxList of anchors.

    Returns:
      a tensor representing N anchor-encoded boxes of the format
      [ty, tx, th, tw].
    """
    # Convert anchors to the center coordinate representation.
    ycenter_a, xcenter_a, ha, wa = anchors.get_center_coordinates_and_sizes()
    ycenter, xcenter, h, w = boxes.get_center_coordinates_and_sizes()
    # Avoid NaN in division and log below.
    ha += EPSILON
    wa += EPSILON
    h += EPSILON
    w += EPSILON

    tx = (xcenter - xcenter_a) / wa
    ty = (ycenter - ycenter_a) / ha
    tw = tf.log(w / wa)
    th = tf.log(h / ha)
    # Scales location targets as used in paper for joint training.
    if self._scale_factors:
      ty *= self._scale_factors[0]
      tx *= self._scale_factors[1]
      th *= self._scale_factors[2]
      tw *= self._scale_factors[3]
    return tf.transpose(tf.stack([ty, tx, th, tw]))

  def _encode_3d(self, boxes, anchors):
    EPSILON = 1e-8
    ycenter_a, xcenter_a, ha, wa = anchors.get_center_coordinates_and_sizes()
    x_c, y_c, w, h, sin_angle, cos_angle = boxes.get_center_coordinates_and_sizes()
    angle = 0.5 * tf.atan2(sin_angle, cos_angle)

    cond = tf.greater(wa, ha)
    angle_a_1 = tf.multiply(tf.ones(tf.shape(ycenter_a)), 0.0)
    angle_a_2 = tf.multiply(tf.ones(tf.shape(ycenter_a)), 90.0)
    ref_angle = tf.where(cond , angle_a_1, angle_a_2)
    #ref_angle = tf.ones(tf.shape(ycenter_a)) * -90

    w_rotated = tf.where(cond, h, w)
    h_rotated = tf.where(cond, w, h)
    w = w_rotated
    h = h_rotated

    # Avoid NaN in division and log below.
    wa += EPSILON
    w += EPSILON
    ha += EPSILON
    h += EPSILON

    da = tf.sqrt(tf.square(wa) + tf.square(ha))

    tx = (x_c - xcenter_a) / da
    ty = (y_c - ycenter_a) / da
    tw = tf.log(w / wa)
    th = tf.log(h / ha)

    # tangle = (angle - ref_angle) * 3.141 / 180
    # t_sin_angle = tf.sin(2 * tangle)
    # t_cos_angle = tf.cos(2 * tangle)
    ref_angle_rad = ref_angle * 3.141 / 180
    t_sin_angle = sin_angle - tf.sin(2 * ref_angle_rad)
    t_cos_angle = cos_angle - tf.cos(2 * ref_angle_rad)

    tx *= self._scale_factors[0]
    ty *= self._scale_factors[1]
    tw *= self._scale_factors[2]
    th *= self._scale_factors[3]
    t_sin_angle *= self._scale_factors[4]
    t_cos_angle *= self._scale_factors[5]
    return tf.transpose(tf.stack([tx, ty, tw, th, t_sin_angle, t_cos_angle]))

  def _decode(self, rel_codes, anchors):
    """Decode relative codes to boxes.

    Args:
      rel_codes: a tensor representing N anchor-encoded boxes.
      anchors: BoxList of anchors.

    Returns:
      boxes: BoxList holding N bounding boxes.
    """
    ycenter_a, xcenter_a, ha, wa = anchors.get_center_coordinates_and_sizes()

    ty, tx, th, tw = tf.unstack(tf.transpose(rel_codes))
    if self._scale_factors:
      ty /= self._scale_factors[0]
      tx /= self._scale_factors[1]
      th /= self._scale_factors[2]
      tw /= self._scale_factors[3]
    w = tf.exp(tw) * wa
    h = tf.exp(th) * ha
    ycenter = ty * ha + ycenter_a
    xcenter = tx * wa + xcenter_a
    ymin = ycenter - h / 2.
    xmin = xcenter - w / 2.
    ymax = ycenter + h / 2.
    xmax = xcenter + w / 2.
    return box_list.BoxList(tf.transpose(tf.stack([ymin, xmin, ymax, xmax])))

  def _decode_3d(self, rel_codes, anchors):
    ycenter_a, xcenter_a, ha, wa = anchors.get_center_coordinates_and_sizes()
    tx, ty, tw, th, t_sin_angle, t_cos_angle = tf.unstack(tf.transpose(rel_codes))
    tangle = 0.5 * tf.atan2(t_sin_angle, t_cos_angle)

    cond = tf.greater(wa, ha)
    angle_a_1 = tf.multiply(tf.ones(tf.shape(ycenter_a)), 0.0)
    angle_a_2 = tf.multiply(tf.ones(tf.shape(ycenter_a)), 90.0)
    ref_angle = tf.where(cond, angle_a_1, angle_a_2)

    da = tf.sqrt(tf.square(wa) + tf.square(ha))

    tx /= self._scale_factors[0]
    ty /= self._scale_factors[1]
    tw /= self._scale_factors[2]
    th /= self._scale_factors[3]
    t_sin_angle /= self._scale_factors[4]
    t_cos_angle /= self._scale_factors[5]
    w = tf.exp(tw) * wa
    h = tf.exp(th) * ha
    y_c = ty * da + ycenter_a
    x_c = tx * da + xcenter_a

    w_rotated = tf.where(cond, h, w)
    h_rotated = tf.where(cond, w, h)
    w = w_rotated
    h = h_rotated

    # angle = (tangle + ref_angle) * 3.141 / 180
    # sin_angle = tf.sin(2 * angle)
    # cos_angle = tf.cos(2 * angle)
    ref_angle_rad = ref_angle * 3.141 / 180
    sin_angle = t_sin_angle + tf.sin(2 * ref_angle_rad)
    cos_angle = t_cos_angle + tf.cos(2 * ref_angle_rad)
    return box_list.Box3dList(tf.transpose(tf.stack([x_c, y_c, w, h, sin_angle, cos_angle])))
