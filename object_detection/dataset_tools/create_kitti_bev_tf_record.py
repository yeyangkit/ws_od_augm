import io
import os
import linecache
import math
import yaml
import hashlib
import base64

import numpy as np
import PIL.Image as pil
import tensorflow as tf

from absl import app
from absl import flags
from absl import logging
from numpy.linalg import inv
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

FLAGS = flags.FLAGS
flags.DEFINE_string('data', None, 'Directory to grid maps.')
flags.DEFINE_string('data_mask', None, 'Directory to occupancy mask.')
flags.DEFINE_string('param', None, 'Directory to grid map parameter file.')
flags.DEFINE_string('labels', '/mrtstorage/datasets/kitti/object_detection/training/label_2/', 'Directory to kitti labels.')
flags.DEFINE_string('calib', '/mrtstorage/datasets/kitti/object_detection/training/calib/', 'Directory to kitti calibrations.')
flags.DEFINE_string('output', '/tmp/', 'Path to directory to output TFRecords.')
flags.DEFINE_string('label_map', '/mrtstorage/datasets/kitti/object_detection/kitti_object_label_map.pbtxt', 'Path to label map proto')
flags.DEFINE_string('train_set', '/mrtstorage/datasets/kitti/object_detection/split_at_5150/train_set_split_at_5150.txt', 'Path to training set file')
flags.DEFINE_string('val_set', '/mrtstorage/datasets/kitti/object_detection/split_at_5150/val_set_split_at_5150.txt', 'Path to validation set file')
flags.DEFINE_integer('subset', 10, 'Only process a subset of examples')


class Label:

  def __init__(self,
               type=None,
               truncation=None,
               occlusion=None,
               alpha=None,
               x1=None,
               y1=None,
               x2=None,
               y2=None,
               h=None,
               w=None,
               l=None,
               tx=None,
               ty=None,
               tz=None,
               ry=None):
    self.type = type
    self.truncation = truncation
    self.occlusion = occlusion
    self.alpha = alpha
    self.x1 = x1
    self.y1 = y1
    self.x2 = x2
    self.y2 = y2
    self.h = h
    self.w = w
    self.l = l
    self.tx = tx
    self.ty = ty
    self.tz = tz
    self.ry = ry

def read_params(param_dir):
    with open(param_dir, 'r') as stream:
        params = yaml.load(stream)
        return params

def _read_calib(line):
    str_tf = linecache.getline(line, 6).split(' ')[1:]
    p = [float(i) for i in str_tf]
    tf = np.array([[p[0], p[1], p[2], p[3]],
                   [p[4], p[5], p[6], p[7]],
                   [p[8], p[9], p[10], p[11]],
                   [0, 0, 0, 1]])
    return tf

def read_label_file(path):
    label_file = open(path)
    labels = []
    for idx, line in enumerate(label_file):
        str_l = line.split(' ')
        labels.append(Label(type=str_l[0],
                            truncation=float(str_l[1]),
                            occlusion=float(str_l[2]),
                            alpha=float(str_l[3]),
                            x1=float(str_l[4]),
                            y1=float(str_l[5]),
                            x2=float(str_l[6]),
                            y2=float(str_l[7]),
                            h=float(str_l[8]),
                            w=float(str_l[9]),
                            l=float(str_l[10]),
                            tx=float(str_l[11]),
                            ty=float(str_l[12]),
                            tz=float(str_l[13]),
                            ry=float(str_l[14])
                            ))
    return labels

def _readImage(path, prefix, name):
    fn = os.path.join(path, prefix + '_' + name + '.png')
    logging.debug('Opening: ' + fn)

    with tf.gfile.GFile(fn, 'rb') as fid:
        encoded_png = fid.read()
    encoded_png_io = io.BytesIO(encoded_png)
    image = pil.open(encoded_png_io)
    image = np.asarray(image)
    key = hashlib.sha256(encoded_png).hexdigest()
    return encoded_png

def _flipAngle(angle_rad):
  if angle_rad < -1.57:
      angle_rad += 3.14
  elif angle_rad > 1.57:
      angle_rad -= 3.14
  return angle_rad

def dict_to_tf_example(labels_image,
                       labels_t,
                       label_data,
                       params,
                       label_map_dict,
                       image_dir,
                       mask_dir,
                       image_prefix):

  width = int(60 / 0.15)
  height = int(60 / 0.15)
  xmin = []
  ymin = []
  xmax = []
  ymax = []
  x_c = []
  y_c = []
  w = []
  h = []
  angle = []
  sin_angle = []
  cos_angle = []
  classes = []
  classes_text = []

  for idx, label_img in enumerate(labels_image):
    xmin.append(min(label_img[0]) / width)
    ymin.append(min(label_img[1]) / height)
    xmax.append(max(label_img[0]) / width)
    ymax.append(max(label_img[1]) / height)
    x_min = min(label_img[0]) / width
    y_min = min(label_img[1]) / height
    x_max = max(label_img[0]) / width
    y_max = max(label_img[1]) / height
    if (x_min >=1) or (y_min >=1) or (x_max >=1) or (y_max >=1):
        print('Higher:', x_min, y_min, x_max, y_max)
    if (x_min <= 0) or (y_min <= 0) or (x_max <= 0) or (y_max <= 0):
        print('Lower:', x_min, y_min, x_max, y_max)
    #x_c.append((min(label_img[0]) + max(label_img[0])) / (2 * width))
    #y_c.append((min(label_img[1]) + max(label_img[1])) / (2 * height))
    x_c.append(labels_t[idx][0])
    y_c.append(labels_t[idx][1])
    angle_rad = _flipAngle(label_data[idx].ry)
    angle.append(angle_rad)
    sin_angle.append(math.sin(2 * angle_rad))
    cos_angle.append(math.cos(2 * angle_rad))
    vec_s_x = math.cos(angle_rad)
    vec_s_y = math.sin(angle_rad)

    w_p = label_data[idx].w / 0.15
    w_p_s = w_p * math.sqrt(vec_s_x * vec_s_x / (height * height) + vec_s_y * vec_s_y / (width * width))
    w.append(w_p_s)

    l_p = label_data[idx].l / 0.15
    l_p_s = l_p * math.sqrt(vec_s_x * vec_s_x / (width * width) + vec_s_y * vec_s_y / (height * height))
    h.append(l_p_s)

    class_name = label_data[idx].type
    classes_text.append(class_name.encode('utf8'))
    classes.append(label_map_dict[class_name])

  return tf.train.Example(features=tf.train.Features(feature={
    'id': dataset_util.bytes_feature(image_prefix.encode('utf8')),
    'image/format': dataset_util.bytes_feature('png'.encode('utf8')),

    'layers/height': dataset_util.int64_feature(height),
    'layers/width': dataset_util.int64_feature(width),

    'layers/detections/encoded': dataset_util.bytes_feature(_readImage(image_dir, image_prefix, 'detections_cartesian')),
    'layers/observations/encoded': dataset_util.bytes_feature(_readImage(image_dir, image_prefix, 'observations_cartesian')),
    'layers/decay_rate/encoded': dataset_util.bytes_feature(_readImage(image_dir, image_prefix, 'decay_rate_cartesian')),
    'layers/intensity/encoded': dataset_util.bytes_feature(_readImage(image_dir, image_prefix, 'intensity_cartesian')),
    'layers/zmin/encoded': dataset_util.bytes_feature(_readImage(image_dir, image_prefix, 'z_min_detections_cartesian')),
    'layers/zmax/encoded': dataset_util.bytes_feature(_readImage(image_dir, image_prefix, 'z_max_detections_cartesian')),
    'layers/occlusions/encoded': dataset_util.bytes_feature(_readImage(image_dir, image_prefix, 'z_max_occlusions_cartesian')),
    'layers/rgb/encoded': dataset_util.bytes_feature(_readImage(image_dir, image_prefix, 'rgb_cartesian')),

    'masks/occupancy_mask': dataset_util.bytes_feature(_readImage(mask_dir, image_prefix, 'occupancy_mask')),

    'boxes/aligned/x_min': dataset_util.float_list_feature(xmin),
    'boxes/aligned/x_max': dataset_util.float_list_feature(xmax),
    'boxes/aligned/y_min': dataset_util.float_list_feature(ymin),
    'boxes/aligned/y_max': dataset_util.float_list_feature(ymax),

    'boxes/inclined/x_c': dataset_util.float_list_feature(x_c),
    'boxes/inclined/y_c': dataset_util.float_list_feature(y_c),
    'boxes/inclined/w': dataset_util.float_list_feature(w),
    'boxes/inclined/h': dataset_util.float_list_feature(h),
    'boxes/inclined/angle': dataset_util.float_list_feature(angle),
    'boxes/inclined/sin_angle': dataset_util.float_list_feature(sin_angle),
    'boxes/inclined/cos_angle': dataset_util.float_list_feature(cos_angle),

    'boxes/class/text': dataset_util.bytes_list_feature(classes_text),
    'boxes/class/label': dataset_util.int64_list_feature(classes),
  }))


def _filter_class(cl):
  if cl == 'Van':
    cl = 'Car'
  elif cl == 'Person_sitting':
    cl = 'Pedestrian'
  return cl


def compute_labels_image(label_data,
                         tf_velo_to_cam,
                         p):

  tf_cam_to_velo = inv(tf_velo_to_cam)
  resolution = 0.15
  length = 60
  width = 60
  grid_map_origin_idx = np.array([length / 2 + 29.99,
                                  width / 2 + 0])
  logging.debug('Origin: ' + str(grid_map_origin_idx))
  labels_velo = []
  labels_t = []
  labels = []
  for obj in label_data:
    if obj.type not in ['Car', 'Pedestrian', 'Cyclist']:
        continue
    obj.type = _filter_class(obj.type)

    corners_obj = np.array([[obj.l / 2, obj.l / 2, - obj.l / 2, - obj.l / 2,
                             obj.l / 2, obj.l / 2, - obj.l / 2, - obj.l / 2],
                            [0, 0, 0, 0,
                             - obj.h, - obj.h, - obj.h, - obj.h],
                            [obj.w / 2, - obj.w / 2, - obj.w / 2, obj.w / 2,
                             obj.w / 2, - obj.w / 2, -obj.w / 2, obj.w / 2],
                            [1, 1, 1, 1, 1, 1, 1, 1]])
    tf_cam = np.array([[math.cos(obj.ry), 0, math.sin(obj.ry), obj.tx],
                          [0, 1, 0, obj.ty],
                          [- math.sin(obj.ry), 0, math.cos(obj.ry), obj.tz],
                          [0, 0, 0, 1]])
    corners_cam = tf_cam.dot(corners_obj)
    corners_velo = tf_cam_to_velo.dot(corners_cam)
    tf_velo_to_image = np.array([[0, -1, grid_map_origin_idx[1]], [-1, 0, grid_map_origin_idx[0]], [0, 0, 1]])
    corners_velo_x_y = np.array([corners_velo[0], corners_velo[1], [1, 1, 1, 1, 1, 1, 1, 1]])
    corners_image = tf_velo_to_image.dot(corners_velo_x_y)
    corners_image_idx = np.array([corners_image[0] / resolution, corners_image[1] / resolution])

    label_t_cam = np.array([obj.tx, obj.ty, obj.tz, 1])
    label_t_velo = tf_cam_to_velo.dot(label_t_cam)
    label_t_image = tf_velo_to_image.dot(np.array([label_t_velo[0], label_t_velo[1], 1]))
    label_t_image_normalized = np.array([label_t_image[0] / width, label_t_image[1] / length])

    if 0 <= min(corners_image_idx[0])\
            and max(corners_image_idx[0]) < width / resolution \
            and 0 <= min(corners_image_idx[1])\
            and max(corners_image_idx[1]) < length / resolution:
      labels_velo.append(corners_image_idx)
      labels_t.append(label_t_image_normalized)
      labels.append(obj)
  return labels_velo, labels_t, labels


def create_tf_record(fn_in, fn_out):

  with open(fn_in) as f:
    examples = f.read().splitlines()
    if FLAGS.subset != 0:
      examples = examples[:FLAGS.subset]

  label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map)
  writer = tf.python_io.TFRecordWriter(fn_out)
  params = read_params(FLAGS.param)
  logging.debug('Params: ' + str(params))
  for idx, example in enumerate(examples):
    if example == '000015':
        continue
    logging.info('Reading ' + example)
    label_data = read_label_file(os.path.join(FLAGS.labels, example + '.txt'))
    tf_velo_to_cam = _read_calib(os.path.join(FLAGS.calib, example + '.txt'))
    labels_image, labels_t, labels = compute_labels_image(label_data, tf_velo_to_cam, params)
    tf_example = dict_to_tf_example(labels_image, labels_t, labels, params, label_map_dict, FLAGS.data, FLAGS.data_mask, example)
    writer.write(tf_example.SerializeToString())


def main(_):

  create_tf_record(FLAGS.train_set, os.path.join(FLAGS.output, 'training.record'))
  create_tf_record(FLAGS.val_set, os.path.join(FLAGS.output, 'validation.record'))


if __name__ == '__main__':
  flags.mark_flags_as_required(['data', 'data_mask', 'param'])
  app.run(main)
