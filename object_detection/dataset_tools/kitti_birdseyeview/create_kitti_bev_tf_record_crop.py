import os
import linecache
import math
import yaml
import cv2
import numpy as np
from numpy.linalg import inv
import tensorflow as tf

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

flags = tf.app.flags
flags.DEFINE_string('data', '', 'Directory to grid maps.')
flags.DEFINE_string('param', '', 'Directory to grid map parameter file.')
flags.DEFINE_string('labels', '', 'Directory to kitti labels.')
flags.DEFINE_string('calib', '', 'Directory to kitti calibrations.')
flags.DEFINE_string('output', '', 'Path to directory to output TFRecords.')
flags.DEFINE_string('label_map', 'data/kitti_object_label_map.pbtxt',
                    'Path to label map proto')
FLAGS = flags.FLAGS



class Label:
    def __init__(self,
                 type,
                 truncation,
                 occlusion,
                 alpha,
                 x1,
                 y1,
                 x2,
                 y2,
                 h,
                 w,
                 l,
                 tx,
                 ty,
                 tz,
                 ry):
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

def read_calib(path_calib,
               idx):
    str_trans = linecache.getline(path_calib, idx).split(' ')[1:]
    list_trans = [float(i) for i in str_trans]
    trans = np.array([[list_trans[0], list_trans[1], list_trans[2], list_trans[3]],
                      [list_trans[4], list_trans[5], list_trans[6], list_trans[7]],
                      [list_trans[8], list_trans[9], list_trans[10], list_trans[11]],
                      [0, 0, 0, 1]])
    return trans

def read_labels(path_label):
    label_file = open(path_label)
    labels = []
    for idx, line in enumerate(label_file):
        str_label = line.split(' ')
        label_struct = Label(str_label[0], float(str_label[1]), float(str_label[2]), float(str_label[3]),
                             float(str_label[4]), float(str_label[5]), float(str_label[6]), float(str_label[7]),
                             float(str_label[8]), float(str_label[9]), float(str_label[10]), float(str_label[11]),
                             float(str_label[12]), float(str_label[13]), float(str_label[14]))
        labels.append(label_struct)
    return labels

def dict_to_tf_example(labels_image,
                       label_data,
                       crop_size,
                       params,
                       label_map_dict,
                       image_dir,
                       image_prefix):
  img_name_hits = image_prefix + '_hits.png'
  img_name_obs = image_prefix + '_observations.png'
  img_name_int = image_prefix + '_intensity.png'
  img_name_zmin = image_prefix + '_zmin.png'
  img_name_zmax = image_prefix + '_zmax.png'

  img_path_hits = os.path.join(image_dir,img_name_hits)
  img_path_obs = os.path.join(image_dir, img_name_obs)
  img_path_int = os.path.join(image_dir, img_name_int)
  img_path_zmin = os.path.join(image_dir, img_name_zmin)
  img_path_zmax = os.path.join(image_dir, img_name_zmax)

  image_hits = cv2.imread(img_path_hits,0)
  image_obs = cv2.imread(img_path_obs,0)
  image_int = cv2.imread(img_path_int,0)
  image_zmin = cv2.imread(img_path_zmin,0)
  image_zmax = cv2.imread(img_path_zmax, 0)
  inputs_stacked = np.stack([image_hits, image_obs, image_int, image_zmin, image_zmax], axis=-1)
  length_crop_diff = inputs_stacked.shape[0] - crop_size[1]
  width_crop_diff = (inputs_stacked.shape[1] - crop_size[0]) / 2
  inputs_stacked = inputs_stacked[int(length_crop_diff):inputs_stacked.shape[0],
               int(width_crop_diff):int(inputs_stacked.shape[1]-width_crop_diff)]
  print(inputs_stacked.shape)
  encoded_inputs = inputs_stacked.tostring()

  width = crop_size[0]
  height = crop_size[1]

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
    #print('xmin', int(min(label_img[0])))
    #print('xmax', int(max(label_img[0])))
    #print('ymin', int(min(label_img[1])))
    #print('ymax', int(max(label_img[1])))
    xmin.append(int(min(label_img[0])) / width)
    ymin.append(int(min(label_img[1])) / height)
    xmax.append(int(max(label_img[0])) / width)
    ymax.append(int(max(label_img[1])) / height)
    x_c.append((int(min(label_img[0])) + int(max(label_img[0]))) / (2 * width))
    y_c.append((int(min(label_img[1])) + int(max(label_img[1]))) / (2 * height))
    angle_rad = label_data[idx].ry
    #print('angle', angle_rad)
    angle.append(angle_rad * 180 / 3.141)
    #print('angle', angle)
    sin_angle.append(math.sin(2 * angle_rad))
    cos_angle.append(math.cos(2 * angle_rad))
    vec_s_x = math.cos(angle_rad)
    vec_s_y = math.sin(angle_rad)
    w_p = label_data[idx].w / params['batch_processor']['resolution']
    #print('w_p', w_p)
    w_p_s = w_p * math.sqrt(vec_s_x * vec_s_x / (height * height) + vec_s_y * vec_s_y / (width * width))
    #print('w_p_s', w_p_s)
    w.append(w_p_s)
    l_p = label_data[idx].l / params['batch_processor']['resolution']
    #print('l_p', l_p)
    l_p_s = l_p * math.sqrt(vec_s_x * vec_s_x / (width * width) + vec_s_y * vec_s_y / (height * height))
    #print('l_p_s', l_p_s)
    h.append(l_p_s)
    class_name = label_data[idx].type
    #print('type', class_name)
    classes_text.append(class_name.encode('utf8'))
    classes.append(label_map_dict[class_name])

  example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(
          (image_prefix + '.png').encode('utf8')),
      'image/source_id': dataset_util.bytes_feature(
          (image_prefix + '.png').encode('utf8')),
      'image/channels': dataset_util.int64_feature(inputs_stacked.shape[2]),
      'image/encoded': dataset_util.bytes_feature(encoded_inputs),
      'image/format': dataset_util.bytes_feature('png'.encode('utf8')),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
      'image/object/bboxrot/x_c': dataset_util.float_list_feature(x_c),
      'image/object/bboxrot/y_c': dataset_util.float_list_feature(y_c),
      'image/object/bboxrot/w': dataset_util.float_list_feature(w),
      'image/object/bboxrot/h': dataset_util.float_list_feature(h),
      'image/object/bboxrot/angle': dataset_util.float_list_feature(angle),
      'image/object/bboxrot/sin_angle': dataset_util.float_list_feature(sin_angle),
      'image/object/bboxrot/cos_angle': dataset_util.float_list_feature(cos_angle),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
  }))
  return example

def compute_labels_image(label_data,
                         velo_to_cam,
                         crop_size,
                         params):
    cam_to_velo = inv(velo_to_cam)
    resolution = params['batch_processor']['resolution']
    length = params['batch_processor']['length']
    width = params['batch_processor']['width']
    length_crop = crop_size[1] * resolution
    width_crop = crop_size[0] * resolution
    if length < length_crop or width < width_crop:
        raise ValueError('Crop size is higher than grid map size')
    length_offset = params['batch_processor']['length_offset']
    width_offset = params['batch_processor']['width_offset']
    grid_map_origin_idx = np.array([length_crop + length_offset - length / 2, width_crop / 2 + width_offset])
    print(grid_map_origin_idx)
    labels_velo = []
    labels = []
    for obj in label_data:
        if obj.type == 'DontCare':
            continue
        corners_obj = np.array([[obj.l / 2, obj.l / 2, - obj.l / 2, - obj.l / 2,
                                 obj.l / 2, obj.l / 2, - obj.l / 2, - obj.l / 2],
                                [0, 0, 0, 0,
                                 - obj.h, - obj.h, - obj.h, - obj.h],
                                [obj.w / 2, - obj.w / 2, - obj.w / 2, obj.w / 2,
                                 obj.w / 2, - obj.w / 2, -obj.w / 2, obj.w / 2],
                                [1, 1, 1, 1, 1, 1, 1, 1]])
        trans_cam = np.array([[math.cos(obj.ry), 0, math.sin(obj.ry), obj.tx],
                              [0, 1, 0, obj.ty],
                              [- math.sin(obj.ry), 0, math.cos(obj.ry), obj.tz],
                              [0, 0, 0, 1]])
        corners_cam = trans_cam.dot(corners_obj)
        corners_velo = cam_to_velo.dot(corners_cam)
        velo_to_image = np.array(
            [[0, -1, grid_map_origin_idx[1]], [-1, 0, grid_map_origin_idx[0]], [0, 0, 1]])
        #print(velo_to_image)
        corners_velo_x_y = np.array([corners_velo[0], corners_velo[1],
                                 [1, 1, 1, 1, 1, 1, 1, 1]])
        corners_image = velo_to_image.dot(corners_velo_x_y)
        corners_image_idx = np.array([corners_image[0]/resolution, corners_image[1]/resolution])
        #print(obj.type)
        #print('max:', max(corners_image_idx[0]), max(corners_image_idx[1]))
        #print('min:', min(corners_image_idx[0]), min(corners_image_idx[1]))
        if (((min(corners_image_idx[0]) and min(corners_image_idx[1])) > 0) and
                (max(corners_image_idx[0]) < crop_size[0]) and (max(corners_image_idx[1]) < crop_size[1])):
            labels.append(obj)
            labels_velo.append(corners_image_idx)
    return labels_velo, labels

def visualize_results(image_dir,
                      image_prefix,
                      labels_image,
                      crop_size,
                      output_dir):
    img_name_prob = image_prefix + '_probability.png'
    img_path_prob = os.path.join(image_dir, img_name_prob)
    img_output_path = os.path.join(output_dir, img_name_prob)
    print('img_path_prob', img_output_path)
    img_prob = cv2.imread(img_path_prob)
    length_crop_diff = img_prob.shape[0]- crop_size[1]
    width_crop_diff = (img_prob.shape[1] - crop_size[0]) / 2
    img_prob = img_prob[int(length_crop_diff):img_prob.shape[0],
               int(width_crop_diff):int(img_prob.shape[1] - width_crop_diff)]
    for label_img in labels_image:
        x_min = int(min(label_img[0]))
        x_max = int(max(label_img[0]))
        y_min = int(min(label_img[1]))
        y_max = int(max(label_img[1]))
        cv2.rectangle(img_prob, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
        cv2.line(img_prob, (int(label_img[0][0]),int(label_img[1][0])),
                 (int(label_img[0][1]), int(label_img[1][1])), (0, 255, 0), 2)
        cv2.line(img_prob, (int(label_img[0][1]),int(label_img[1][1])),
                 (int(label_img[0][2]), int(label_img[1][2])), (0, 255, 0), 2)
        cv2.line(img_prob, (int(label_img[0][2]),int(label_img[1][2])),
                 (int(label_img[0][3]), int(label_img[1][3])), (0, 255, 0), 2)
        cv2.line(img_prob, (int(label_img[0][3]),int(label_img[1][3])),
                 (int(label_img[0][0]), int(label_img[1][0])), (0, 255, 0), 2)
    cv2.imwrite(img_output_path, img_prob)


def create_tf_record(output_filename,
                     label_map_dict,
                     label_dir,
                     calib_dir,
                     image_dir,
                     param_dir,
                     examples,
                     crop_size,
                     vis_results,
                     vis_output_dir):

  writer = tf.python_io.TFRecordWriter(output_filename)
  params = read_params(param_dir)
  print(params)
  for idx, example in enumerate(examples):
    label_path = os.path.join(label_dir, example + '.txt')
    calib_path = os.path.join(calib_dir, example + '.txt')
    label_data = read_labels(label_path)
    velo_to_cam = read_calib(calib_path, 6)
    labels_image, labels = compute_labels_image(label_data, velo_to_cam, crop_size, params)
    tf_example = dict_to_tf_example(labels_image, labels, crop_size, params, label_map_dict, image_dir, example)
    if (vis_results):
        visualize_results(image_dir, example, labels_image, crop_size, vis_output_dir)
    writer.write(tf_example.SerializeToString())

  writer.close()

def main(_):
    vis_results = True
    crop_size = [1600, 1600] # [width, length]

    with open('/mrtstorage/datasets/kitti/object_detection/split_at_5150/train_set_split_at_5150.txt') as f:
        examples_train = f.read().splitlines()

    with open('/mrtstorage/datasets/kitti/object_detection/split_at_5150/val_set_split_at_5150.txt') as f:
        examples_val= f.read().splitlines()

    label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map)
    train_output_path = os.path.join(FLAGS.output, 'kitti_train.record')
    val_output_path = os.path.join(FLAGS.output, 'kitti_val.record')
    vis_output_dir = os.path.join(FLAGS.output, 'Debug')

    create_tf_record(train_output_path, label_map_dict, FLAGS.labels, FLAGS.calib, FLAGS.data,
                     FLAGS.param, examples_train, crop_size, vis_results, vis_output_dir)
    create_tf_record(val_output_path, label_map_dict, FLAGS.labels, FLAGS.calib, FLAGS.data,
                     FLAGS.param, examples_val, crop_size, vis_results, vis_output_dir)

if __name__ == '__main__':
  tf.app.run()