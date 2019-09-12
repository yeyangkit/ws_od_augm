import sys

# sys.path.remove('/opt/mrtsoftware/release/lib/python2.7/dist-packages')
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import io
import hashlib
import math
import os
import random
import yaml

import numpy as np
import PIL.Image as pil
import tensorflow as tf
import time

from absl import app
from absl import flags
from absl import logging
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import BoxVisibility
from nuscenes.eval.detection.utils import category_to_detection_name
from nuscenes.utils.splits import create_splits_logs
from pyquaternion import Quaternion
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

FLAGS = flags.FLAGS
flags.DEFINE_string('data', '/mrtstorage/datasets/nuscenes/grid_map/15cm_100m/v1.0-trainval_meta', 'Directory to grid maps.')
flags.DEFINE_string('data_beliefs', '/mrtstorage/projects/grid_map_learning/nuScenes_erzeugte_lidar_gridMaps/processed0904NuScenes_fused7Layers_keyFrame_trainval', 'Directory to evidential grid maps.')
flags.DEFINE_string('param', '/mrtstorage/datasets/nuscenes/grid_map/15cm_100m/batch_processor_parameters_nuscenes.yaml', 'Directory to grid map parameter file.')
flags.DEFINE_string('nuscenes', '/mrtstorage/datasets/nuscenes/data/v1.0-trainval/v1.0-trainval_meta', 'Directory to nuscenes data.')
flags.DEFINE_string('output', '/mrtstorage/projects/grid_map_learning/nuScenes_object_detection/nuScenes_erzeugte_gridMaps_tfrecord/with_30m_10skip_FusedBeliefs0905_withPrev/', 'Path to directory to output TFRecords.')
flags.DEFINE_string('label_map','/mrtstorage/datasets/nuscenes/nuscenes_object_label_map.pbtxt', # '/mrtstorage/datasets/nuscenes/nuscenes_object_label_map.pbtxt', aug15 krenew incompatible with options/flags here
                    'Path to label map proto')


class Label:
    def __init__(self,
                 type,
                 l,
                 w,
                 h,
                 rz):
        self.type = type
        self.l = l
        self.w = w
        self.h = h
        self.rz = rz


def read_params(param_dir):
    with open(param_dir, 'r') as stream:
        params = yaml.load(stream)
        return params


def split_to_samples(nusc, split_logs):
    samples = []
    for sample in nusc.sample:
        scene = nusc.get('scene', sample['scene_token'])
        log = nusc.get('log', scene['log_token'])
        logfile = log['logfile']
        if logfile in split_logs:
            samples.append(sample['token'])
    return samples


def compute_labels_image(nusc, sample, sensor, nu_to_kitti_lidar, p):
    resolution = p['pointcloud_grid_map_interface']['grids']['cartesian']['resolution']['x']
    if resolution - p['pointcloud_grid_map_interface']['grids']['cartesian']['resolution']['y'] > 0.001:
        raise ValueError('Grid Map resolution in x and y direction need to be equal')
    length = p['pointcloud_grid_map_interface']['grids']['cartesian']['range']['x']
    width = p['pointcloud_grid_map_interface']['grids']['cartesian']['range']['y']
    grid_map_origin_idx = np.array(
        [length / 2 + p['pointcloud_grid_map_interface']['grids']['cartesian']['offset']['x'],
         width / 2])
    labels_corners = []
    labels_center = []
    labels_data = []
    for annotation_token in sample['anns']:
        annotation_metadata = nusc.get('sample_annotation', annotation_token)
        if annotation_metadata['num_lidar_pts'] == 0:
            continue
        _, box_lidar, _ = nusc.get_sample_data(sample['data'][sensor], box_vis_level=BoxVisibility.NONE,
                                               selected_anntokens=[annotation_token])
        box_lidar = box_lidar[0]
        box_lidar.rotate(nu_to_kitti_lidar)
        detection_name = category_to_detection_name(annotation_metadata['category_name'])
        if detection_name is None:
            continue
        # corners_obj: 4 * 8 matrix, each clomun indicates a corner (l, w, h) of a 3d bounding box
        corners_obj = np.array(
            [[box_lidar.wlh[1] / 2, box_lidar.wlh[1] / 2, - box_lidar.wlh[1] / 2, - box_lidar.wlh[1] / 2,
              box_lidar.wlh[1] / 2, box_lidar.wlh[1] / 2, - box_lidar.wlh[1] / 2, - box_lidar.wlh[1] / 2],
             [box_lidar.wlh[0] / 2, - box_lidar.wlh[0] / 2, - box_lidar.wlh[0] / 2, box_lidar.wlh[0] / 2,
              box_lidar.wlh[0] / 2, - box_lidar.wlh[0] / 2, -box_lidar.wlh[0] / 2, box_lidar.wlh[0] / 2],
             [box_lidar.wlh[2] / 2, box_lidar.wlh[2] / 2, box_lidar.wlh[2] / 2, box_lidar.wlh[2] / 2,
              - box_lidar.wlh[2] / 2, - box_lidar.wlh[2] / 2, - box_lidar.wlh[2] / 2, - box_lidar.wlh[2] / 2],
             [1, 1, 1, 1, 1, 1, 1, 1]])

        # tf_velo: box 3d pose affine transformation with respect to lidar origin
        tf_velo = np.array(
            [[box_lidar.rotation_matrix[0][0], box_lidar.rotation_matrix[0][1], box_lidar.rotation_matrix[0][2],
              box_lidar.center[0]],
             [box_lidar.rotation_matrix[1][0], box_lidar.rotation_matrix[1][1], box_lidar.rotation_matrix[1][2],
              box_lidar.center[1]],
             [box_lidar.rotation_matrix[2][0], box_lidar.rotation_matrix[2][1], box_lidar.rotation_matrix[2][2],
              box_lidar.center[2]],
             [0, 0, 0, 1]])

        # tf_velo_to_image: affine transformation from lidar coordinate to image coordinate
        tf_velo_to_image = np.array([[0, -1, grid_map_origin_idx[1]], [-1, 0, grid_map_origin_idx[0]], [0, 0, 1]])

        # corners_velo: corners in lidar coordinate system
        # corners_velo_x_y: corners_velo in x-y BEV plane
        corners_velo = tf_velo.dot(corners_obj)
        corners_velo_x_y = np.array([corners_velo[0], corners_velo[1], [1, 1, 1, 1, 1, 1, 1, 1]])

        # corners_image: corners in grid map with unit meter
        # corners_image_idx: 8 * 2 matrix, corners in grid map with unit pixel
        corners_image = tf_velo_to_image.dot(corners_velo_x_y)
        corners_image_idx = np.array([corners_image[0] / resolution, corners_image[1] / resolution])

        # label_t_image_normalized: 2 * 1 vector, box center in grid map with unit pixel
        label_t_velo = np.array([box_lidar.center[0], box_lidar.center[1], box_lidar.center[2], 1])
        label_t_image = tf_velo_to_image.dot(np.array([label_t_velo[0], label_t_velo[1], 1]))
        label_t_image_normalized = np.array([label_t_image[0] / width, label_t_image[1] / length])

        # Convert rotation matrix to yaw angle
        v = np.dot(box_lidar.rotation_matrix, np.array([1, 0, 0]))
        yaw = np.arctan2(v[1], v[0])

        if 0 <= min(corners_image_idx[0]) \
                and max(corners_image_idx[0]) < width / resolution \
                and 0 <= min(corners_image_idx[1]) \
                and max(corners_image_idx[1]) < length / resolution:
            labels_corners.append(corners_image_idx)
            labels_center.append(label_t_image_normalized)
            labels_data.append(Label(type=detection_name,
                                     l=box_lidar.wlh[1],
                                     w=box_lidar.wlh[0],
                                     h=box_lidar.wlh[2],
                                     rz=yaw))
    return labels_corners, labels_center, labels_data


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


def dict_to_tf_example(labels_corners,
                       labels_center,
                       labels_data,
                       params,
                       label_map_dict,
                       image_dir,
                       image_dir_beliefs,
                       image_prefix,
                       image_prev_prefix):
    width = round(params['pointcloud_grid_map_interface']['grids']['cartesian']['range']['y'] /
                  params['pointcloud_grid_map_interface']['grids']['cartesian']['resolution']['y'])
    height = round(params['pointcloud_grid_map_interface']['grids']['cartesian']['range']['x'] /
                   params['pointcloud_grid_map_interface']['grids']['cartesian']['resolution']['x'])
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

    for idx, label_corner in enumerate(labels_corners):
        xmin.append(min(label_corner[0]) / width)
        ymin.append(min(label_corner[1]) / height)
        xmax.append(max(label_corner[0]) / width)
        ymax.append(max(label_corner[1]) / height)
        x_min = min(label_corner[0]) / width
        y_min = min(label_corner[1]) / height
        x_max = max(label_corner[0]) / width
        y_max = max(label_corner[1]) / height
        if (x_min >= 1) or (y_min >= 1) or (x_max >= 1) or (y_max >= 1):
            print(x_min, y_min, x_max, y_max)
            raise ValueError('Box Parameters greather than 1.0')
        if (x_min <= 0) or (y_min <= 0) or (x_max <= 0) or (y_max <= 0):
            raise ValueError('Box Parameters less than 0.0')
        x_c.append(labels_center[idx][0])
        y_c.append(labels_center[idx][1])
        angle_rad = _flipAngle(labels_data[idx].rz)
        angle.append(angle_rad)
        sin_angle.append(math.sin(2 * angle_rad))
        cos_angle.append(math.cos(2 * angle_rad))
        vec_s_x = math.cos(angle_rad)
        vec_s_y = math.sin(angle_rad)

        w_p = labels_data[idx].w / params['pointcloud_grid_map_interface']['grids']['cartesian']['resolution']['y']
        w_p_s = w_p * math.sqrt(vec_s_x * vec_s_x / (width * width) + vec_s_y * vec_s_y / (height * height))
        w.append(w_p_s)

        l_p = labels_data[idx].l / params['pointcloud_grid_map_interface']['grids']['cartesian']['resolution']['x']
        l_p_s = l_p * math.sqrt(vec_s_x * vec_s_x / (height * height) + vec_s_y * vec_s_y / (width * width))
        h.append(l_p_s)

        class_name = labels_data[idx].type
        classes_text.append(class_name.encode('utf8'))
        classes.append(label_map_dict[class_name])

    return tf.train.Example(features=tf.train.Features(feature={
        'id': dataset_util.bytes_feature(image_prefix.encode('utf8')),
        'image/format': dataset_util.bytes_feature('png'.encode('utf8')),

        'layers/height': dataset_util.int64_feature(height),
        'layers/width': dataset_util.int64_feature(width),

        'layers/detections/encoded': dataset_util.bytes_feature(
            _readImage(image_dir, image_prefix, 'detections_cartesian')),
        'layers/observations/encoded': dataset_util.bytes_feature(
            _readImage(image_dir, image_prefix, 'observations_cartesian')),
        'layers/decay_rate/encoded': dataset_util.bytes_feature(
            _readImage(image_dir, image_prefix, 'decay_rate_cartesian')),
        'layers/intensity/encoded': dataset_util.bytes_feature(
            _readImage(image_dir, image_prefix, 'intensity_cartesian')),
        'layers/zmin/encoded': dataset_util.bytes_feature(
            _readImage(image_dir, image_prefix, 'z_min_detections_cartesian')),
        'layers/zmax/encoded': dataset_util.bytes_feature(
            _readImage(image_dir, image_prefix, 'z_max_detections_cartesian')),
        'layers/occlusions/encoded': dataset_util.bytes_feature(
            _readImage(image_dir, image_prefix, 'z_max_occlusions_cartesian')),

        'layers/bel_O_FUSED/encoded': dataset_util.bytes_feature(
            _readImage(image_dir_beliefs, image_prefix, 'bel_O_FUSED_cartesian')),
        'layers/bel_F_FUSED/encoded': dataset_util.bytes_feature(
            _readImage(image_dir_beliefs, image_prefix, 'bel_F_FUSED_cartesian')),
        'layers/bel_U_FUSED/encoded': dataset_util.bytes_feature(
            _readImage(image_dir_beliefs, image_prefix, 'bel_U_FUSED_cartesian')),

        # 'layers/detections_drivingCorridor_FUSED/encoded': dataset_util.bytes_feature(
        #     _readImage(image_dir_beliefs, image_prefix, 'detections_drivingCorridor_FUSED_cartesian')),
        # 'layers/z_max_detections_FUSED/encoded': dataset_util.bytes_feature(
        #     _readImage(image_dir_beliefs, image_prefix, 'z_max_detections_FUSED_cartesian')),
        # 'layers/z_min_detections_FUSED/encoded': dataset_util.bytes_feature(
        #     _readImage(image_dir_beliefs, image_prefix, 'z_min_detections_FUSED_cartesian')),
        # 'layers/observations_z_min_FUSED/encoded': dataset_util.bytes_feature(
        #     _readImage(image_dir_beliefs, image_prefix, 'observations_z_min_FUSED_cartesian')),


        'layers_prev/detections/encoded': dataset_util.bytes_feature(
            _readImage(image_dir, image_prev_prefix, 'detections_cartesian')),
        'layers_prev/observations/encoded': dataset_util.bytes_feature(
            _readImage(image_dir, image_prev_prefix, 'observations_cartesian')),
        'layers_prev/decay_rate/encoded': dataset_util.bytes_feature(
            _readImage(image_dir, image_prev_prefix, 'decay_rate_cartesian')),
        'layers_prev/intensity/encoded': dataset_util.bytes_feature(
            _readImage(image_dir, image_prev_prefix, 'intensity_cartesian')),
        'layers_prev/zmin/encoded': dataset_util.bytes_feature(
            _readImage(image_dir, image_prev_prefix, 'z_min_detections_cartesian')),
        'layers_prev/zmax/encoded': dataset_util.bytes_feature(
            _readImage(image_dir, image_prev_prefix, 'z_max_detections_cartesian')),
        'layers_prev/occlusions/encoded': dataset_util.bytes_feature(
            _readImage(image_dir, image_prev_prefix, 'z_max_occlusions_cartesian')),

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


def create_tf_record(fn_out, split, vis_results):
    label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map)
    writer = tf.python_io.TFRecordWriter(fn_out)
    params = read_params(FLAGS.param)
    logging.debug('Params: ' + str(params))
    nusc = NuScenes(version='v1.0-trainval', dataroot=FLAGS.nuscenes, verbose=True)
    sensor = 'LIDAR_TOP'
    nu_to_kitti_lidar = Quaternion(axis=(0, 0, 1), angle=np.pi / 2).inverse

    split_logs = create_splits_logs(split, nusc)
    # split_logs = selected_train_set_1
    sample_tokens = split_to_samples(nusc, split_logs)
    random.shuffle(sample_tokens)
    print('Number of samples:', len(sample_tokens))

    for sample_token in sample_tokens:
        sample = nusc.get('sample', sample_token)
        lidar_top_data = nusc.get('sample_data', sample['data'][sensor])
        if not lidar_top_data['prev']:
            continue
        lidar_top_data_prev = nusc.get('sample_data', lidar_top_data['prev'])
        labels_corners, labels_center, labels_data = compute_labels_image(nusc, sample, sensor,
                                                                          nu_to_kitti_lidar, params)
        filename = os.path.splitext(os.path.splitext(lidar_top_data['filename'])[0])[0]
        filename_prev = os.path.splitext(os.path.splitext(lidar_top_data_prev['filename'])[0])[0]
        tf_example = dict_to_tf_example(labels_corners, labels_center, labels_data, params, label_map_dict,
                                        FLAGS.data, FLAGS.data_beliefs, filename, filename_prev)
        writer.write(tf_example.SerializeToString())
        if (vis_results):
            visualize_results(FLAGS.data, filename, labels_corners, os.path.join(FLAGS.output, 'Debug'))


def visualize_results(dir,
                      file_name_prefix,
                      labels_image,
                      output_dir):
    img_name = file_name_prefix + '_decay_rate_cartesian.png'
    img_path_vis = os.path.join(dir, img_name)
    img_output_vis = os.path.join(output_dir, os.path.basename(img_name))
    img_prob = cv2.imread(img_path_vis)
    for label_img in labels_image:
        x_min = int(min(label_img[0]))
        x_max = int(max(label_img[0]))
        y_min = int(min(label_img[1]))
        y_max = int(max(label_img[1]))
        cv2.rectangle(img_prob, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
        cv2.line(img_prob, (int(label_img[0][0]), int(label_img[1][0])),
                 (int(label_img[0][1]), int(label_img[1][1])), (0, 255, 0), 2)
        cv2.line(img_prob, (int(label_img[0][1]), int(label_img[1][1])),
                 (int(label_img[0][2]), int(label_img[1][2])), (0, 255, 0), 2)
        cv2.line(img_prob, (int(label_img[0][2]), int(label_img[1][2])),
                 (int(label_img[0][3]), int(label_img[1][3])), (0, 255, 0), 2)
        cv2.line(img_prob, (int(label_img[0][3]), int(label_img[1][3])),
                 (int(label_img[0][0]), int(label_img[1][0])), (0, 255, 0), 2)
    cv2.imwrite(img_output_vis, img_prob)


def main(_):
    vis_results = False
    time.sleep(1800)
    # create_tf_record(os.path.join(FLAGS.output, 'training.record'), 'train', vis_results)
    create_tf_record(os.path.join(FLAGS.output, 'validation.record'), 'val', vis_results)



if __name__ == '__main__':
    flags.mark_flags_as_required(['data', 'param'])
    app.run(main)
