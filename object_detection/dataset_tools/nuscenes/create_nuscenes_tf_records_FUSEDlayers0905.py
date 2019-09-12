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
flags.DEFINE_string('output', '/mrtstorage/projects/grid_map_learning/nuScenes_object_detection/nuScenes_erzeugte_gridMaps_tfrecord/with_30m_10skip_FusedLayers0905/new/', 'Path to directory to output TFRecords.')
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

        'layers/detections_drivingCorridor_FUSED/encoded': dataset_util.bytes_feature(
            _readImage(image_dir_beliefs, image_prefix, 'detections_drivingCorridor_FUSED_cartesian')),
        'layers/z_max_detections_FUSED/encoded': dataset_util.bytes_feature(
            _readImage(image_dir_beliefs, image_prefix, 'z_max_detections_FUSED_cartesian')),
        'layers/z_min_detections_FUSED/encoded': dataset_util.bytes_feature(
            _readImage(image_dir_beliefs, image_prefix, 'z_min_detections_FUSED_cartesian')),
        'layers/observations_z_min_FUSED/encoded': dataset_util.bytes_feature(
            _readImage(image_dir_beliefs, image_prefix, 'observations_z_min_FUSED_cartesian')),


        # 'layers_prev/detections/encoded': dataset_util.bytes_feature(
        #     _readImage(image_dir, image_prev_prefix, 'detections_cartesian')),
        # 'layers_prev/observations/encoded': dataset_util.bytes_feature(
        #     _readImage(image_dir, image_prev_prefix, 'observations_cartesian')),
        # 'layers_prev/decay_rate/encoded': dataset_util.bytes_feature(
        #     _readImage(image_dir, image_prev_prefix, 'decay_rate_cartesian')),
        # 'layers_prev/intensity/encoded': dataset_util.bytes_feature(
        #     _readImage(image_dir, image_prev_prefix, 'intensity_cartesian')),
        # 'layers_prev/zmin/encoded': dataset_util.bytes_feature(
        #     _readImage(image_dir, image_prev_prefix, 'z_min_detections_cartesian')),
        # 'layers_prev/zmax/encoded': dataset_util.bytes_feature(
        #     _readImage(image_dir, image_prev_prefix, 'z_max_detections_cartesian')),
        # 'layers_prev/occlusions/encoded': dataset_util.bytes_feature(
        #     _readImage(image_dir, image_prev_prefix, 'z_max_occlusions_cartesian')),

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
    #
    # selected_train_set_1 = [
    #     'scene-0030', 'scene-0349', 'scene-0001', 'scene-0741', 'scene-0350', 'scene-0002', 'scene-0744', 'scene-0351',
    #     'scene-0746', 'scene-0004', 'scene-0352', 'scene-0005', 'scene-0747', 'scene-0353', 'scene-0354', 'scene-0006',
    #     'scene-0355', 'scene-0749', 'scene-0007', 'scene-0008', 'scene-0750', 'scene-0356', 'scene-0009', 'scene-0010',
    #     'scene-0751', 'scene-0357', 'scene-0011', 'scene-0752', 'scene-0019', 'scene-0358', 'scene-0757', 'scene-0020',
    #     'scene-0758', 'scene-0359', 'scene-0021', 'scene-0360', 'scene-0759', 'scene-0361', 'scene-0022', 'scene-0760',
    #     'scene-0023', 'scene-0362', 'scene-0024', 'scene-0761', 'scene-0363', 'scene-0762', 'scene-0364', 'scene-0763',
    #     'scene-0025', 'scene-0365', 'scene-0764', 'scene-0366', 'scene-0765', 'scene-0026', 'scene-0767', 'scene-0367',
    #     'scene-0027', 'scene-0768', 'scene-0368', 'scene-0769', 'scene-0028', 'scene-0786', 'scene-0369', 'scene-0029',
    #     'scene-0787', 'scene-0789', 'scene-0370', 'scene-0371', 'scene-0031', 'scene-0790', 'scene-0372', 'scene-0032',
    #     'scene-0791', 'scene-0373', 'scene-0792', 'scene-0033', 'scene-0803', 'scene-0034', 'scene-0374', 'scene-0804',
    #     'scene-0041', 'scene-0042', 'scene-0805', 'scene-0043', 'scene-0375', 'scene-0806', 'scene-0376', 'scene-0044',
    #     'scene-0377', 'scene-0808', 'scene-0045', 'scene-0378', 'scene-0809', 'scene-0379', 'scene-0046', 'scene-0810',
    #     'scene-0380', 'scene-0811', 'scene-0047', 'scene-0381', 'scene-0048', 'scene-0382', 'scene-0049', 'scene-0812',
    #     'scene-0050', 'scene-0383', 'scene-0813', 'scene-0051', 'scene-0384', 'scene-0815', 'scene-0052', 'scene-0816',
    #     'scene-0053', 'scene-0385', 'scene-0054', 'scene-0817', 'scene-0386', 'scene-0819', 'scene-0055', 'scene-0056',
    #     'scene-0388', 'scene-0820', 'scene-0057', 'scene-0821', 'scene-0058', 'scene-0389', 'scene-0822', 'scene-0059',
    #     'scene-0390', 'scene-0847', 'scene-0391', 'scene-0848', 'scene-0849', 'scene-0392', 'scene-0850', 'scene-0393',
    #     'scene-0060', 'scene-0851', 'scene-0852', 'scene-0394', 'scene-0853', 'scene-0395', 'scene-0061', 'scene-0396',
    #     'scene-0854', 'scene-0397', 'scene-0062', 'scene-0855', 'scene-0063', 'scene-0398', 'scene-0856', 'scene-0064',
    #     'scene-0858', 'scene-0065', 'scene-0399', 'scene-0860', 'scene-0066', 'scene-0861', 'scene-0400', 'scene-0067',
    #     'scene-0983', 'scene-1050', 'scene-0862', 'scene-1051', 'scene-0401', 'scene-0798', 'scene-0984', 'scene-0003',
    #     'scene-1052', 'scene-0068', 'scene-0402', 'scene-0988', 'scene-0799', 'scene-1053', 'scene-0863', 'scene-0403',
    #     'scene-0069', 'scene-0800', 'scene-1054', 'scene-0405', 'scene-0802', 'scene-0989', 'scene-0864', 'scene-0012',
    #     'scene-1055', 'scene-0406', 'scene-0904', 'scene-0070', 'scene-0013', 'scene-0865', 'scene-0407', 'scene-1056',
    #     'scene-0905', 'scene-0071', 'scene-0990', 'scene-0906', 'scene-0408', 'scene-1057', 'scene-0014', 'scene-0866',
    #     'scene-0991', 'scene-0907', 'scene-1058', 'scene-0072', 'scene-0868', 'scene-1074', 'scene-0015', 'scene-0410',
    #     'scene-1075', 'scene-0869', 'scene-0908', 'scene-1076', 'scene-0411', 'scene-0992', 'scene-0870', 'scene-0016',
    #     'scene-1077', 'scene-0909', 'scene-0073', 'scene-0412', 'scene-0345', 'scene-1078', 'scene-0994', 'scene-0910',
    #     'scene-1079', 'scene-0017', 'scene-0074', 'scene-0413', 'scene-0871', 'scene-0995', 'scene-0911', 'scene-1080',
    #     'scene-0346', 'scene-0912', 'scene-0996', 'scene-0413', 'scene-1081', 'scene-0018', 'scene-0913', 'scene-0914',
    #     'scene-0872', 'scene-0075', 'scene-1082', 'scene-0414', 'scene-0076', 'scene-0997', 'scene-0519', 'scene-0915',
    #     'scene-0873', 'scene-0916', 'scene-0520', 'scene-0120', 'scene-0998', 'scene-1083', 'scene-0415', 'scene-0035',
    #     'scene-0917', 'scene-0875', 'scene-0521', 'scene-0416', 'scene-1084', 'scene-0036', 'scene-0121', 'scene-0999',
    #     'scene-1085', 'scene-0122', 'scene-0919', 'scene-0522', 'scene-0127', 'scene-0038', 'scene-0417', 'scene-0876',
    #     'scene-1000', 'scene-1086', 'scene-0920', 'scene-0523', 'scene-0877', 'scene-0039', 'scene-0128', 'scene-1001',
    #     'scene-0418', 'scene-1087', 'scene-0878', 'scene-0123', 'scene-0524', 'scene-0129', 'scene-0092', 'scene-1002',
    #     'scene-1088', 'scene-0921', 'scene-0419', 'scene-0124', 'scene-0552', 'scene-0130', 'scene-1003', 'scene-0880',
    #     'scene-0093', 'scene-0125', 'scene-1089', 'scene-1004', 'scene-0420', 'scene-0882', 'scene-0922', 'scene-1005',
    #     'scene-0094', 'scene-0126', 'scene-0131', 'scene-1090', 'scene-0553', 'scene-0883', 'scene-1006', 'scene-1091',
    #     'scene-0421', 'scene-0095', 'scene-0923', 'scene-0132', 'scene-0127', 'scene-0924', 'scene-1092', 'scene-1007',
    #     'scene-1093', 'scene-0133', 'scene-0554', 'scene-0925', 'scene-0422', 'scene-0884', 'scene-0128', 'scene-1008',
    #     'scene-0134', 'scene-1009', 'scene-1094', 'scene-0423', 'scene-0096', 'scene-0885', 'scene-0926', 'scene-0129',
    #     'scene-1010', 'scene-1095', 'scene-0555', 'scene-0886', 'scene-0135', 'scene-1011', 'scene-1096', 'scene-0097',
    #     'scene-0130', 'scene-0424', 'scene-0138', 'scene-0927', 'scene-0887', 'scene-1012', 'scene-0556', 'scene-0139',
    #     'scene-1013', 'scene-1097', 'scene-0425', 'scene-0928', 'scene-0149', 'scene-0098', 'scene-1014', 'scene-0888',
    #     'scene-0131', 'scene-0099', 'scene-0187', 'scene-1015', 'scene-0557', 'scene-1098', 'scene-0929', 'scene-0426',
    #     'scene-0889', 'scene-1016', 'scene-0100', 'scene-0188', 'scene-0132', 'scene-1019', 'scene-0890', 'scene-1099',
    #     'scene-0427', 'scene-0930', 'scene-0558', 'scene-1024', 'scene-0133', 'scene-0931', 'scene-0891', 'scene-1100',
    #     'scene-0101', 'scene-1025', 'scene-0231', 'scene-0428', 'scene-0892', 'scene-0559', 'scene-1044', 'scene-0962',
    #     'scene-0102', 'scene-0893', 'scene-0134', 'scene-1101', 'scene-1045', 'scene-0560', 'scene-0429', 'scene-0894',
    #     'scene-1102', 'scene-0232', 'scene-0963', 'scene-1104', 'scene-0103', 'scene-0430', 'scene-0895', 'scene-0135',
    #     'scene-1046', 'scene-1105', 'scene-0431', 'scene-0561', 'scene-0966', 'scene-0896', 'scene-0138', 'scene-1106',
    #     'scene-0233', 'scene-0104', 'scene-1107', 'scene-1108', 'scene-1047', 'scene-0432', 'scene-0897', 'scene-0967',
    #     'scene-0105', 'scene-0139', 'scene-0562', 'scene-0433', 'scene-0898', 'scene-1048', 'scene-1109', 'scene-0106',
    #     'scene-0968', 'scene-0563', 'scene-0234', 'scene-0149', 'scene-0434', 'scene-1049', 'scene-0289', 'scene-0899',
    #     'scene-1110', 'scene-0107', 'scene-0290', 'scene-0435', 'scene-0900', 'scene-0150', 'scene-0291', 'scene-0235',
    #     'scene-0150', 'scene-0108', 'scene-0564', 'scene-0151', 'scene-0292', 'scene-0969', 'scene-0152', 'scene-0901',
    #     'scene-0293', 'scene-0151', 'scene-0436', 'scene-0236', 'scene-0902', 'scene-0154', 'scene-0294', 'scene-0109',
    #     'scene-0237', 'scene-0971', 'scene-0565', 'scene-0152', 'scene-0437', 'scene-0239', 'scene-0903', 'scene-0295',
    #     'scene-0438', 'scene-0154', 'scene-0110', 'scene-0972', 'scene-0625', 'scene-0238', 'scene-0439', 'scene-0240',
    #     'scene-0945', 'scene-0221', 'scene-0626', 'scene-0296', 'scene-0155', 'scene-0440', 'scene-1059', 'scene-0157',
    #     'scene-0653', 'scene-0947', 'scene-0241', 'scene-0441', 'scene-0627', 'scene-0297', 'scene-0268', 'scene-0949',
    #     'scene-0158', 'scene-0442', 'scene-0242', 'scene-0629', 'scene-0654', 'scene-1060', 'scene-0298', 'scene-0269',
    #     'scene-0443', 'scene-1061', 'scene-0299', 'scene-0630', 'scene-0655', 'scene-0952', 'scene-0270', 'scene-0250',
    #     'scene-1062', 'scene-0300', 'scene-0439', 'scene-0251', 'scene-0632', 'scene-0656', 'scene-0440', 'scene-0953',
    #     'scene-1063', 'scene-0444', 'scene-0703', 'scene-0159', 'scene-0633', 'scene-0479', 'scene-0271', 'scene-0445',
    #     'scene-0955', 'scene-0634', 'scene-0657', 'scene-0501', 'scene-0956', 'scene-0704', 'scene-0446', 'scene-1064',
    #     'scene-0635', 'scene-0272', 'scene-0447', 'scene-0957', 'scene-0705', 'scene-0636', 'scene-0958', 'scene-0273',
    #     'scene-0706', 'scene-0448', 'scene-0658', 'scene-0160', 'scene-1065', 'scene-0959', 'scene-0707', 'scene-0449',
    #     'scene-0274', 'scene-0637', 'scene-1066', 'scene-0708', 'scene-0960', 'scene-0638', 'scene-0275', 'scene-0450',
    #     'scene-1067', 'scene-0413', 'scene-0659', 'scene-0413', 'scene-0660', 'scene-0961', 'scene-0161', 'scene-0770',
    #     'scene-0451', 'scene-1068', 'scene-0661', 'scene-0975', 'scene-0452', 'scene-0276', 'scene-0771', 'scene-0976',
    #     'scene-0453', 'scene-0685', 'scene-1069', 'scene-1070', 'scene-0162', 'scene-0277', 'scene-0977', 'scene-0686',
    #     'scene-1071', 'scene-0775', 'scene-0278', 'scene-0978', 'scene-0454', 'scene-1072', 'scene-0979', 'scene-1073',
    #     'scene-0455', 'scene-0777', 'scene-0329', 'scene-0687', 'scene-0980', 'scene-0163', 'scene-0456', 'scene-0981',
    #     'scene-0688', 'scene-0164', 'scene-0457', 'scene-0778', 'scene-0330', 'scene-0982', 'scene-0165', 'scene-0689',
    #     'scene-0458', 'scene-0983', 'scene-0984', 'scene-0166', 'scene-0331', 'scene-0780', 'scene-0459', 'scene-0988',
    #     'scene-0167', 'scene-0695', 'scene-0989', 'scene-0461', 'scene-0696', 'scene-0168', 'scene-0781', 'scene-0697',
    #     'scene-0990', 'scene-0170', 'scene-0332', 'scene-0462', 'scene-0991', 'scene-0698', 'scene-0171', 'scene-0463',
    #     'scene-0700', 'scene-0782', 'scene-0992', 'scene-0344', 'scene-0994', 'scene-0172', 'scene-0783', 'scene-0701',
    #     'scene-0995', 'scene-0784', 'scene-0996', 'scene-0730', 'scene-0464', 'scene-0173', 'scene-0997', 'scene-0731',
    #     'scene-0998', 'scene-0174', 'scene-0794', 'scene-0465', 'scene-0999', 'scene-0175', 'scene-0733', 'scene-1000',
    #     'scene-0467', 'scene-0795', 'scene-0796', 'scene-1001', 'scene-0797', 'scene-0176', 'scene-0734', 'scene-1002',
    #     'scene-0468', 'scene-0177', 'scene-0735', 'scene-0469', 'scene-0471', 'scene-0736', 'scene-0472', 'scene-0474',
    #     'scene-0475', 'scene-0737', 'scene-0178', 'scene-0476', 'scene-0738', 'scene-0477', 'scene-0179', 'scene-0739',
    #     'scene-0478', 'scene-0479', 'scene-0180', 'scene-0480', 'scene-0740', 'scene-0499', 'scene-0500', 'scene-0181',
    #     'scene-0501', 'scene-0502', 'scene-0182', 'scene-0183', 'scene-0504', 'scene-0505', 'scene-0184', 'scene-0506',
    #     'scene-0507', 'scene-0185', 'scene-0508', 'scene-0509', 'scene-0187', 'scene-0510', 'scene-0511', 'scene-0188',
    #     'scene-0512', 'scene-0513', 'scene-0190', 'scene-0514', 'scene-0191', 'scene-0515', 'scene-0517', 'scene-0192',
    #     'scene-0518', 'scene-0525', 'scene-0193', 'scene-0526', 'scene-0527', 'scene-0194', 'scene-0528', 'scene-0195',
    #     'scene-0529', 'scene-0196', 'scene-0530', 'scene-0199', 'scene-0531', 'scene-0200', 'scene-0532', 'scene-0202',
    #     'scene-0533', 'scene-0203', 'scene-0534', 'scene-0535', 'scene-0204', 'scene-0206', 'scene-0536', 'scene-0207',
    #     'scene-0537', 'scene-0208', 'scene-0538', 'scene-0209', 'scene-0539', 'scene-0210', 'scene-0211', 'scene-0541',
    #     'scene-0542', 'scene-0212', 'scene-0213', 'scene-0214', 'scene-0543', 'scene-0218', 'scene-0544', 'scene-0545',
    #     'scene-0219', 'scene-0546', 'scene-0220', 'scene-0222', 'scene-0566', 'scene-0224', 'scene-0568', 'scene-0225',
    #     'scene-0570', 'scene-0226', 'scene-0571', 'scene-0572', 'scene-0227', 'scene-0573', 'scene-0228', 'scene-0574',
    #     'scene-0229', 'scene-0575', 'scene-0230', 'scene-0576', 'scene-0231', 'scene-0232', 'scene-0577', 'scene-0233',
    #     'scene-0578', 'scene-0234', 'scene-0235', 'scene-0580', 'scene-0236', 'scene-0582', 'scene-0237', 'scene-0583',
    #     'scene-0238', 'scene-0584', 'scene-0239', 'scene-0585', 'scene-0240', 'scene-0586', 'scene-0241', 'scene-0587',
    #     'scene-0242', 'scene-0588', 'scene-0243', 'scene-0589', 'scene-0244', 'scene-0590', 'scene-0245', 'scene-0246',
    #     'scene-0591', 'scene-0247', 'scene-0248', 'scene-0592', 'scene-0593', 'scene-0249', 'scene-0250', 'scene-0594',
    #     'scene-0595', 'scene-0251', 'scene-0596', 'scene-0252', 'scene-0597', 'scene-0598', 'scene-0599', 'scene-0600',
    #     'scene-0253', 'scene-0639', 'scene-0640', 'scene-0254', 'scene-0641', 'scene-0642', 'scene-0255', 'scene-0643',
    #     'scene-0644', 'scene-0645', 'scene-0256', 'scene-0646', 'scene-0647', 'scene-0648', 'scene-0649', 'scene-0257',
    #     'scene-0650', 'scene-0651', 'scene-0258', 'scene-0652', 'scene-0259', 'scene-0653', 'scene-0260', 'scene-0654',
    #     'scene-0655', 'scene-0261', 'scene-0656', 'scene-0262', 'scene-0657', 'scene-0658', 'scene-0263', 'scene-0264'
    # ]
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

# def create_tf_record_train_as_val(fn_out, split, vis_results):
#     label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map)
#     writer = tf.python_io.TFRecordWriter(fn_out)
#     params = read_params(FLAGS.param)
#     logging.debug('Params: ' + str(params))
#     nusc = NuScenes(version='v1.0-trainval', dataroot=FLAGS.nuscenes, verbose=True)
#     sensor = 'LIDAR_TOP'
#     nu_to_kitti_lidar = Quaternion(axis=(0, 0, 1), angle=np.pi / 2).inverse
#
#     selected_train_set_1 = [
#         'scene-0002', 'scene-0252', 'scene-0390', 'scene-0480', 'scene-0644', 'scene-0749', 'scene-1017', 'scene-0391',
#         'scene-0645', 'scene-0499', 'scene-0004', 'scene-0392', 'scene-0500', 'scene-1018', 'scene-0005', 'scene-0393',
#         'scene-0646', 'scene-1020', 'scene-0649', 'scene-0891', 'scene-1021', 'scene-0892', 'scene-1022', 'scene-0893',
#         'scene-0894', 'scene-0650', 'scene-0750', 'scene-0895', 'scene-0651', 'scene-1023', 'scene-0896', 'scene-0897',
#         'scene-0001', 'scene-0055', 'scene-0155', 'scene-0191', 'scene-0243', 'scene-0301', 'scene-0373', 'scene-0416',
#         'scene-0452', 'scene-0515', 'scene-0576', 'scene-0662', 'scene-0709', 'scene-0056', 'scene-0517', 'scene-0417',
#         'scene-0710', 'scene-0057', 'scene-0192', 'scene-0006', 'scene-0453', 'scene-0418', 'scene-0577', 'scene-0518',
#         'scene-0058', 'scene-0157', 'scene-0419', 'scene-0059', 'scene-0578', 'scene-0007', 'scene-0663', 'scene-0420',
#         'scene-0580', 'scene-0525', 'scene-0664', 'scene-0008', 'scene-0582', 'scene-0421', 'scene-0374', 'scene-0583',
#         'scene-0009', 'scene-0665', 'scene-0526', 'scene-0010', 'scene-0158', 'scene-0244', 'scene-0422', 'scene-0454',
#         'scene-0193', 'scene-0527', 'scene-0011', 'scene-0711', 'scene-0455', 'scene-0584', 'scene-0423', 'scene-0528',
#         'scene-0456', 'scene-0019', 'scene-0529', 'scene-0457', 'scene-0302', 'scene-0585', 'scene-0194', 'scene-0424',
#         'scene-0303', 'scene-0159', 'scene-0712', 'scene-0020', 'scene-0586', 'scene-0425', 'scene-0530', 'scene-0195',
#         'scene-0304', 'scene-0458', 'scene-0666', 'scene-0426', 'scene-0713', 'scene-0305', 'scene-0196', 'scene-0459',
#         'scene-0375', 'scene-0531', 'scene-0199', 'scene-0245', 'scene-0306', 'scene-0667', 'scene-0376'
#     ]
#     # split_logs = create_splits_logs(split, nusc)
#     split_logs = selected_train_set_1
#     sample_tokens = split_to_samples(nusc, split_logs)
#     random.shuffle(sample_tokens)
#     print('Number of samples:', len(sample_tokens))
#
#     random.shuffle(sample_tokens)
#     print('Number of samples:', len(sample_tokens))
#
#     for sample_token in sample_tokens[1:100]:
#         sample = nusc.get('sample', sample_token)
#         lidar_top_data = nusc.get('sample_data', sample['data'][sensor])
#         if not lidar_top_data['prev']:
#             continue
#         lidar_top_data_prev = nusc.get('sample_data', lidar_top_data['prev'])
#         labels_corners, labels_center, labels_data = compute_labels_image(nusc, sample, sensor,
#                                                                           nu_to_kitti_lidar, params)
#         filename = os.path.splitext(os.path.splitext(lidar_top_data['filename'])[0])[0]
#         filename_prev = os.path.splitext(os.path.splitext(lidar_top_data_prev['filename'])[0])[0]
#         tf_example = dict_to_tf_example(labels_corners, labels_center, labels_data, params, label_map_dict,
#                                         FLAGS.data, FLAGS.data_beliefs, filename, filename_prev)
#         writer.write(tf_example.SerializeToString())
#         if (vis_results):
#             visualize_results(FLAGS.data, filename, labels_corners, os.path.join(FLAGS.output, 'Debug'))

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

    # time.sleep(1800)
    create_tf_record(os.path.join(FLAGS.output, 'training.record'), 'train', vis_results) # erledigt
    # create_tf_record(os.path.join(FLAGS.output, 'validation.record'), 'val', vis_results) # zurzeit nicht vorhanden
    # create_tf_record_train_as_val(os.path.join(FLAGS.output, 'validation.record'), 'train', vis_results)


if __name__ == '__main__':
    flags.mark_flags_as_required(['data', 'param'])
    app.run(main)
