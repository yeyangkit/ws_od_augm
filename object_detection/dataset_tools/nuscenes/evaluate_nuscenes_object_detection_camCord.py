import sys
# sys.path.remove('/opt/mrtsoftware/release/lib/python2.7/dist-packages')
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import json
import math
import os
import time
import yaml
import numpy as np
import tensorflow as tf

from absl import flags
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion
from object_detection.utils import label_map_util

FLAGS = flags.FLAGS
# flags.DEFINE_string('data', None, 'Directory to grid maps.')
# flags.DEFINE_string('param', None, 'Directory to grid map parameter file.')
flags.DEFINE_string('data', '/mrtstorage/datasets/nuscenes/grid_map/15cm_100m/v1.0-trainval_meta', 'Directory to grid maps.')
# flags.DEFINE_string('data_beliefs', '/mrtstorage/projects/grid_map_learning/nuScenes_erzeugte_lidar_gridMaps/output0815NuScenes_singleBeliefs_keyFrame_train', 'Directory to evidential grid maps.')
flags.DEFINE_string('param', '/mrtstorage/datasets/nuscenes/grid_map/15cm_100m/batch_processor_parameters_nuscenes.yaml', 'Directory to grid map parameter file.')
flags.DEFINE_string('graph', None, 'Directory to frozen inferecne graph.')
flags.DEFINE_string('nuscenes', None, 'Directory to nuscenes data.')
flags.DEFINE_string('output', '/tmp/', 'Output directory of json file.')
flags.DEFINE_string('label_map', '/mrtstorage/datasets/nuscenes/nuscenes_object_label_map.pbtxt',
                    'Path to label map proto')
flags.DEFINE_integer('range', None, 'The range of ROI. None if ranges of x and y are 100m.')


def read_params(param_dir):
  with open(param_dir, 'r') as stream:
    params = yaml.load(stream)
    return params

def read_images(data_dir, prefix):
    image_path_det = os.path.join(data_dir, prefix + '_detections_cartesian.png')
    image_path_obs = os.path.join(data_dir, prefix + '_observations_cartesian.png')
    image_path_int = os.path.join(data_dir, prefix + '_intensity_cartesian.png')
    image_path_zmin = os.path.join(data_dir, prefix + '_z_min_detections_cartesian.png')
    image_path_zmax = os.path.join(data_dir, prefix + '_z_max_detections_cartesian.png')
    image_path_occ = os.path.join(data_dir, prefix + '_z_max_occlusions_cartesian.png')
    image_path_ground = os.path.join(data_dir, prefix + '_ground_surface_cartesian.png')

    # image_path_fused_zmax_det = os.path.join(data_dir, prefix + '_z_max_detections_FUSED_cartesian.png')
    # image_path_fused_obs_zmin = os.path.join(data_dir, prefix + '_observations_z_min_FUSED_cartesian.png')

    # image_path_fused_bel_F = os.path.join(data_dir, prefix + '_bel_F_FUSED_cartesian.png')
    # image_path_fused_bel_O = os.path.join(data_dir, prefix + '_bel_O_FUSED_cartesian.png')
    # image_path_fused_bel_U = os.path.join(data_dir, prefix + '_bel_U_FUSED_cartesian.png')


    image_det = cv2.imread(image_path_det, 0)
    image_obs = cv2.imread(image_path_obs, 0)
    image_int = cv2.imread(image_path_int, 0)
    image_zmin = cv2.imread(image_path_zmin, 0)
    image_zmax = cv2.imread(image_path_zmax, 0)
    image_occ = cv2.imread(image_path_occ, 0)
    image_ground = cv2.imread(image_path_ground, 0)

    # image_fused_zmax_det = cv2.imread(image_path_fused_zmax_det, 0)
    # image_fused_obs_zmin = cv2.imread(image_path_fused_obs_zmin, 0)

    # image_fused_bel_F = cv2.imread(image_path_fused_bel_F, 0)
    # image_fused_bel_O = cv2.imread(image_path_fused_bel_O, 0)
    # image_fused_bel_U = cv2.imread(image_path_fused_bel_U, 0)

    # print('image_fused_bel_U.shape')
    # print(image_fused_bel_U.shape)
    #
    # print('image_fused_bel_F.shape')
    # print(image_fused_bel_F.shape)
    #
    # print('image_fused_bel_O.shape')
    # print(image_fused_bel_O.shape)

    image_stacked = np.stack([image_det, image_occ, image_obs, image_int, image_zmin, image_zmax], axis=-1)

    detection_mask = image_det > 0.0001
    return np.expand_dims(image_stacked, axis=0), detection_mask, image_ground, image_zmax

def calculate_object_box(box_aligned, box_inclined, image_ground, image_zmax, object_class, score, p):
    # Grid Map data
    resolution = p['pointcloud_grid_map_interface']['grids']['cartesian']['resolution']['x']
    if resolution - p['pointcloud_grid_map_interface']['grids']['cartesian']['resolution']['y'] > 0.001:
        raise ValueError('Grid Map resolution in x and y direction need to be equal')
    if FLAGS.range is None:
        grid_map_data_origin_idx = np.array(
            [p['pointcloud_grid_map_interface']['grids']['cartesian']['range']['x'] / 2 +
             p['pointcloud_grid_map_interface']['grids']['cartesian']['offset']['x'],
             p['pointcloud_grid_map_interface']['grids']['cartesian']['range']['y'] / 2])
    else:
        grid_map_data_origin_idx = np.array([float(FLAGS.range) / 2, float(FLAGS.range) / 2])
    image_to_velo = np.array([[0, -1, grid_map_data_origin_idx[0]], [-1, 0, grid_map_data_origin_idx[1]], [0, 0, 1]])
    heigth_diff = (p['pointcloud_grid_map_interface']['z_max'] - p['pointcloud_grid_map_interface']['z_min'])
    height_offset = p['pointcloud_grid_map_interface']['z_min']

    # Convert box
    height, width = image_zmax.shape
    y_min = box_aligned[0] * height
    x_min = box_aligned[1] * width
    y_max = box_aligned[2] * height
    x_max = box_aligned[3] * width
    x_c = box_inclined[0] * width
    y_c = box_inclined[1] * height
    w_s = box_inclined[2]
    h_s = box_inclined[3]
    sin_angle = box_inclined[4]
    cos_angle = box_inclined[5]
    angle_rad = math.atan2(sin_angle, cos_angle) / 2
    # Tranform angle from kitti camera to kitti lidar:
    # angle_rad = - angle_rad - math.pi / 2

    vec_s_x = math.cos(angle_rad)
    vec_s_y = math.sin(angle_rad)
    object_width = w_s * resolution / \
                   math.sqrt(vec_s_x * vec_s_x / (height * height) + vec_s_y * vec_s_y / (width * width))
    object_length = h_s * resolution / \
                    math.sqrt(vec_s_x * vec_s_x / (width * width) + vec_s_y * vec_s_y / (height * height))
    image_ground_box = image_ground[int(y_min):math.ceil(y_max),
                       int(x_min):math.ceil(x_max)]
    mean_ground = image_ground_box.mean()
    image_height_max_box = image_zmax[int(y_min):math.ceil(y_max),
                           int(x_min):math.ceil(x_max)]
    height_max = image_height_max_box.max()
    height_max_m = heigth_diff * height_max / 255.0 + height_offset
    mean_ground_m = heigth_diff * mean_ground / 255.0 + height_offset
    if height_max_m - mean_ground_m > 0:
        object_height = height_max_m - mean_ground_m
    else:
        object_height = 0.1

    # Box output in box coordinate system
    object_wlh = [object_width, object_length, object_height]
    object_center = np.array([x_c * resolution, y_c * resolution])
    object_center_velo = image_to_velo.dot(np.append(object_center, 1))
    object_center_velo[2] = (height_max_m + mean_ground_m) / 2
    object_quat = Quaternion(axis=(0, 0, 1), angle=angle_rad)
    return Box(object_center_velo, object_wlh, object_quat, score=score,
               velocity=(0, 0, 0), name=object_class)

def box_to_sample_result(sample_token, box):
    sample_result = dict()
    sample_result['sample_token'] = sample_token
    sample_result['translation'] = box.center.tolist()
    sample_result['size'] = box.wlh.tolist()
    sample_result['rotation'] = box.orientation.q.tolist()
    sample_result['velocity'] = box.velocity.tolist()[:2]  # Only need vx, vy.
    sample_result['detection_name'] = box.name
    sample_result['detection_score'] = box.score
    sample_result['attribute_name'] = ''
    return sample_result

def evaluate(split):
    params = read_params(FLAGS.param)
    nusc = NuScenes(version='v1.0-trainval', dataroot=FLAGS.nuscenes, verbose=True)
    sensor = 'LIDAR_TOP'
    kitti_to_nu_lidar = Quaternion(axis=(0, 0, 1), angle=np.pi / 2)
    meta = {
        'use_camera': False,
        'use_lidar': True,
        'use_radar': False,
        'use_map': False,
        'use_external': False,
    }
    results = {}

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(FLAGS.graph, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            for node in od_graph_def.node:
                if 'BatchMultiClassNonMaxSuppression' in node.name:
                    node.device = '/device:CPU:0'
            tf.import_graph_def(od_graph_def, name='')

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            detection_boxes_inclined = detection_graph.get_tensor_by_name('detection_boxes_3d:0')
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            scene_splits = create_splits_scenes()
            for scene in nusc.scene:
                if scene['name'] not in scene_splits[split]:
                    continue
                current_sample_token = scene['first_sample_token']
                last_sample_token = scene['last_sample_token']
                sample_in_scene = True
                while sample_in_scene:
                    if current_sample_token == last_sample_token:
                        sample_in_scene = False
                    sample = nusc.get('sample', current_sample_token)
                    lidar_top_data = nusc.get('sample_data', sample['data'][sensor])
                    # Get global pose and calibration data
                    ego_pose = nusc.get('ego_pose', lidar_top_data['ego_pose_token'])
                    calib_sensor = nusc.get('calibrated_sensor', lidar_top_data['calibrated_sensor_token'])
                    ego_to_global = transform_matrix(ego_pose['translation'], Quaternion(ego_pose['rotation']))
                    lidar_to_ego = transform_matrix(calib_sensor['translation'], Quaternion(calib_sensor['rotation']))

                    # Read input data
                    filename_prefix = os.path.splitext(os.path.splitext(lidar_top_data['filename'])[0])[0]
                    image_stacked, det_mask, image_ground, image_zmax = read_images(FLAGS.data, filename_prefix)
                    print(image_stacked.shape)
                    # Inference
                    start_time = time.time()
                    (boxes_aligned, boxes_inclined, scores, classes, num) = sess.run(
                        [detection_boxes, detection_boxes_inclined, detection_scores, detection_classes,
                         num_detections],
                        feed_dict={image_tensor: image_stacked})
                    print('Inference time:', time.time() - start_time)

                    # Evaluate object detection
                    label_map = label_map_util.load_labelmap(FLAGS.label_map)
                    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=10,
                                                                                use_display_name=True)
                    category_index = label_map_util.create_category_index(categories)
                    boxes = []
                    scores = np.squeeze(scores)
                    for i in range(scores.shape[0]):
                        if scores[i] > .3:
                            object_class = category_index[int(np.squeeze(classes)[i])]['name']
                            box = calculate_object_box(tuple(np.squeeze(boxes_aligned)[i]),
                                                       tuple(np.squeeze(boxes_inclined)[i]), image_ground, image_zmax,
                                                       object_class, scores[i], params)
                            # Transformation box coordinate system to nuscenes lidar coordinate system
                            box.rotate(kitti_to_nu_lidar)
                            # Transformation nuscenes lidar coordinate system to ego vehicle frame
                            box.rotate(Quaternion(matrix=lidar_to_ego[:3, :3]))
                            box.translate(lidar_to_ego[:3, 3])
                            # Transformation ego vehicle frame to global frame
                            box.rotate(Quaternion(matrix=ego_to_global[:3, :3]))
                            box.translate(ego_to_global[:3, 3])
                            boxes.append(box)
                    # Convert boxes to nuScenes detection challenge result format.
                    sample_results = [box_to_sample_result(current_sample_token, box) for box in boxes]
                    results[current_sample_token] = sample_results

                    current_sample_token = sample['next']

    submission = {
        'meta': meta,
        'results': results
    }
    submission_path = os.path.join(FLAGS.output, 'submission.json')
    with open(submission_path, 'w') as f:
        json.dump(submission, f, indent=2)


def main(_):
    evaluate('val')


if __name__ == '__main__':
  tf.app.run()
