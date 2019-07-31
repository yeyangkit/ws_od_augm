import math
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import BoxVisibility
from nuscenes.eval.detection.utils import category_to_detection_name
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion


# Checks if a matrix is a valid rotation matrix.
def _isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-5


# Calculates rotation matrix to euler angles
def rotationMatrixToEulerAngles(R):
    assert (_isRotationMatrix(R))
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])

class_list = ['car', 'pedestrian', 'bicycle', 'motorcycle', 'truck', 'trailer', 'bus',
              'construction_vehicle', 'barrier', 'traffic_cone']

nuscenes_path = '/mrtstorage/datasets/nuscenes/data/data_LIDAR/v1.0-trainval/v1.0-trainval_meta'
nusc = NuScenes(version='v1.0-trainval', dataroot=nuscenes_path, verbose=True)
x_positions = dict.fromkeys(class_list)
y_positions = dict.fromkeys(class_list)
length = dict.fromkeys(class_list)
width = dict.fromkeys(class_list)
num_lidar_points = dict.fromkeys(class_list)
y_translation = []
yaw = []
sensor = 'LIDAR_TOP'
nu_to_kitti_lidar = Quaternion(axis=(0, 0, 1), angle=np.pi / 2).inverse
for scene in nusc.scene:
    current_sample_token = scene['first_sample_token']
    last_sample_token = scene['last_sample_token']
    while current_sample_token != last_sample_token:
        sample = nusc.get('sample', current_sample_token)
        lidar_top_data = nusc.get('sample_data', sample['data'][sensor])
        if lidar_top_data['prev']:
            ego_pose = nusc.get('ego_pose', lidar_top_data['ego_pose_token'])
            calib_sensor = nusc.get('calibrated_sensor', lidar_top_data['calibrated_sensor_token'])
            lidar_top_data_prev = nusc.get('sample_data', lidar_top_data['prev'])
            ego_pose_prev = nusc.get('ego_pose', lidar_top_data_prev['ego_pose_token'])
            calib_sensor_prev = nusc.get('calibrated_sensor', lidar_top_data_prev['calibrated_sensor_token'])
            ego_to_global = transform_matrix(ego_pose['translation'], Quaternion(ego_pose['rotation']))
            lidar_to_ego = transform_matrix(calib_sensor['translation'], Quaternion(calib_sensor['rotation']))
            ego_to_global_prev = transform_matrix(ego_pose_prev['translation'], Quaternion(ego_pose_prev['rotation']))
            lidar_to_ego_prev = transform_matrix(calib_sensor_prev['translation'],
                                                 Quaternion(calib_sensor_prev['rotation']))
            lidar_to_global = np.dot(ego_to_global, lidar_to_ego)
            lidar_to_global_prev = np.dot(ego_to_global_prev, lidar_to_ego_prev)
            delta = inv(lidar_to_global_prev).dot(lidar_to_global)
            y_translation.append(delta[1, 3])
            delta_angles = rotationMatrixToEulerAngles(delta[0:3, 0:3]) * 180 / np.pi
            yaw.append(delta_angles[2])

        for annotation_token in sample['anns']:
            annotation_metadata = nusc.get('sample_annotation', annotation_token)
            if annotation_metadata['num_lidar_pts'] == 0:
                continue
            detection_name = category_to_detection_name(annotation_metadata['category_name'])
            if detection_name is None:
                continue
            _, box_lidar, _ = nusc.get_sample_data(sample['data'][sensor], box_vis_level=BoxVisibility.NONE,
                                                   selected_anntokens=[annotation_token])
            box_lidar = box_lidar[0]
            box_lidar.rotate(nu_to_kitti_lidar)
            if x_positions[detection_name] is None:
                x_positions[detection_name] = [box_lidar.center[0]]
                y_positions[detection_name] = [box_lidar.center[1]]
                length[detection_name] = [box_lidar.wlh[1]]
                width[detection_name] = [box_lidar.wlh[0]]
                num_lidar_points[detection_name] = [annotation_metadata['num_lidar_pts']]
            else:
                x_positions[detection_name].append(box_lidar.center[0])
                y_positions[detection_name].append(box_lidar.center[1])
                length[detection_name].append(box_lidar.wlh[1])
                width[detection_name].append(box_lidar.wlh[0])
                num_lidar_points[detection_name].append(annotation_metadata['num_lidar_pts'])
        current_sample_token = sample['next']

for class_name in class_list:
    if x_positions[class_name] is None:
        continue
    plt.figure(class_name)
    plt.subplot(2, 2, 1)
    plt.title('Position in x direction')
    plt.hist(x_positions[class_name], bins=50)
    plt.subplot(2, 2, 2)
    plt.title('Position in y direction')
    plt.hist(y_positions[class_name], bins=50)
    plt.subplot(2, 2, 3)
    plt.title('Length')
    plt.hist(length[class_name], bins=50)
    plt.subplot(2, 2, 4)
    plt.title('Width')
    plt.hist(width[class_name], bins=50)
    #plt.title('Number of lidar points')
    #plt.hist(num_lidar_points[class_name], bins=50)

plt.figure('Rotation between successive scans')
plt.title('Yaw')
plt.hist(yaw, bins=50)

plt.figure('Translation between successive scans')
plt.title('Translation in y direction')
plt.hist(y_translation, bins=50)

plt.show()