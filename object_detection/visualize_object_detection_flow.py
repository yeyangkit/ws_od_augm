import cv2
import math
import os
import time
import numpy as np
import tensorflow as tf

from absl import flags
from google.protobuf import text_format
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from object_detection.protos import pipeline_pb2
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

FLAGS = flags.FLAGS
flags.DEFINE_string('data', None, 'Directory to grid maps.')
flags.DEFINE_string('param', None, 'Directory to grid map parameter file.')
flags.DEFINE_string('pipeline_config_path', None, 'Directory to pipeline config file.')
flags.DEFINE_string('graph', None, 'Directory to frozen inferecne graph.')
flags.DEFINE_string('nuscenes', None, 'Directory to nuscenes data.')
flags.DEFINE_string('output', '/tmp/', 'Output directory of json file.')
flags.DEFINE_string('label_map', '/mrtstorage/datasets/nuscenes/nuscenes_object_label_map.pbtxt',
                    'Path to label map proto')


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

def read_images(data_dir, prefix):
    image_path_det = os.path.join(data_dir, prefix + '_detections_cartesian.png')
    image_path_obs = os.path.join(data_dir, prefix + '_observations_cartesian.png')
    image_path_int = os.path.join(data_dir, prefix + '_intensity_cartesian.png')
    image_path_zmin = os.path.join(data_dir, prefix + '_z_min_detections_cartesian.png')
    image_path_zmax = os.path.join(data_dir, prefix + '_z_max_detections_cartesian.png')
    image_path_occ = os.path.join(data_dir, prefix + '_z_max_occlusions_cartesian.png')
    image_det = cv2.imread(image_path_det, 0)
    image_obs = cv2.imread(image_path_obs, 0)
    image_int = cv2.imread(image_path_int, 0)
    image_zmin = cv2.imread(image_path_zmin, 0)
    image_zmax = cv2.imread(image_path_zmax, 0)
    image_occ = cv2.imread(image_path_occ, 0)
    image_stacked = np.stack([image_det, image_int, image_zmin, image_zmax], axis=-1)
    detection_mask = image_det > 0.0001
    observation_mask = image_obs > 0.0001
    return np.expand_dims(image_stacked, axis=0), detection_mask, observation_mask

def resize_flow(flow, height, width):
    old_height, old_width, _ = flow.shape
    flow = cv2.resize(flow, (width, height), interpolation=cv2.INTER_LINEAR)
    flow_v, flow_u = cv2.split(flow)
    flow_v = flow_v * height / old_height
    flow_u = flow_u * width / old_width
    flow = cv2.merge((flow_v, flow_u))
    return flow

def flow_to_image_rgb(flow):
    flow_mag_max = 3.0
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    flow_magnitude, flow_angle = cv2.cartToPolar(flow[..., 0].astype(np.float32), flow[..., 1].astype(np.float32))

    # A couple times, we've gotten NaNs out of the above...
    nans = np.isnan(flow_magnitude)
    if np.any(nans):
        nans = np.where(nans)
        flow_magnitude[nans] = 0.

    # Normalize
    hsv[..., 0] = flow_angle * 180 / np.pi / 2
    hsv[..., 1] = np.minimum(flow_magnitude * 255 / flow_mag_max, 255)
    hsv[..., 2] = hsv[..., 1]
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img

def create_border_mask(flow):
    height = flow.shape[0]
    width = flow.shape[1]
    border_mask = np.zeros((height, width), dtype=bool)
    border_size = np.array([height, width]) * 0.1
    border_mask[int(border_size[0]):int(height-border_size[0]), int(border_size[1]):int(width-border_size[1])] = True
    return border_mask

def calculate_object_flow(flow_fw, det_mask, box_aligned):
    height, width = det_mask.shape
    y_min = box_aligned[0] * height
    x_min = box_aligned[1] * width
    y_max = box_aligned[2] * height
    x_max = box_aligned[3] * width
    flow_fw_box = flow_fw[int(y_min):math.ceil(y_max), int(x_min):math.ceil(x_max)]
    det_mask_box = det_mask[int(y_min):math.ceil(y_max), int(x_min):math.ceil(x_max)]
    flow_fw_det_u = []
    flow_fw_det_v = []
    for (v, u), val in np.ndenumerate(det_mask_box):
        if val:
            flow_fw_det_u.append(flow_fw_box[v, u, 1])
            flow_fw_det_v.append(flow_fw_box[v, u, 0])
    return np.asarray(flow_fw_det_u).mean(), np.asarray(flow_fw_det_v).mean()

def draw_object_flow(img, flow_fw, det_mask, box_inclined, box_aligned, scaling_factor):
    flow_mag_max = 3.0
    flow_u, flow_v = calculate_object_flow(flow_fw, det_mask, box_aligned)
    flow_magnitude, flow_angle = cv2.cartToPolar(np.array(flow_v), np.array(flow_u))
    hsv = np.zeros((1, 1, 3), dtype=np.uint8)
    hsv[..., 0] = flow_angle * 180 / np.pi / 2
    hsv[..., 1] = np.minimum(flow_magnitude * 255 / flow_mag_max, 255)
    hsv[..., 2] = hsv[..., 1]
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    color = (int(bgr[0,0,0]), int(bgr[0,0,1]), int(bgr[0,0,2]))
    height, width, _ = img.shape
    x_c = box_inclined[0] * width
    y_c = box_inclined[1] * height
    if np.linalg.norm(np.array([flow_u, flow_v])) > 0.5:
        pt1 = (int(x_c), int(y_c))
        pt2 = (int(x_c + flow_u * scaling_factor), int(y_c + flow_v * scaling_factor))
        cv2.arrowedLine(img, pt1, pt2, color, 3)

def get_correspondeces_and_grid(flow, mask):
    height = flow.shape[0]
    width = flow.shape[1]
    center = [(height - 1.0) / 2.0, (width - 1.0) / 2.0]
    img_v = np.zeros((height, width))
    img_u = np.zeros((height, width))
    for v in range(height):
        img_v[v, :] = v - center[0]
    for u in range(width):
        img_u[:, u] = u - center[1]
    img_v_warped = img_v - flow[:, :, 0]
    img_u_warped = img_u - flow[:, :, 1]
    grid = []
    corr_target = []
    corr_source = []
    for (v, u), val in np.ndenumerate(mask):
        grid.append([img_u[v, u], img_v[v, u]])
        if val:
            corr_source.append([img_u_warped[v, u], img_v_warped[v, u]])
            corr_target.append([img_u[v, u], img_v[v, u]])
    grid = np.transpose(np.asarray(grid))
    corr_source = np.transpose(np.asarray(corr_source))
    corr_target = np.transpose(np.asarray(corr_target))
    return corr_source, corr_target, grid

def weighted_rigid_alignment(source, target, number_iterations=10):
    def squared_errors_to_weights(R, t, source, target):
        t = np.expand_dims(t, 1)
        distance = np.add(np.dot(R, source), t) - target
        err = np.sum(np.square(distance), axis=0)
        max = np.clip(np.max(err), a_min=1e-30, a_max=1e100)
        weights = (max - err) / max
        weights = np.diag(weights)
        return weights

    m, n = source.shape
    weights = np.diag(np.ones(n))
    for i in range(number_iterations):
        sum_p = np.sum(weights)
        mx = np.sum(np.dot(source, weights), 1) / sum_p
        my = np.sum(np.dot(target, weights), 1) / sum_p
        Xc = source - np.expand_dims(mx, 1)
        Yc = target - np.expand_dims(my, 1)

        Sxy = np.dot(np.dot(Yc, weights), Xc.T) / sum_p
        U,D,V = np.linalg.svd(Sxy, full_matrices=True, compute_uv=True)

        #ToDo: prevent R from being a reflection matrix
        S = np.eye(m)
        R = np.dot(np.dot(U, S), V)

        t = my - np.dot(R, mx)
        weights = squared_errors_to_weights(R, t, source, target)
    return R, t

def estimate_rigid_flow(flow, det_mask):
    # Estimate rigid transformation
    bord_mask = create_border_mask(flow)
    corr_source, corr_target, grid = get_correspondeces_and_grid(flow, det_mask*bord_mask)
    R, t = weighted_rigid_alignment(corr_source, corr_target)
    # Calculate rigid flow
    t = np.expand_dims(t, 1)
    grid_warped = np.dot(R, grid) + t
    grid = np.reshape(np.transpose(grid), (flow.shape[0], flow.shape[1], 2))
    grid_warped = np.reshape(np.transpose(grid_warped), (flow.shape[0], flow.shape[1], 2))
    return np.flip(grid_warped - grid, axis=-1)

def visualize(split, use_10hz_capture_frequency, compensate_rigid_flow, use_feature_aggregation):
    nusc = NuScenes(version='v1.0-trainval', dataroot=FLAGS.nuscenes, verbose=True)
    sensor = 'LIDAR_TOP'

    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.gfile.GFile(FLAGS.pipeline_config_path, 'r') as f:
        text_format.Merge(f.read(), pipeline_config)
    if not pipeline_config.model.HasField('ssd_flow'):
        raise ValueError('Model with flow estimation is required.')

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
            prev_image_tensor = detection_graph.get_tensor_by_name('prev_image_tensor:0')
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            detection_boxes_inclined = detection_graph.get_tensor_by_name('detection_boxes_3d:0')
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            flow_forward = detection_graph.get_tensor_by_name('flow_prediction_fw:0')
            feature_maps_aggr = []
            feature_maps_prev = []
            if use_feature_aggregation:
                fpn_levels = pipeline_config.model.ssd_flow.feature_extractor.fpn
                for feat_idx in range(fpn_levels.min_level, fpn_levels.max_level+1):
                    feature_maps_aggr.append(detection_graph.get_tensor_by_name(
                        'feature_map_aggregated_level_{}'.format(feat_idx)+':0'))
                    feature_maps_prev.append(detection_graph.get_tensor_by_name(
                        'feature_map_prev_level_{}'.format(feat_idx)+':0'))
            scene_splits = create_splits_scenes()
            for scene in nusc.scene:
                if scene['name'] not in ['scene-1062']:
                    continue
                first_sample = nusc.get('sample', scene['first_sample_token'])
                current_token = first_sample['data'][sensor]
                first_inference = True
                while current_token:
                    lidar_top_data = nusc.get('sample_data', current_token)
                    if not lidar_top_data['prev']:
                        current_token = lidar_top_data['next']
                        if use_10hz_capture_frequency:
                            lidar_top_data_next = nusc.get('sample_data', current_token)
                            current_token = lidar_top_data_next['next']
                        continue
                    lidar_top_data_prev = nusc.get('sample_data', lidar_top_data['prev'])
                    if use_10hz_capture_frequency:
                        lidar_top_data_prev = nusc.get('sample_data', lidar_top_data_prev['prev'])
                    # Read input data
                    filename_prefix = os.path.splitext(os.path.splitext(lidar_top_data['filename'])[0])[0]
                    filename_prev_prefix = os.path.splitext(os.path.splitext(lidar_top_data_prev['filename'])[0])[0]
                    image_stacked, det_mask, observation_mask = read_images(FLAGS.data, filename_prefix)
                    image_prev_stacked, det_mask_prev, _ = read_images(FLAGS.data, filename_prev_prefix)
                    # Inference
                    start_time = time.time()
                    if use_feature_aggregation:
                        if first_inference:
                            (boxes_aligned, boxes_inclined, scores, classes, num, flow_fw, features_aggr) = sess.run(
                                [detection_boxes, detection_boxes_inclined, detection_scores, detection_classes,
                                 num_detections, flow_forward, feature_maps_aggr],
                                feed_dict={image_tensor: image_stacked, prev_image_tensor: image_prev_stacked})
                        else:
                            feed_dict = {image_tensor: image_stacked, prev_image_tensor: image_prev_stacked}
                            for tensor, value in zip(feature_maps_prev, features_aggr):
                                feed_dict[tensor] = value
                            (boxes_aligned, boxes_inclined, scores, classes, num, flow_fw, features_aggr) = sess.run(
                                [detection_boxes, detection_boxes_inclined, detection_scores, detection_classes,
                                 num_detections, flow_forward, feature_maps_aggr],
                                feed_dict=feed_dict)
                    else:
                        (boxes_aligned, boxes_inclined, scores, classes, num, flow_fw) = sess.run(
                            [detection_boxes, detection_boxes_inclined, detection_scores, detection_classes,
                             num_detections, flow_forward],
                            feed_dict={image_tensor: image_stacked, prev_image_tensor: image_prev_stacked})
                    print('Inference time:', time.time() - start_time)

                    # Resize flow
                    flow_fw = resize_flow(flow_fw[0], image_stacked.shape[1], image_stacked.shape[2])
                    #cv2.imwrite(os.path.join(FLAGS.output, filename_prefix.split('/')[-1] + '_fw.png'),
                    #            flow_to_image_rgb(flow_fw))

                    # Estimate rigid flow and compensate ego motion
                    if compensate_rigid_flow:
                        rigid_flow = estimate_rigid_flow(flow_fw, det_mask)
                        flow_fw = flow_fw - rigid_flow
                        #flow_rigid_rgb = flow_to_image_rgb(rigid_flow)
                        #cv2.imwrite(os.path.join(FLAGS.output, filename_prefix.split('/')[-1] + '_rigid.png'),
                        #            flow_rigid_rgb)

                    # Flow to rgb image
                    flow_rgb = flow_to_image_rgb(flow_fw)

                    # Visualize object detection and scene flow
                    label_map = label_map_util.load_labelmap(FLAGS.label_map)
                    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=10,
                                                                                use_display_name=True)
                    category_index = label_map_util.create_category_index(categories)
                    # Create grid map to visualize
                    image_vis = np.zeros((image_stacked.shape[1], image_stacked.shape[2], 3), dtype=np.uint8)
                    for (v, u), val in np.ndenumerate(observation_mask):
                        if val:
                            image_vis[v, u, :] = 50
                    image_vis_inv = cv2.bitwise_not(image_vis)
                    for (v, u), val in np.ndenumerate(det_mask):
                        if val:
                            image_vis[v, u] = flow_rgb[v, u]
                            image_vis_inv[v, u] = flow_rgb[v, u]

                    # Resize image for visualization
                    scaling_factor = 4
                    height = image_stacked.shape[1] * scaling_factor
                    width = image_stacked.shape[2] * scaling_factor
                    image_vis = cv2.resize(image_vis, (width, height), interpolation=cv2.INTER_LINEAR)
                    image_vis_inv = cv2.resize(image_vis_inv, (width, height), interpolation=cv2.INTER_LINEAR)

                    # Draw object flow as arrow
                    scores = np.squeeze(scores)
                    for i in range(scores.shape[0]):
                        if scores[i] > .3:
                            draw_object_flow(image_vis, flow_fw, det_mask, tuple(np.squeeze(boxes_inclined)[i]),
                                             tuple(np.squeeze(boxes_aligned)[i]), scaling_factor*5)
                            draw_object_flow(image_vis_inv, flow_fw, det_mask, tuple(np.squeeze(boxes_inclined)[i]),
                                             tuple(np.squeeze(boxes_aligned)[i]), scaling_factor*5)

                    # Draw inclined detection box
                    vis_util.visualize_boxes_and_labels_on_image_array(
                        image_vis,
                        np.squeeze(boxes_aligned),
                        np.squeeze(classes).astype(np.int32),
                        np.squeeze(scores),
                        category_index,
                        boxes_3d=np.squeeze(boxes_inclined),
                        min_score_thresh=0.3,
                        use_normalized_coordinates=True,
                        line_thickness=10)
                    vis_util.visualize_boxes_and_labels_on_image_array(
                        image_vis_inv,
                        np.squeeze(boxes_aligned),
                        np.squeeze(classes).astype(np.int32),
                        np.squeeze(scores),
                        category_index,
                        boxes_3d=np.squeeze(boxes_inclined),
                        min_score_thresh=0.3,
                        use_normalized_coordinates=True,
                        line_thickness=10)

                    # Save image
                    print(filename_prefix.split('/')[-1])
                    output_path = os.path.join(FLAGS.output, filename_prefix.split('/')[-1] + '.png')
                    output_path_inv = os.path.join(FLAGS.output, 'inverse', filename_prefix.split('/')[-1] + '.png')
                    cv2.imwrite(output_path, image_vis)
                    cv2.imwrite(output_path_inv, image_vis_inv)

                    first_inference = False
                    current_token = lidar_top_data['next']
                    if use_10hz_capture_frequency:
                        if current_token:
                            lidar_top_data_next = nusc.get('sample_data', current_token)
                            current_token = lidar_top_data_next['next']

def main(_):
    use_10hz_capture_frequency = True
    compensate_rigid_flow = True
    use_feature_aggregation = False
    visualize('val', use_10hz_capture_frequency, compensate_rigid_flow, use_feature_aggregation)


if __name__ == '__main__':
  tf.app.run()