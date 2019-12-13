import cv2
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
import itertools

FLAGS = flags.FLAGS
flags.DEFINE_string('data', '/mrtstorage/datasets/nuscenes/grid_map/15cm_100m/v1.0-trainval_meta',
                    'Directory to grid maps.')
flags.DEFINE_string('data_beliefs',
                    '/mrtstorage/projects/grid_map_learning/nuScenes_erzeugte_lidar_gridMaps/processed0904NuScenes_fused7Layers_keyFrame_trainval',
                    'Directory to evidential grid maps.')
flags.DEFINE_string('param',
                    '/mrtstorage/datasets/nuscenes/grid_map/15cm_100m/batch_processor_parameters_nuscenes.yaml',
                    'Directory to grid map parameter file.')
flags.DEFINE_string('graph', None, 'Directory to frozen inferecne graph.')
flags.DEFINE_string('nuscenes', '/mrtstorage/datasets/nuscenes/data/v1.0-trainval/v1.0-trainval_meta',
                    'Directory to nuscenes data.')
flags.DEFINE_string('output', '/tmp/', 'Output directory of json file.')
flags.DEFINE_string('label_map', '/mrtstorage/datasets/nuscenes/nuscenes_object_label_map.pbtxt',
                    'Path to label map proto')

vis_set = [
    'scene-0401', 'scene-0252', 'scene-1062', 'scene-0075', 'scene-0133', 'scene-0655',
'scene-0400',
  'scene-0251',
  'scene-1061', 'scene-0074', 'scene-0132',
'scene-0402', 'scene-0253', 'scene-1063', 'scene-0076', 'scene-0134',
'scene-0403', 'scene-0254', 'scene-1064', 'scene-0077', 'scene-0135',
'scene-0404', 'scene-0255', 'scene-1065', 'scene-0078', 'scene-0136',
'scene-0291'

]
vis_set_full = [
    'scene-0003', 'scene-0012', 'scene-0013',
    'scene-0014',
    'scene-0015', 'scene-0016',
    'scene-0017', 'scene-0018',
    'scene-0035', 'scene-0036', 'scene-0038', 'scene-0039',
    'scene-0092', 'scene-0093', 'scene-0094', 'scene-0095',
    'scene-0096', 'scene-0097', 'scene-0098', 'scene-0099',
    'scene-0100', 'scene-0101', 'scene-0102', 'scene-0103',
    'scene-0104', 'scene-0105',
    'scene-0106', 'scene-0107',
    'scene-0108', 'scene-0109', 'scene-0110', 'scene-0221',
    'scene-0268', 'scene-0269', 'scene-0270', 'scene-0271',
    'scene-0272', 'scene-0273', 'scene-0274', 'scene-0275',
    'scene-0276', 'scene-0277', 'scene-0278', 'scene-0329',
    'scene-0330', 'scene-0331', 'scene-0332', 'scene-0344',
    'scene-0345', 'scene-0346', 'scene-0519', 'scene-0520',
    'scene-0521', 'scene-0522', 'scene-0523', 'scene-0524',
    'scene-0552', 'scene-0553', 'scene-0554', 'scene-0555',
    'scene-0556', 'scene-0557', 'scene-0558', 'scene-0559',
    'scene-0560', 'scene-0561', 'scene-0562', 'scene-0563',
    'scene-0564', 'scene-0565', 'scene-0625', 'scene-0626',
    'scene-0627', 'scene-0629', 'scene-0630', 'scene-0632',
    'scene-0633', 'scene-0634', 'scene-0635', 'scene-0636',
    'scene-0637', 'scene-0638', 'scene-0770', 'scene-0771',
    'scene-0775', 'scene-0777', 'scene-0778', 'scene-0780',
    'scene-0781', 'scene-0782', 'scene-0783', 'scene-0784',
    'scene-0794', 'scene-0795', 'scene-0796', 'scene-0797',
    'scene-0798', 'scene-0799', 'scene-0800', 'scene-0802',
    'scene-0904', 'scene-0905', 'scene-0906', 'scene-0907',
    'scene-0908', 'scene-0909', 'scene-0910', 'scene-0911',
    'scene-0912', 'scene-0913', 'scene-0914', 'scene-0915',
    'scene-0916', 'scene-0917', 'scene-0919', 'scene-0920',
    'scene-0921', 'scene-0922', 'scene-0923', 'scene-0924',
    'scene-0925', 'scene-0926', 'scene-0927', 'scene-0928',
    'scene-0929', 'scene-0930', 'scene-0931', 'scene-0962',
    'scene-0963', 'scene-0966', 'scene-0967', 'scene-0968',
    'scene-0969', 'scene-0971', 'scene-0972', 'scene-1059',
    'scene-1060', 'scene-1061',
    'scene-1062', 'scene-1063',
    'scene-1064', 'scene-1065', 'scene-1066', 'scene-1067',
    'scene-1068', 'scene-1069', 'scene-1070', 'scene-1071',
    'scene-1072', 'scene-1073'
]


def read_images(data_dir, data_beliefs_dir, prefix):
    print(data_beliefs_dir)
    image_path_det = os.path.join(data_dir, prefix + '_detections_cartesian.png')
    image_path_obs = os.path.join(data_dir, prefix + '_observations_cartesian.png')
    image_path_int = os.path.join(data_dir, prefix + '_intensity_cartesian.png')
    image_path_zmin = os.path.join(data_dir, prefix + '_z_min_detections_cartesian.png')
    image_path_zmax = os.path.join(data_dir, prefix + '_z_max_detections_cartesian.png')
    image_path_occ = os.path.join(data_dir, prefix + '_z_max_occlusions_cartesian.png')
    image_path_ground = os.path.join(data_dir, prefix + '_ground_surface_cartesian.png')
    print(image_path_det)

    image_path_fused_zmax_det = os.path.join(data_beliefs_dir, prefix + '_z_max_detections_FUSED_cartesian.png')
    image_path_fused_obs_zmin = os.path.join(data_beliefs_dir, prefix + '_observations_z_min_FUSED_cartesian.png')
    image_path_fused_bel_F = os.path.join(data_beliefs_dir, prefix + '_bel_F_FUSED_cartesian.png')
    image_path_fused_bel_O = os.path.join(data_beliefs_dir, prefix + '_bel_O_FUSED_cartesian.png')
    image_path_fused_bel_U = os.path.join(data_beliefs_dir, prefix + '_bel_U_FUSED_cartesian.png')

    image_path_fused_zmin_det = os.path.join(data_beliefs_dir, prefix + '_z_min_detections_FUSED_cartesian.png')
    image_path_fused_det_dc = os.path.join(data_beliefs_dir,
                                             prefix + '_detections_drivingCorridor_FUSED_cartesian.png')

    image_det = cv2.imread(image_path_det, 0)
    image_obs = cv2.imread(image_path_obs, 0)
    image_int = cv2.imread(image_path_int, 0)
    image_zmin = cv2.imread(image_path_zmin, 0)
    image_zmax = cv2.imread(image_path_zmax, 0)
    image_occ = cv2.imread(image_path_occ, 0)
    image_ground = cv2.imread(image_path_ground, 0)

    image_fused_zmax_det = cv2.imread(image_path_fused_zmax_det, 0)
    image_fused_obs_zmin = cv2.imread(image_path_fused_obs_zmin, 0)
    image_fused_bel_F = cv2.imread(image_path_fused_bel_F, 0)
    image_fused_bel_O = cv2.imread(image_path_fused_bel_O, 0)
    image_fused_bel_U = cv2.imread(image_path_fused_bel_U, 0)
    print(image_path_fused_bel_F)

    print('image_det.shape')
    height, width = image_det.shape
    print(height)
    print(width)
    # print(channels)

    # print('image_fused_bel_U.shape')
    # a, b = image_fused_bel_U.shape
    # print(a)
    # print(b)

    print('image_fused_bel_F.shape')
    height1, width1 = image_fused_bel_F.shape
    print(height1)
    print(width1)

    print('image_fused_bel_U.shape')
    height21, width21 = image_fused_bel_U.shape
    print(height21)
    print(width21)

    # image_stacked = np.stack(
    #     [image_fused_bel_O, image_fused_bel_F, image_occ, image_det, image_obs, image_int, image_zmin, image_zmax,
    #      image_fused_bel_U], axis=-1)
    #
    image_stacked = np.stack(
        [image_det, image_obs, image_occ, image_int, image_zmin, image_zmax], axis=-1)

    # image_stacked = np.stack(
    #     [image_zmax, image_zmin, image_int, image_occ, image_obs, image_det], axis=-1)

    detection_mask = image_det > 0.0001
    observation_mask = image_obs > 0.0001
    zmax_mask = image_zmax > 0.00001
    occ_mask = image_occ > 0.0001

    fused_zmax_det_mask = image_fused_zmax_det > 0.0001
    fused_bel_F_mask = image_fused_bel_F > 0.0001

    return np.expand_dims(image_stacked, axis=0), detection_mask, fused_bel_F_mask, zmax_mask


def resize_augm(augm, height, width):
    old_height, old_width, _ = augm.shape

    augm = cv2.resize(augm, (height, width), interpolation=cv2.INTER_LINEAR)
    return augm


def augm_to_image_gray(augm_original, mode_norm255=False):
    # augm_mag_max = 3.0
    # augm_max = tf.reduce_max(augm)
    # augm_min = tf.reduce_min(augm)
    # augm_original = tf.squeeze(augm_original) # tf.squeeze(
    augm = np.zeros((augm_original.shape[0], augm_original.shape[1], 3), dtype=np.uint8)
    print(augm_original)
    print("augm")
    if mode_norm255:
        augm[:, :, 0] = augm_original.astype(np.float) * 255
        augm[:, :, 1] = augm_original.astype(np.float) * 255
        augm[:, :, 2] = augm_original.astype(np.float) * 255
    else:
        augm[:, :, 0] = augm_original.astype(np.float)
        augm[:, :, 1] = augm_original.astype(np.float)
        augm[:, :, 2] = augm_original.astype(np.float)
    print(augm)
    print("max--------------------------------")
    augm_max = augm.max()
    print(augm_max)
    print("min--------------------------------")
    augm_min = augm.min()
    print(augm_min)
    # # A couple times, we've gotten NaNs out of the above...
    # nans = np.isnan(augm_magnitude)
    # if np.any(nans):
    #     nans = np.where(nans)
    #     augm_magnitude[nans] = 0.
    #
    # # Normalize
    # hsv[..., 0] = augm_angle * 180 / np.pi / 2
    # hsv[..., 1] = np.minimum(augm_magnitude * 255 / augm_mag_max, 255)
    # hsv[..., 2] = hsv[..., 1]


    return augm


def augm_to_image_gray_8(augm_original, mode_norm255=False):
  # augm_mag_max = 3.0
  # augm_max = tf.reduce_max(augm)
  # augm_min = tf.reduce_min(augm)
  # augm_original = tf.squeeze(augm_original) # tf.squeeze(
  tv=8
  augm = np.zeros((augm_original.shape[0], augm_original.shape[1], 3), dtype=np.uint8)
  print(augm_original)
  tm = np.float32([ [1,0,0],[0,1,tv]])
  augm_original_t = cv2.warpAffine(augm_original,tm, (augm_original.shape[0], augm_original.shape[1]))
  print("augm")
  if mode_norm255:
    augm[tv:,:, 0] = augm_original_t[tv:,:].astype(np.float) * 255
    augm[tv:,:, 1] = augm_original_t[tv:,:].astype(np.float) * 255
    augm[tv:,:, 2] = augm_original_t[tv:,:].astype(np.float) * 255
  else:
    augm[tv:,:, 0] = augm_original_t[tv:,:].astype(np.float)
    augm[tv:,:, 1] = augm_original_t[tv:,:].astype(np.float)
    augm[tv:,:, 2] = augm_original_t[tv:,:].astype(np.float)
  print(augm)
  print("max--------------------------------")
  augm_max = augm.max()
  print(augm_max)
  print("min--------------------------------")
  augm_min = augm.min()
  print(augm_min)
  # # A couple times, we've gotten NaNs out of the above...
  # nans = np.isnan(augm_magnitude)
  # if np.any(nans):
  #     nans = np.where(nans)
  #     augm_magnitude[nans] = 0.
  #
  # # Normalize
  # hsv[..., 0] = augm_angle * 180 / np.pi / 2
  # hsv[..., 1] = np.minimum(augm_magnitude * 255 / augm_mag_max, 255)
  # hsv[..., 2] = hsv[..., 1]

  return augm


def augm_to_image_rgb(augm_original1,augm_original2,augm_original3):
    # augm_mag_max = 3.0
    # augm_max = tf.reduce_max(augm)
    # augm_min = tf.reduce_min(augm)
    # augm_original = tf.squeeze(augm_original) # tf.squeeze(
    augm = np.zeros((augm_original1.shape[0], augm_original1.shape[1], 3), dtype=np.uint8)
    print(augm_original1)
    print("augm")
    augm[:, :, 0] = augm_original1.astype(np.float) * 10
    augm[:, :, 1] = augm_original2.astype(np.float) * 40
    augm[:, :, 2] = augm_original3.astype(np.float) * 25
    print(augm)
    print("max--------------------------------")
    augm_max = augm.max()
    print(augm_max)
    print("min--------------------------------")
    augm_min = augm.min()
    print(augm_min)
    # # A couple times, we've gotten NaNs out of the above...
    # nans = np.isnan(augm_magnitude)
    # if np.any(nans):
    #     nans = np.where(nans)
    #     augm_magnitude[nans] = 0.
    #
    # # Normalize
    # hsv[..., 0] = augm_angle * 180 / np.pi / 2
    # hsv[..., 1] = np.minimum(augm_magnitude * 255 / augm_mag_max, 255)
    # hsv[..., 2] = hsv[..., 1]

    return augm


def augm_to_image_rg( augm_original2, augm_original3):
  # augm_mag_max = 3.0
  # augm_max = tf.reduce_max(augm)
  # augm_min = tf.reduce_min(augm)
  # augm_original = tf.squeeze(augm_original) # tf.squeeze(
  augm = np.zeros((augm_original2.shape[0], augm_original2.shape[1], 3), dtype=np.uint8)

  print("augm1")
  print(augm_original2)
  print("augm2")
  print(augm_original3)
  augm[:, :, 0] = augm_original3.astype(np.float) * 4
  augm[:, :, 1] = augm_original2.astype(np.float) * 40
  augm[:, :, 2] = 0
  print(augm)
  print("max--------------------------------")
  augm_max = augm.max()
  print(augm_max)
  print("min--------------------------------")
  augm_min = augm.min()
  print(augm_min)
  # # A couple times, we've gotten NaNs out of the above...
  # nans = np.isnan(augm_magnitude)
  # if np.any(nans):
  #     nans = np.where(nans)
  #     augm_magnitude[nans] = 0.
  #
  # # Normalize
  # hsv[..., 0] = augm_angle * 180 / np.pi / 2
  # hsv[..., 1] = np.minimum(augm_magnitude * 255 / augm_mag_max, 255)
  # hsv[..., 2] = hsv[..., 1]

  return augm

def create_border_mask(flow):
    height = flow.shape[0]
    width = flow.shape[1]
    border_mask = np.zeros((height, width), dtype=bool)
    border_size = np.array([height, width]) * 0.1
    border_mask[int(border_size[0]):int(height - border_size[0]),
    int(border_size[1]):int(width - border_size[1])] = True
    return border_mask


def visualize(split):
    nusc = NuScenes(version='v1.0-trainval', dataroot=FLAGS.nuscenes, verbose=True)
    sensor = 'LIDAR_TOP'

    # pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    # with tf.gfile.GFile(FLAGS.pipeline_config_path, 'r') as f:
    #     text_format.Merge(f.read(), pipeline_config)
    # if not pipeline_config.model.HasField('ssd_augmentation'):
    #     raise ValueError('Model with ssd_augmentation estimation is required.')



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
            belief_F_prediction = detection_graph.get_tensor_by_name('belief_F_prediction:0')
            belief_O_prediction = detection_graph.get_tensor_by_name('belief_O_prediction:0')
            z_max_detections_prediction = detection_graph.get_tensor_by_name('z_max_detections_prediction:0')
            detections_drivingCorridor_prediction = detection_graph.get_tensor_by_name('detections_drivingCorridor_prediction:0')
            scene_splits = create_splits_scenes()
            for scene in nusc.scene:
                if scene['name'] not in vis_set:
                    continue
                scene_dir = os.path.join(FLAGS.output, scene['name'])
                os.system('mkdir {}'.format(scene_dir))
                folder_inverse = os.path.join(scene_dir, 'inverse')
                folder_color = os.path.join(scene_dir, 'color')
                folder_color_inverse = os.path.join(scene_dir, 'color_inverse')
                os.system('mkdir {}'.format(folder_inverse))
                os.system('mkdir {}'.format(folder_color))
                os.system('mkdir {}'.format(folder_color_inverse))
                folder_belF = os.path.join(scene_dir, 'belF')
                folder_belO = os.path.join(scene_dir, 'belO')
                folder_zMaxDet = os.path.join(scene_dir, 'zMaxDet')
                os.system('mkdir {}'.format(folder_belF))
                os.system('mkdir {}'.format(folder_belO))
                os.system('mkdir {}'.format(folder_zMaxDet))

                folder_belF_clean = os.path.join(scene_dir, 'belF_clean')
                folder_belO_clean = os.path.join(scene_dir, 'belO_clean')
                folder_zMaxDet_clean = os.path.join(scene_dir, 'zMaxDet_clean')
                os.system('mkdir {}'.format(folder_belF_clean))
                os.system('mkdir {}'.format(folder_belO_clean))
                os.system('mkdir {}'.format(folder_zMaxDet_clean))


                current_sample_token = scene['first_sample_token']
                last_sample_token = scene['last_sample_token']
                # first_sample = nusc.get('sample', scene['first_sample_token'])
                # current_token = first_sample['data'][sensor]
                sample_in_scene = True
                first_inference = True
                while sample_in_scene:
                    # while current_token:
                    if current_sample_token == last_sample_token:
                        sample_in_scene = False
                    sample = nusc.get('sample', current_sample_token)
                    lidar_top_data = nusc.get('sample_data', sample['data'][sensor])
                    if first_inference:
                        # current_token = lidar_top_data['next']
                        # if use_10hz_capture_frequency:
                        #    if current_token:
                        #        lidar_top_data_next = nusc.get('sample_data', current_token)
                        #        current_token = lidar_top_data_next['next']
                        current_sample_token = sample['next']
                        first_inference = False
                        continue

                    # Read input data
                    filename_prefix = os.path.splitext(os.path.splitext(lidar_top_data['filename'])[0])[0]
                    image_stacked, det_mask, observation_mask, z_mask = read_images(FLAGS.data, FLAGS.data_beliefs,
                                                                                    filename_prefix)
                    # Inference
                    start_time = time.time()

                    (boxes_aligned, boxes_inclined, scores, classes, num, belief_F_pred, belief_O_pred,
                     z_max_detections_pred,  detections_drivingCorridor_pred) = sess.run(
                        [detection_boxes, detection_boxes_inclined, detection_scores, detection_classes,
                         num_detections, belief_F_prediction, belief_O_prediction, z_max_detections_prediction, detections_drivingCorridor_prediction],
                        feed_dict={image_tensor: image_stacked})
                    print('Inference time:', time.time() - start_time)

                    # Visualize object detection and scene flow
                    label_map = label_map_util.load_labelmap(FLAGS.label_map)
                    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=10,
                                                                                use_display_name=True)
                    category_index = label_map_util.create_category_index(categories)

                    # Create grid map to visualize
                    image_vis = np.zeros((image_stacked.shape[1], image_stacked.shape[2], 3),
                                         dtype=np.uint8)
                    image_vis_inv = np.zeros((image_stacked.shape[1], image_stacked.shape[2], 3),
                                             dtype=np.uint8)

                    # Write augmentation maps
                    # print("image_stacked--------------------------")
                    # print(image_stacked.shape)
                    # print("belief_F_prediction--------------------")
                    # print(belief_F_pred)
                    belief_F_prediction_np = resize_augm(belief_F_pred[0], image_stacked.shape[1], image_stacked.shape[2])
                    belief_O_prediction_np = resize_augm(belief_O_pred[0], image_stacked.shape[1], image_stacked.shape[2])
                    z_max_detections_prediction_np = resize_augm(z_max_detections_pred[0], image_stacked.shape[1], image_stacked.shape[2])
                    detections_drivingCorridor_prediction_np = resize_augm(detections_drivingCorridor_pred[0], image_stacked.shape[1], image_stacked.shape[2])

                    image_bel_F = augm_to_image_gray_8(belief_F_prediction_np, mode_norm255=True)
                    # image_bel_F = cv2.bitwise_not(image_bel_F)
                    image_bel_F_clean = image_bel_F.copy()

                    image_bel_O = augm_to_image_gray_8(belief_O_prediction_np, mode_norm255=True)
                    image_bel_O = cv2.bitwise_not(image_bel_O)
                    image_bel_O_clean = image_bel_O.copy()

                    image_z_max_detections = augm_to_image_gray_8(z_max_detections_prediction_np, mode_norm255=False)
                    image_z_max_detections = cv2.bitwise_not(image_z_max_detections)
                    image_z_max_detections_clean = image_z_max_detections.copy()

                    # image_vis_color = augm_to_image_rgb(detections_drivingCorridor_prediction_np, belief_F_prediction_np, z_max_detections_prediction_np)
                    image_vis_color = augm_to_image_rg(belief_F_prediction_np, z_max_detections_prediction_np)

                    image_vis_color_inv = cv2.bitwise_not(image_vis_color)

                    for (v, u), val in np.ndenumerate(det_mask):
                        if val:
                            image_vis[v, u] = 255
                            image_vis_inv[v, u] = 0

                    image_vis = np.zeros((image_stacked.shape[1], image_stacked.shape[2], 3), dtype=np.uint8)
                    for (v, u), val in np.ndenumerate(observation_mask):
                        if val:
                            image_vis[v, u, :] = 50
                    image_vis_inv = cv2.bitwise_not(image_vis)
                    for (v, u), val in np.ndenumerate(det_mask):
                        if val:
                            image_vis[v, u] = 255
                            image_vis_inv[v, u] = 0

                    # Draw inclined detection box
                    vis_util.visualize_boxes_and_labels_on_image_array(
                        image_vis_color,
                        np.squeeze(boxes_aligned),
                        np.squeeze(classes).astype(np.int32),
                        np.squeeze(scores),
                        category_index,
                        boxes_3d=np.squeeze(boxes_inclined),
                        min_score_thresh=0.23,
                        use_normalized_coordinates=True,
                        line_thickness=3)
                    vis_util.visualize_boxes_and_labels_on_image_array(
                        image_vis_color_inv,
                        np.squeeze(boxes_aligned),
                        np.squeeze(classes).astype(np.int32),
                        np.squeeze(scores),
                        category_index,
                        boxes_3d=np.squeeze(boxes_inclined),
                        min_score_thresh=0.23,
                        use_normalized_coordinates=True,
                        line_thickness=3)
                    vis_util.visualize_boxes_and_labels_on_image_array(
                        image_vis,
                        np.squeeze(boxes_aligned),
                        np.squeeze(classes).astype(np.int32),
                        np.squeeze(scores),
                        category_index,
                        boxes_3d=np.squeeze(boxes_inclined),
                        min_score_thresh=0.23,
                        use_normalized_coordinates=True,
                        line_thickness=3)
                    print(image_vis.shape)
                    vis_util.visualize_boxes_and_labels_on_image_array(
                        image_vis_inv,
                        np.squeeze(boxes_aligned),
                        np.squeeze(classes).astype(np.int32),
                        np.squeeze(scores),
                        category_index,
                        boxes_3d=np.squeeze(boxes_inclined),
                        min_score_thresh=0.23,
                        use_normalized_coordinates=True,
                        line_thickness=3)

                    print(image_bel_F.shape)

                    vis_util.visualize_boxes_and_labels_on_image_array(
                        image_bel_F,
                        np.squeeze(boxes_aligned),
                        np.squeeze(classes).astype(np.int32),
                        np.squeeze(scores),
                        category_index,
                        boxes_3d=np.squeeze(boxes_inclined),
                        min_score_thresh=0.23,
                        use_normalized_coordinates=True,
                        line_thickness=3)


                    vis_util.visualize_boxes_and_labels_on_image_array(
                        image_bel_O,
                        np.squeeze(boxes_aligned),
                        np.squeeze(classes).astype(np.int32),
                        np.squeeze(scores),
                        category_index,
                        boxes_3d=np.squeeze(boxes_inclined),
                        min_score_thresh=0.23,
                        use_normalized_coordinates=True,
                        line_thickness=3)
                    vis_util.visualize_boxes_and_labels_on_image_array(
                        image_z_max_detections,
                        np.squeeze(boxes_aligned),
                        np.squeeze(classes).astype(np.int32),
                        np.squeeze(scores),
                        category_index,
                        boxes_3d=np.squeeze(boxes_inclined),
                        min_score_thresh=0.23,
                        use_normalized_coordinates=True,
                        line_thickness=3)

                    # Save image
                    print(filename_prefix.split('/')[-1])
                    output_path = os.path.join(scene_dir, filename_prefix.split('/')[-1] + '.png')
                    cv2.imwrite(output_path, image_vis)

                    output_path_inv = os.path.join(folder_inverse, filename_prefix.split('/')[-1] + 'inv.png')
                    output_color_path = os.path.join(folder_color, filename_prefix.split('/')[-1] + 'color.png')
                    output_color_path_inv = os.path.join(folder_color_inverse, filename_prefix.split('/')[-1] + 'colorInv.png')

                    output_path_belO = os.path.join(folder_belO, filename_prefix.split('/')[-1] + 'belo.png')
                    output_path_belF = os.path.join(folder_belF, filename_prefix.split('/')[-1] + 'belf.png')
                    output_path_zMaxDet = os.path.join(folder_zMaxDet, filename_prefix.split('/')[-1] + 'zmax.png')

                    output_path_belO_clean = os.path.join(folder_belO_clean, filename_prefix.split('/')[-1] + 'belo_clean.png')
                    output_path_belF_clean = os.path.join(folder_belF_clean, filename_prefix.split('/')[-1] + 'belf_clean.png')
                    output_path_zMaxDet_clean = os.path.join(folder_zMaxDet_clean, filename_prefix.split('/')[-1] + 'zmax_clean.png')

                    cv2.imwrite(output_path_inv, image_vis_inv)
                    cv2.imwrite(output_color_path, image_vis_color)
                    cv2.imwrite(output_color_path_inv, image_vis_color_inv)

                    cv2.imwrite(output_path_belO, image_bel_O)
                    cv2.imwrite(output_path_belF, image_bel_F)
                    cv2.imwrite(output_path_zMaxDet, image_z_max_detections)

                    cv2.imwrite(output_path_belO_clean, image_bel_O_clean)
                    cv2.imwrite(output_path_belF_clean, image_bel_F_clean)
                    cv2.imwrite(output_path_zMaxDet_clean, image_z_max_detections_clean)
                    current_sample_token = sample['next']

                    # current_token = lidar_top_data['next']
                    # if use_10hz_capture_frequency:
                    #     if current_token:
                    #         lidar_top_data_next = nusc.get('sample_data', current_token)
                    #         current_token = lidar_top_data_next['next']


def main(_):
    # use_10hz_capture_frequency = True

    visualize('val')


if __name__ == '__main__':
    tf.app.run()
