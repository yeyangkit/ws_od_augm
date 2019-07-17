import os
import sys
sys.path.remove('/opt/mrtsoftware/release/lib/python2.7/dist-packages')
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
print(sys.path)
import linecache
import cv2
import math
import time
import tensorflow as tf
import numpy as np
from shutil import copyfile

from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import label_map_util
from object_detection.utils import dataset_util

flags = tf.app.flags
flags.DEFINE_string('data', '', 'Directory to grid maps.')
flags.DEFINE_string('data_gr', '', 'Directory to grid maps ground.')
flags.DEFINE_string('data_mask', '', 'Directory to occupancy mask.')
flags.DEFINE_string('labels', '', 'Directory to kitti labels.')
flags.DEFINE_string('calib', '', 'Directory to kitti calibrations.')
flags.DEFINE_string('graph', '', 'Path to frozen inference graph.')
flags.DEFINE_string('output', '', 'Path to directory to output labels.')
flags.DEFINE_string('label_map', 'data/kitti_object_label_map.pbtxt',
                    'Path to label map proto')
FLAGS = flags.FLAGS

def write_labels(file_output, object_class, object_corners_image_2d, object_height, object_width, object_length, object_t_cam, object_angle, score):
    file_output.write(object_class)
    file_output.write(' ')
    file_output.write('-1 ')
    file_output.write('-1 ')
    file_output.write('-10 ')
    file_output.write(str(max(min(object_corners_image_2d[0]),0)))
    file_output.write(' ')
    file_output.write(str(max(min(object_corners_image_2d[1]),0)))
    file_output.write(' ')
    file_output.write(str(min(max(object_corners_image_2d[0]),1242)))
    file_output.write(' ')
    file_output.write(str(min(max(object_corners_image_2d[1]),375)))
    file_output.write(' ')
    file_output.write(str(object_height))
    file_output.write(' ')
    file_output.write(str(object_width))
    file_output.write(' ')
    file_output.write(str(object_length))
    file_output.write(' ')
    file_output.write(str(object_t_cam[0]))
    file_output.write(' ')
    file_output.write(str(object_t_cam[1]))
    file_output.write(' ')
    file_output.write(str(object_t_cam[2]))
    file_output.write(' ')
    file_output.write(str(object_angle))
    file_output.write(' ')
    file_output.write(str(score))
    file_output.write('\n')


def read_calib(path_calib, idx):
    str_mat= linecache.getline(path_calib, idx).split(' ')[1:]
    list_mat=[float(i) for i in str_mat]
    if (idx!=5):
        mat = np.array([[list_mat[0], list_mat[1], list_mat[2], list_mat[3]],
                        [list_mat[4], list_mat[5], list_mat[6], list_mat[7]],
                        [list_mat[8], list_mat[9], list_mat[10], list_mat[11]],
                        [0, 0, 0, 1]])
    else:
        mat = np.array([[list_mat[0], list_mat[1], list_mat[2], 0],
                        [list_mat[3], list_mat[4], list_mat[5], 0],
                        [list_mat[6], list_mat[7], list_mat[8], 0],
                        [0, 0, 0, 1]])
    return mat

def create_kitti_labels(output_path, label_map_path, label_dir, calib_dir, image_dir, image_ground_dir, mask_dir, graph_dir, examples):
    grid_map_data_resolution=0.15
    grid_map_data_origin_idx=np.array([60, 30])

    label_map = label_map_util.load_labelmap(label_map_path)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=6, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    path_to_graph = graph_dir
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(path_to_graph, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            mask_tensor = detection_graph.get_tensor_by_name('mask_tensor:0')
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            detection_boxes_3d = detection_graph.get_tensor_by_name('detection_boxes_3d:0')
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            for idx, example in enumerate(examples):
                label_calib_name = '%06d' % (int(example)) + '.txt'
                path_label = os.path.join(label_dir, label_calib_name)
                label_name = '%06d' % (idx) + '.txt'
                output_label = os.path.join(output_path, 'label', label_name)
                output_label_gt = os.path.join(output_path, 'label_gt', label_name)
                copyfile(path_label, output_label_gt)
                path_calib = os.path.join(calib_dir, label_calib_name)
                velo_to_cam = read_calib(path_calib, 6)
                P2 = read_calib(path_calib, 3)
                R0_rect = read_calib(path_calib, 5)
                trans_image = P2.dot(R0_rect)
                image_to_velo = np.array([[0, -1, grid_map_data_origin_idx[0]], [-1, 0, grid_map_data_origin_idx[1]], [0, 0, 1]])
                image_name_hits = example+'_detections_cartesian.png'
                image_path_hits = os.path.join(image_dir,image_name_hits)
                image_name_obs = example+'_observations_cartesian.png'
                image_path_obs = os.path.join(image_dir,image_name_obs)
                image_name_int = example+'_intensity_cartesian.png'
                image_path_int = os.path.join(image_dir,image_name_int)
                image_name_zmin = example+'_z_min_detections_cartesian.png'
                image_path_zmin = os.path.join(image_dir,image_name_zmin)
                image_name_zmax = example+'_z_max_detections_cartesian.png'
                image_path_zmax = os.path.join(image_dir,image_name_zmax)
                image_name_prob = example+'_decay_rate_cartesian.png'
                image_path_prob = os.path.join(image_dir,image_name_prob)
                image_name_rgb = example+'_rgb_cartesian.png'
                image_path_rgb = os.path.join(image_dir,image_name_rgb)
                image_name_ground = example + '_ground_surface_cartesian.png'
                image_path_ground = os.path.join(image_dir, image_name_ground)
                image_name_occlusion = example + '_z_max_occlusions_cartesian.png'
                image_path_occlusion = os.path.join(image_dir, image_name_occlusion)
                image_hits = cv2.imread(image_path_hits, 0)
                image_obs = cv2.imread(image_path_obs, 0)
                image_int = cv2.imread(image_path_int, 0)
                image_zmin = cv2.imread(image_path_zmin, 0)
                image_zmax = cv2.imread(image_path_zmax, 0)
                image_prob = cv2.imread(image_path_prob, 0)
                image_rgb = cv2.imread(image_path_rgb)
                image_occlusion = cv2.imread(image_path_occlusion, 0)
                image_mask = cv2.imread(occ_mask_path, 0)

                ## todo: input features as argument
                #image_stacked = np.stack([image_hits, image_obs, image_int, image_zmin, image_zmax], axis=-1)
                image_stacked = np.stack([image_hits, image_occlusion, image_obs, image_int, image_zmin, image_zmax], axis=-1)
                #image_stacked = np.stack([image_prob, image_int, image_zmin, image_zmax,
                #                          image_rgb[:,:,0], image_rgb[:,:,1], image_rgb[:,:,2]], axis=-1)
                image_ground = cv2.imread(image_path_ground,0)
                image_np_expanded = np.expand_dims(image_stacked, axis=0)
                mask_np_expanded = np.expand_dims(image_mask, axis=0)
                mask_np_expanded = np.expand_dims(mask_np_expanded, axis=3)
                start_time = time.time()
                (boxes, boxes_3d, scores, classes, num) = sess.run(
                    [detection_boxes, detection_boxes_3d ,detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded,
                               mask_tensor: mask_np_expanded})
                print('Inference time:',time.time()-start_time)
                boxes_3d_np = np.squeeze(boxes_3d)
                boxes_np = np.squeeze(boxes)
                scores_np = np.squeeze(scores)
                classes_np = np.squeeze(classes)
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_rgb,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    boxes_3d=np.squeeze(boxes_3d),
                    use_normalized_coordinates=True,
                    line_thickness=3)
                test_image_name = 'image'+ str(idx) + '.png'
                test_image_path = os.path.join(output_path, test_image_name)
                cv2.imwrite(test_image_path, image_rgb)
                file_output=open(output_label,'w')
                for i in range(scores_np.shape[0]):
                    if scores_np[i] > .3:
                        object_class=category_index[int(classes_np[i])]['name']
                        box = tuple(boxes_np[i])
                        y_min = box[0] * image_stacked.shape[0]
                        x_min = box[1] * image_stacked.shape[1]
                        y_max = box[2] * image_stacked.shape[0]
                        x_max = box[3] * image_stacked.shape[1]
                        box_rot = tuple(boxes_3d_np[i])
                        x_c = box_rot[0] * image_stacked.shape[1]
                        y_c = box_rot[1] * image_stacked.shape[0]
                        w_s = box_rot[2]
                        h_s = box_rot[3]
                        sin_angle = box_rot[4]
                        cos_angle = box_rot[5]
                        angle_rad = math.atan2(sin_angle, cos_angle) / 2
                        vec_h_x = h_s * math.cos(angle_rad) / 2.0
                        vec_h_y = h_s * math.sin(angle_rad) / 2.0
                        vec_w_x = - w_s * math.sin(angle_rad) / 2.0
                        vec_w_y = w_s * math.cos(angle_rad) / 2.0
                        x1 = (x_c - vec_w_x - vec_h_x) * image_stacked.shape[1]
                        x2 = (x_c - vec_w_x + vec_h_x) * image_stacked.shape[1]
                        x3 = (x_c + vec_w_x + vec_h_x) * image_stacked.shape[1]
                        y1 = (y_c - vec_w_y - vec_h_y) * image_stacked.shape[0]
                        y2 = (y_c - vec_w_y + vec_h_y) * image_stacked.shape[0]
                        y3 = (y_c + vec_w_y + vec_h_y) * image_stacked.shape[0]
                        l = math.sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1))
                        w = math.sqrt((x3 - x2) * (x3 - x2) + (y3 - y2) * (y3 - y2))
                        image_ground_box = image_ground[int(round(y_min)):int(round(y_max)), int(round(x_min)):int(round(x_max))]
                        mean_ground = image_ground_box.mean()
                        image_height_max_box = image_zmax[int(round(y_min)):int(round(y_max)), int(round(x_min)):int(round(x_max))]
                        height_max = image_height_max_box.max()
                        height_max_m = 2.6 * height_max / 255 - 2.2
                        mean_ground_m = 2.6 * mean_ground / 255 - 2.2

                        object_length_rot = l * grid_map_data_resolution
                        object_width_rot = w * grid_map_data_resolution
                        object_height = height_max_m - mean_ground_m
                        if object_class=='Car':
                            if object_height < 1.3 or object_height > 1.9:
                                object_height = 1.56
                        object_t_rot = np.array([x_c * grid_map_data_resolution, y_c * grid_map_data_resolution])
                        object_t_velo_rot = image_to_velo.dot(np.append(object_t_rot,1))
                        object_t_velo_rot[2] = mean_ground_m
                        object_t_cam_rot = velo_to_cam.dot(np.append(object_t_velo_rot, 1))
                        object_corners_rot = np.array([[object_length_rot/2, object_length_rot/2, -object_length_rot/2, -object_length_rot/2,
                                                        object_length_rot/2, object_length_rot/2, -object_length_rot/2,-object_length_rot/2],
                                                   [0, 0, 0, 0,
                                                    -object_height, -object_height, -object_height,-object_height],
                                                   [object_width_rot/2, -object_width_rot/2, -object_width_rot/2, object_width_rot/2,
                                                    object_width_rot/2, -object_width_rot/2, -object_width_rot/2, object_width_rot/2],
                                                   [1, 1, 1, 1, 1, 1, 1, 1]])
                        corners_to_cam = np.array([[math.cos(angle_rad), 0, math.sin(angle_rad), object_t_cam_rot[0]],
                                                   [0, 1, 0, object_t_cam_rot[1]] ,[-math.sin(angle_rad), 0 , math.cos(angle_rad), object_t_cam_rot[2]],
                                                   [0, 0, 0, 1]])
                        object_corners_cam = corners_to_cam.dot(object_corners_rot)
                        object_corners_image = trans_image.dot(object_corners_cam)
                        object_corners_image_2d = np.array([object_corners_image[0]/object_corners_image[2],
                                                          object_corners_image[1]/object_corners_image[2]])
                        write_labels(file_output, object_class, object_corners_image_2d, object_height, object_width_rot,
                                     object_length_rot, object_t_cam_rot, angle_rad, scores_np[i])


def main(_):
    data_dir = FLAGS.data
    data_ground_dir = FLAGS.data_gr
    data_mask = FLAGS.data_mask

    with open('/mrtstorage/datasets/kitti/object_detection/split_at_5150/train_set_split_at_5150.txt') as f:
        examples_train = f.read().splitlines()

    with open('/mrtstorage/datasets/kitti/object_detection/split_at_5150/val_set_split_at_5150.txt') as f:
        examples_val= f.read().splitlines()

    label_map_path=FLAGS.label_map

    label_dir=FLAGS.labels
    calib_dir=FLAGS.calib

    graph_dir=FLAGS.graph

    train_output_path = os.path.join(FLAGS.output, 'train')
    val_output_path = os.path.join(FLAGS.output, 'val')

    val_path = val_output_path
    os.mkdir(val_path)
    os.mkdir(os.path.join(val_path, 'label'))
    os.mkdir(os.path.join(val_path, 'label_gt'))


    create_kitti_labels(val_output_path, label_map_path, label_dir, calib_dir, data_dir, data_ground_dir, data_mask, graph_dir, examples_val)


if __name__ == '__main__':
    tf.app.run()
