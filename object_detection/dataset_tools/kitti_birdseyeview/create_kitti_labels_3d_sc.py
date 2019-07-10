import os
import linecache
import cv2
import math
import time
import tensorflow as tf
import numpy as np
from shutil import copyfile

from utils import visualization_utils as vis_util
from utils import label_map_util
from utils import dataset_util

flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to raw kitti dataset.')
flags.DEFINE_string('label_dir', '', 'Directory to kitti labels.')
flags.DEFINE_string('calib_dir', '', 'Directory to kitti calibrations.')
flags.DEFINE_string('graph_path', '', 'Path to frozen inference graph.')
flags.DEFINE_string('output_dir', '', 'Path to directory to output labels.')
flags.DEFINE_string('label_map_path', 'data/kitti_object_label_map.pbtxt',
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

def create_kitti_labels(output_path, label_map_path, label_dir, calib_dir, image_dir, graph_dir, examples):
    grid_map_data_resolution=0.1
    grid_map_data_origin_idx=np.array([79, 35])

    label_map = label_map_util.load_labelmap(label_map_path)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=8, use_display_name=True)
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
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            detection_boxes_rot = detection_graph.get_tensor_by_name('detection_boxes_rot:0')
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
                print(velo_to_cam)
                P2 = read_calib(path_calib, 3)
                R0_rect = read_calib(path_calib, 5)
                trans_image=P2.dot(R0_rect)
                image_to_velo = np.array([[0, -1, grid_map_data_origin_idx[0]], [-1, 0, grid_map_data_origin_idx[1]], [0, 0, 1]])
                image_name=example+'_euclidean_merge.png'
                image_path=os.path.join(image_dir,image_name)
                image_name_ground = example + '_euclidean_height_ground.png'
                image_path_ground = os.path.join(image_dir, image_name_ground)
                image_name_height_max = example + '_euclidean_height_max.png'
                image_path_height_max = os.path.join(image_dir, image_name_height_max)
                image_name_height_min = example + '_euclidean_height_min.png'
                image_path_height_min = os.path.join(image_dir, image_name_height_min)
                image_name_height_norm_max = example + '_euclidean_height_norm_max.png'
                image_path_height_norm_max = os.path.join(image_dir, image_name_height_norm_max)
                print('Image path', image_path)
                image = cv2.imread(image_path)
                image = image[:, :, [2, 1, 0]]
                image_ground = cv2.imread(image_path_ground,0)
                image_height_max = cv2.imread(image_path_height_max, 0)
                image_height_min = cv2.imread(image_path_height_min, 0)
                image_height_norm_max = cv2.imread(image_path_height_norm_max, 0)
                image_np_expanded = np.expand_dims(image, axis=0)
                start_time = time.time()
                (boxes, boxes_rot, scores, classes, num) = sess.run(
                    [detection_boxes, detection_boxes_rot ,detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                print('Inference time:',time.time()-start_time)
                boxes_rot_np = np.squeeze(boxes_rot)
                boxes_np = np.squeeze(boxes)
                scores_np = np.squeeze(scores)
                classes_np = np.squeeze(classes)
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    boxes_rot=np.squeeze(boxes_rot),
                    use_normalized_coordinates=True,
                    line_thickness=3)
                test_image_name = 'image'+ str(idx) + '.png'
                test_image_path = os.path.join(output_path, test_image_name)
                cv2.imwrite(test_image_path, image)
                file_output=open(output_label,'w')
                for i in range(scores_np.shape[0]):
                    if scores_np[i] > .75:
                        object_class=category_index[int(classes_np[i])]['name']
                        box = tuple(boxes_np[i])
                        y_min = box[0] * image.shape[0]
                        x_min = box[1] * image.shape[1]
                        y_max = box[2] * image.shape[0]
                        x_max = box[3] * image.shape[1]
                        box_rot = tuple(boxes_rot_np[i])
                        print('box_rot:', box_rot)
                        x_c = box_rot[0] * image.shape[1]
                        y_c = box_rot[1] * image.shape[0]
                        w_s = box_rot[2]
                        h_s = box_rot[3]
                        angle = box_rot[4]
                        angle_rad = angle * 3.141 / 180
                        vec_h_x = h_s * math.cos(angle_rad) / 2.0
                        vec_h_y = h_s * math.sin(angle_rad) / 2.0
                        vec_w_x = - w_s * math.sin(angle_rad) / 2.0
                        vec_w_y = w_s * math.cos(angle_rad) / 2.0
                        x1 = (x_c - vec_w_x - vec_h_x) * image.shape[1]
                        x2 = (x_c - vec_w_x + vec_h_x) * image.shape[1]
                        x3 = (x_c + vec_w_x + vec_h_x) * image.shape[1]
                        y1 = (y_c - vec_w_y - vec_h_y) * image.shape[0]
                        y2 = (y_c - vec_w_y + vec_h_y) * image.shape[0]
                        y3 = (y_c + vec_w_y + vec_h_y) * image.shape[0]
                        l = math.sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1))
                        w = math.sqrt((x3 - x2) * (x3 - x2) + (y3 - y2) * (y3 - y2))
                        image_ground_box = image_ground[int(round(y_min)):int(round(y_max)), int(round(x_min)):int(round(x_max))]
                        if (image_ground_box[image_ground_box < 255]).size==0:
                            continue
                        mean_ground = image_ground_box[image_ground_box < 255].mean()
                        image_height_max_box = image_height_max[int(round(y_min)):int(round(y_max)), int(round(x_min)):int(round(x_max))]
                        if (image_height_max_box[image_height_max_box < 255]).size==0:
                            continue
                        height_max = image_height_max_box[image_height_max_box < 255].min()
                        image_height_min_box = image_height_min[int(round(y_min)):int(round(y_max)), int(round(x_min)):int(round(x_max))]
                        if (image_height_min_box[image_height_min_box < 255]).size==0:
                            continue
                        height_min = image_height_min_box[image_height_min_box < 255].max()
                        image_height_norm_max_box = image_height_norm_max[int(round(y_min)):int(round(y_max)), int(round(x_min)):int(round(x_max))]
                        if (image_height_norm_max_box[image_height_min_box < 255]).size==0:
                            continue
                        height_norm_max = image_height_norm_max_box[image_height_min_box < 255].min()
                        object_length = (y_max-y_min)*grid_map_data_resolution
                        object_width = (x_max-x_min)*grid_map_data_resolution
                        object_length_rot = l * grid_map_data_resolution
                        object_width_rot = w * grid_map_data_resolution
                        object_height = -3 * height_norm_max / 255 + 3
                        if object_class=='Car':
                            if object_height < 1.3 or object_height > 1.9:
                                object_height = 1.56
                        #    if object_width < 1.4 or object_width > 1.8:
                        #        object_width = 1.6
                        #    if object_length < 3.2 or object_length > 4.7:
                        #        object_length = 3.9
                        object_t_rot = np.array([x_c * grid_map_data_resolution, y_c * grid_map_data_resolution])
                        object_t_velo_rot=image_to_velo.dot(np.append(object_t_rot,1))
                        object_t_velo_rot[2] = - 2*mean_ground/255 - 0.75
                        object_t_cam_rot = velo_to_cam.dot(np.append(object_t_velo_rot, 1))
                        print('object_t_cam_rot:', object_t_cam_rot)


                        #print('object_height', object_height)
                        #print('object_width_rot', object_width_rot)
                        #print('object_length_rot', object_length_rot)
                        #print('object_t_velo_rot', object_t_velo_rot)
                        #print('object_t_cam_rot', object_t_cam_rot)
                        #print('angle_rad', angle_rad)
                        #print('')

                        object_corners_rot = np.array([[object_length_rot/2, object_length_rot/2, -object_length_rot/2, -object_length_rot/2,
                                                        object_length_rot/2, object_length_rot/2, -object_length_rot/2,-object_length_rot/2],
                                                   [0, 0, 0, 0,
                                                    -object_height, -object_height, -object_height,-object_height],
                                                   [object_width_rot/2, -object_width_rot/2, -object_width_rot/2, object_width_rot/2,
                                                    object_width_rot/2, -object_width_rot/2, -object_width_rot/2, object_width_rot/2],
                                                   [1, 1, 1, 1, 1, 1, 1, 1]])

                        #corners_to_cam = np.array([[1, 0, 0, object_t_cam_rot[0]],[0, 1, 0, object_t_cam_rot[1]]
                        #                              ,[0, 0 , 1, object_t_cam_rot[2]],[0, 0, 0, 1]])
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
    data_dir = FLAGS.data_dir
    image_dir = os.path.join(data_dir, 'grid_maps')
    examples_path = os.path.join(data_dir, 'trainval.txt')
    examples_list = dataset_util.read_examples_list(examples_path)
    num_examples = len(examples_list)
    num_train = int(0.7 * num_examples)
    train_examples = examples_list[:num_train]
    val_examples = examples_list[num_train:]

    label_map_path=FLAGS.label_map_path

    label_dir=FLAGS.label_dir
    calib_dir=FLAGS.calib_dir

    graph_dir=FLAGS.graph_path

    train_output_path = os.path.join(FLAGS.output_dir, 'train')
    val_output_path = os.path.join(FLAGS.output_dir, 'val')

    create_kitti_labels(val_output_path, label_map_path, label_dir, calib_dir, image_dir, graph_dir, val_examples)


if __name__ == '__main__':
  tf.app.run()