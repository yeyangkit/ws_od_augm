import os
import sys
sys.path.remove('/opt/mrtsoftware/release/lib/python2.7/dist-packages')
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
print(sys.path)
import cv2
import time
import tensorflow as tf
import numpy as np

from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import label_map_util
from object_detection.utils import dataset_util

flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to raw kitti dataset.')
flags.DEFINE_string('graph_path', '', 'Path to frozen inference graph.')
flags.DEFINE_string('output_dir', '', 'Path to directory to output.')
flags.DEFINE_string('label_map_path', 'data/kitti_object_label_map.pbtxt',
                    'Path to label map proto')
FLAGS = flags.FLAGS

def create_sequence(output_path, label_map_path, image_dir, graph_dir, examples):

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
            detection_boxes_rot = detection_graph.get_tensor_by_name('detection_boxes_3d:0')
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            for idx, example in enumerate(examples):
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
                image_hits = cv2.imread(image_path_hits,0)
                image_obs = cv2.imread(image_path_obs, 0)
                image_int = cv2.imread(image_path_int, 0)
                image_zmin = cv2.imread(image_path_zmin, 0)
                image_zmax = cv2.imread(image_path_zmax, 0)
                image_prob = cv2.imread(image_path_prob, 0)
                image_rgb = cv2.imread(image_path_rgb)

                inv_image_hits = cv2.bitwise_not(image_hits)
                inv_image_obs = cv2.bitwise_not(image_obs)
                image_vis = np.stack([inv_image_hits, inv_image_obs, inv_image_hits], axis=-1)
                image_occlusion = cv2.imread(image_path_occlusion, 0)

                for x in range(0, image_vis.shape[0]):
                    for y in range(0, image_vis.shape[1]):
                        if image_vis[x, y, 0] < 255:
                            image_vis[x, y, 0] = 0
                            image_vis[x, y, 1] = 0
                            image_vis[x, y, 2] = 0
                        elif image_vis[x, y, 1] < 255:
                            value = 255 - image_vis[x, y, 1]
                            value = value * 0.7
                            value = 255 - value
                            value = 220
                            image_vis[x, y, 0] = value
                            image_vis[x, y, 2] = value
                            image_vis[x, y, 1] = value

                ## todo: input features as argument
                #image_stacked = np.stack([image_hits, image_obs, image_int, image_zmin, image_zmax], axis=-1)
                image = np.stack([image_hits, image_occlusion, image_obs, image_int, image_zmin, image_zmax], axis=-1)
                #image_stacked = np.stack([image_prob, image_int, image_zmin, image_zmax,
                #                          image_rgb[:,:,0], image_rgb[:,:,1], image_rgb[:,:,2]], axis=-1)
                image_ground = cv2.imread(image_path_ground,0)

                image_np_expanded = np.expand_dims(image, axis=0)
                start_time = time.time()
                (boxes, boxes_rot, scores, classes, num) = sess.run(
                    [detection_boxes, detection_boxes_rot, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                print('Inference time:',time.time()-start_time)
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_vis,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    boxes_3d=np.squeeze(boxes_rot),
                    use_normalized_coordinates=True,
                    line_thickness=2,
                    max_boxes_to_draw=100,
                    min_score_thresh=0.5,
                    skip_scores=False,
                    skip_labels=True)
                test_image_name = '%04d' % (idx) + '.png'
                test_image_path = os.path.join(output_path, test_image_name)
                cv2.imwrite(test_image_path, image_vis)

def main(_):
    data_dir = FLAGS.data_dir
    examples_list = []
    for i in range(2761):
        examples_list.append('%06d' % int(i))

    print('Size', len(examples_list))

    label_map_path=FLAGS.label_map_path

    graph_dir=FLAGS.graph_path

    seq_output_path = FLAGS.output_dir

    create_sequence(seq_output_path, label_map_path, data_dir, graph_dir, examples_list)


if __name__ == '__main__':
  tf.app.run()
