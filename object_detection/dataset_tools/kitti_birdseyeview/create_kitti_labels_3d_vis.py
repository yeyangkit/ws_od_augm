import os
import cv2
import time
import tensorflow as tf
import numpy as np

from utils import visualization_utils as vis_util
from utils import label_map_util
from utils import dataset_util

flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to kitti dataset used for inference.')
flags.DEFINE_string('vis_dir', '', 'Root directory to kitti dataset used for visualization.')
flags.DEFINE_string('graph_path', '', 'Path to frozen inference graph.')
flags.DEFINE_string('output_dir', '', 'Path to directory to output labels.')
flags.DEFINE_string('label_map_path', 'data/kitti_object_label_map.pbtxt',
                    'Path to label map proto')
FLAGS = flags.FLAGS


def create_kitti_labels(output_path, label_map_path, image_dir, vis_dir, graph_dir, examples):
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
                image_name=example+'_euclidean_merge.png'
                image_path=os.path.join(image_dir,image_name)
                image_name_prob =  '%06d' % int(example) + '_probability.png'
                image_path_prob = os.path.join(vis_dir, image_name_prob)
                image_name_obs = '%06d' % int(example) + '_observations.png'
                image_path_obs = os.path.join(vis_dir, image_name_obs)
                print('Image path', image_path)
                image = cv2.imread(image_path)
                image = image[:, :, [2, 1, 0]]
                image_prob = cv2.imread(image_path_prob,0)
                print(image_path_prob)
                image_obs = cv2.imread(image_path_obs, 0)
                image_prob = cv2.bitwise_not(image_prob)
                image_obs = cv2.bitwise_not(image_obs)
                image_vis = np.stack([image_prob, image_obs, image_prob], axis=-1)
                image_np_expanded = np.expand_dims(image, axis=0)
                start_time = time.time()
                (boxes, boxes_rot, scores, classes, num) = sess.run(
                    [detection_boxes, detection_boxes_rot ,detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                print(image_vis.shape)
                crop_size = 200
                image_vis = image_vis[crop_size:image_vis.shape[0],
                            int(crop_size/2):int(image_vis.shape[0]-crop_size/2)]
                print('Inference time:',time.time()-start_time)
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
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_vis,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    boxes_rot=np.squeeze(boxes_rot),
                    use_normalized_coordinates=True,
                    line_thickness=5)
                test_image_name = 'image'+ str(idx) + '.png'
                test_image_path = os.path.join(output_path, test_image_name)
                print(test_image_path)
                image_vis = image_vis[:, :, [2, 1, 0]]
                cv2.imwrite(test_image_path, image_vis)


def main(_):
    data_dir = FLAGS.data_dir
    vis_dir = FLAGS.vis_dir
    image_dir = os.path.join(data_dir, 'grid_maps')
    examples_path = os.path.join(data_dir, 'trainval.txt')
    examples_list = dataset_util.read_examples_list(examples_path)
    num_examples = len(examples_list)
    num_train = int(0.7 * num_examples)
    train_examples = examples_list[:num_train]
    val_examples = examples_list[num_train:]

    label_map_path=FLAGS.label_map_path

    graph_dir=FLAGS.graph_path

    train_output_path = os.path.join(FLAGS.output_dir, 'train')
    val_output_path = os.path.join(FLAGS.output_dir, 'val')

    create_kitti_labels(val_output_path, label_map_path, image_dir, vis_dir, graph_dir, val_examples)


if __name__ == '__main__':
  tf.app.run()