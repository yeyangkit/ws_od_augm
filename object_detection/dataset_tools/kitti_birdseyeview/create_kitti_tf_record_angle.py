import hashlib
import io
import logging
import os
import math

import PIL.Image
import tensorflow as tf

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to raw kitti dataset.')
flags.DEFINE_string('output_dir', '', 'Path to directory to output TFRecords.')
flags.DEFINE_string('label_map_path', 'data/kitti_object_label_map.pbtxt',
                    'Path to label map proto')
FLAGS = flags.FLAGS

def convert_string_to_dict(string):
    dict_b={'object':{}}
    b=string.splitlines()
    for str_el in b:
        if str_el:
            str_name,str_value=str_el.split(':')
            if str_name=='filename' or str_name=='width' or str_name =='height':
                dict_b[str_name] = str_value
            elif str_name not in dict_b['object']:
                dict_b['object'][str_name]=[str_value]
            else:
                dict_b['object'][str_name].append(str_value)
    return dict_b

def dict_to_tf_example(data,
                       label_map_dict,
                       image_subdirectory):

  img_path = os.path.join(image_subdirectory, data['filename'])
  print(img_path)
  with tf.gfile.GFile(img_path, 'rb') as fid:
    encoded_png = fid.read()
  encoded_png_io = io.BytesIO(encoded_png)
  image = PIL.Image.open(encoded_png_io)
  #print(image.format)
  key = hashlib.sha256(encoded_png).hexdigest()

  #width = int(data['size']['width'])
  #height = int(data['size']['height'])
  width = int(data['width'])
  height = int(data['height'])
  #print('widht', width, 'height', height)

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
  #truncated = []
  #difficult_obj = []
  #for obj in data['object']:
  for num_label in range(len(data['object']['type'])):
    #difficult = bool(int(obj['difficult']))
    #if ignore_difficult_instances and difficult:
    #  continue

    #difficult_obj.append(int(difficult))

    xmin.append(float(data['object']['x_min'][num_label]) / width)
    #print('xmin',xmin)
    ymin.append(float(data['object']['y_min'][num_label]) / height)
    #print('ymin',ymin)
    xmax.append(float(data['object']['x_max'][num_label]) / width)
    #print('xmax',xmax)
    ymax.append(float(data['object']['y_max'][num_label]) / height)
    #print('ymax',ymax)
    x_c.append((float(data['object']['x_min'][num_label]) + float(data['object']['x_max'][num_label])) / (2 * width))
    #print('x_c:', x_c)
    y_c.append((float(data['object']['y_min'][num_label]) + float(data['object']['y_max'][num_label])) / (2 * height))
    #print('y_c:', y_c)
    angle_rad = float(data['object']['ry'][num_label])
    if  angle_rad * 180 / 3.141 > 90:
        print('ANGLE_RAD:',angle_rad)
    angle.append(angle_rad * 180 / 3.141)
    #print('angle:', angle)
    sin_angle.append(math.sin(2 * angle_rad))
    cos_angle.append(math.cos(2 * angle_rad))
    vec_s_x = math.cos(angle_rad)
    vec_s_y = math.sin(angle_rad)
    w_p = float(data['object']['w_p'][num_label])
    w_p_s = w_p * math.sqrt(vec_s_x * vec_s_x / (height * height) + vec_s_y * vec_s_y / (width * width))
    w.append(w_p_s)
    #print('w:', w)
    l_p = float(data['object']['l_p'][num_label])
    l_p_s = l_p * math.sqrt(vec_s_x * vec_s_x / (width * width) + vec_s_y * vec_s_y / (height * height))
    h.append(l_p_s)
    #print('h:', h)
    class_name = data['object']['type'][num_label]
    #print(class_name)
    classes_text.append(class_name.encode('utf8'))
    #print(classes_text)
    classes.append(label_map_dict[class_name])
    #print('class number',classes)
    #truncated.append(int(data['object']['truncation'][num_label]))

    x_min=float(data['object']['x_min'][num_label]) / width
    y_min=float(data['object']['y_min'][num_label]) / height
    x_max=float(data['object']['x_max'][num_label]) / width
    y_max=float(data['object']['y_max'][num_label]) / height

    if x_min<0 or x_min>1 or x_max<0 or x_max>1 or y_min<0 or y_min>1 or y_max<0 or y_max>1:
        print(data['filename'])
        print("xmin:", x_min)
        print("xmax:", x_max)
        print("ymin:", y_min)
        print("ymax:", y_max)

  example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(
          data['filename'].encode('utf8')),
      'image/source_id': dataset_util.bytes_feature(
          data['filename'].encode('utf8')),
      'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_png),
      'image/format': dataset_util.bytes_feature('png'.encode('utf8')),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
      'image/object/bboxrot/x_c': dataset_util.float_list_feature(x_c),
      'image/object/bboxrot/y_c': dataset_util.float_list_feature(y_c),
      'image/object/bboxrot/w': dataset_util.float_list_feature(w),
      'image/object/bboxrot/h': dataset_util.float_list_feature(h),
      'image/object/bboxrot/angle': dataset_util.float_list_feature(angle),
      'image/object/bboxrot/sin_angle': dataset_util.float_list_feature(sin_angle),
      'image/object/bboxrot/cos_angle': dataset_util.float_list_feature(cos_angle),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
  }))
  return example


def create_tf_record(output_filename,
                     label_map_dict,
                     annotations_dir,
                     image_dir,
                     examples):

  writer = tf.python_io.TFRecordWriter(output_filename)
  print(output_filename)
  for idx, example in enumerate(examples):
    if idx % 100 == 0:
      logging.info('On image %d of %d', idx, len(examples))
    path = os.path.join(annotations_dir, example + '.txt')

    if not os.path.exists(path):
      logging.warning('Could not find %s, ignoring example.', path)
      continue
    with tf.gfile.GFile(path, 'r') as fid:
        str_label = fid.read()
    data = convert_string_to_dict(str_label)

    tf_example=dict_to_tf_example(data, label_map_dict, image_dir)
    writer.write(tf_example.SerializeToString())

  writer.close()

def main(_):
  data_dir = FLAGS.data_dir
  label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)

  logging.info('Reading from Kitti dataset.')
  image_dir = os.path.join(data_dir, 'grid_maps')
  label_dir = os.path.join(data_dir, 'labels')
  examples_path = os.path.join(data_dir, 'trainval.txt')
  examples_list = dataset_util.read_examples_list(examples_path)

  num_examples = len(examples_list)
  num_train = int(0.7 * num_examples)
  train_examples = examples_list[:num_train]
  val_examples = examples_list[num_train:]

  logging.info('%d training and %d validation examples.',
               len(train_examples), len(val_examples))
  print('%d training and %d validation examples.',
               len(train_examples), len(val_examples))

  train_output_path = os.path.join(FLAGS.output_dir, 'kitti_train.record')
  val_output_path = os.path.join(FLAGS.output_dir, 'kitti_val.record')

  create_tf_record(train_output_path, label_map_dict, label_dir,
                   image_dir, train_examples)
  create_tf_record(val_output_path, label_map_dict, label_dir,
                   image_dir, val_examples)

if __name__ == '__main__':
  tf.app.run()