import argparse
import os
import linecache
import math
import numpy as np

import sys
sys.path.remove('/opt/mrtsoftware/release/lib/python2.7/dist-packages')
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
print(sys.path)

import cv2

parser = argparse.ArgumentParser(description='Paths to directories')
parser.add_argument('--labels', '-l', required=True)
parser.add_argument('--calib', '-c', required=True)
parser.add_argument('--images', '-i', required=True)
parser.add_argument('--output', '-o', required=True)
args = parser.parse_args()

class Label:
    def __init__(self,
                 type,
                 truncation,
                 occlusion,
                 alpha,
                 x1,
                 y1,
                 x2,
                 y2,
                 h,
                 w,
                 l,
                 tx,
                 ty,
                 tz,
                 ry,
                 score):
        self.type = type
        self.truncation = truncation
        self.occlusion = occlusion
        self.alpha = alpha
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.h = h
        self.w = w
        self.l = l
        self.tx = tx
        self.ty = ty
        self.tz = tz
        self.ry = ry
        self.score = score

def read_labels(path_label):
    label_file = open(path_label)
    labels = []
    for idx, line in enumerate(label_file):
        str_label = line.split(' ')
        print(str_label[0], str_label[8],str_label[9],str_label[10],str_label[11])
        label_struct = Label(str_label[0], float(str_label[1]), float(str_label[2]), float(str_label[3]),
                             float(str_label[4]), float(str_label[5]), float(str_label[6]), float(str_label[7]),
                             float(str_label[8]), float(str_label[9]), float(str_label[10]), float(str_label[11]),
                             float(str_label[12]), float(str_label[13]), float(str_label[14]), float(str_label[15]))
        labels.append(label_struct)
    return labels

def read_calib(path_calib, idx):
    str_trans = linecache.getline(path_calib, idx).split(' ')[1:]
    list_trans = [float(i) for i in str_trans]
    if idx == 5:
        trans = np.array([[list_trans[0], list_trans[1], list_trans[2], 0],
                          [list_trans[3], list_trans[4], list_trans[5], 0],
                          [list_trans[6], list_trans[7], list_trans[8], 0],
                          [0, 0, 0, 1]])
    else:
        trans = np.array([[list_trans[0], list_trans[1], list_trans[2], list_trans[3]],
                          [list_trans[4], list_trans[5], list_trans[6], list_trans[7]],
                          [list_trans[8], list_trans[9], list_trans[10], list_trans[11]],
                          [0, 0, 0, 1]])
    return trans

num_test_images = 7518
draw_poly = True
draw_rect = False
draw_circles = False

test_list = ["%06d" % idx for idx in range(0, num_test_images)]
for idx in test_list:
    output_path = os.path.join(args.output, idx + '.png')
    label_path = os.path.join(args.labels, idx + '.txt')
    labels = read_labels(label_path)
    calib_path = os.path.join(args.calib, idx + '.txt')
    velo_to_cam = read_calib(calib_path, 6)
    P2 = read_calib(calib_path, 3)
    R0_rect = read_calib(calib_path, 5)
    trans_image = P2.dot(R0_rect)
    image_path = os.path.join(args.images, idx + '.png')
    image = cv2.imread(image_path)
    for obj in labels:
        corners_obj = np.array([[obj.l / 2, obj.l / 2, - obj.l / 2, - obj.l / 2,
                                 obj.l / 2, obj.l / 2, - obj.l / 2, - obj.l / 2],
                                [0, 0, 0, 0,
                                 - obj.h, - obj.h, - obj.h, - obj.h],
                                [obj.w / 2, - obj.w / 2, - obj.w / 2, obj.w / 2,
                                 obj.w / 2, - obj.w / 2, -obj.w / 2, obj.w / 2],
                                [1, 1, 1, 1, 1, 1, 1, 1]])
        trans_cam = np.array([[math.cos(obj.ry), 0, math.sin(obj.ry), obj.tx],
                              [0, 1, 0, obj.ty],
                              [- math.sin(obj.ry), 0, math.cos(obj.ry), obj.tz],
                              [0, 0, 0, 1]])
        corners_cam = trans_cam.dot(corners_obj)
        corners_image = trans_image.dot(corners_cam)
        corners_image_2d = np.array([corners_image[0] / corners_image[2],
                                     corners_image[1] / corners_image[2]])
        if draw_circles:
            for i in range(len(corners_image_2d[0])):
                cv2.circle(image, (int(corners_image_2d[0][i]), int(corners_image_2d[1][i])), 3, (0, 0, 255), -1)
        if draw_poly:
            cv2.line(image, (int(corners_image_2d[0][0]), int(corners_image_2d[1][0])),
                     (int(corners_image_2d[0][1]), int(corners_image_2d[1][1])), (0, 0, 255), 2)
            cv2.line(image, (int(corners_image_2d[0][1]), int(corners_image_2d[1][1])),
                     (int(corners_image_2d[0][2]), int(corners_image_2d[1][2])), (0, 0, 255), 2)
            cv2.line(image, (int(corners_image_2d[0][2]), int(corners_image_2d[1][2])),
                     (int(corners_image_2d[0][3]), int(corners_image_2d[1][3])), (0, 0, 255), 2)
            cv2.line(image, (int(corners_image_2d[0][3]), int(corners_image_2d[1][3])),
                     (int(corners_image_2d[0][0]), int(corners_image_2d[1][0])), (0, 0, 255), 2)
            cv2.line(image, (int(corners_image_2d[0][7]), int(corners_image_2d[1][7])),
                     (int(corners_image_2d[0][6]), int(corners_image_2d[1][6])), (0, 0, 255), 2)
            cv2.line(image, (int(corners_image_2d[0][6]), int(corners_image_2d[1][6])),
                     (int(corners_image_2d[0][5]), int(corners_image_2d[1][5])), (0, 0, 255), 2)
            cv2.line(image, (int(corners_image_2d[0][5]), int(corners_image_2d[1][5])),
                     (int(corners_image_2d[0][4]), int(corners_image_2d[1][4])), (0, 0, 255), 2)
            cv2.line(image, (int(corners_image_2d[0][4]), int(corners_image_2d[1][4])),
                     (int(corners_image_2d[0][7]), int(corners_image_2d[1][7])), (0, 0, 255), 2)
            cv2.line(image, (int(corners_image_2d[0][0]), int(corners_image_2d[1][0])),
                     (int(corners_image_2d[0][4]), int(corners_image_2d[1][4])), (0, 0, 255), 2)
            cv2.line(image, (int(corners_image_2d[0][1]), int(corners_image_2d[1][1])),
                     (int(corners_image_2d[0][5]), int(corners_image_2d[1][5])), (0, 0, 255), 2)
            cv2.line(image, (int(corners_image_2d[0][2]), int(corners_image_2d[1][2])),
                     (int(corners_image_2d[0][6]), int(corners_image_2d[1][6])), (0, 0, 255), 2)
            cv2.line(image, (int(corners_image_2d[0][3]), int(corners_image_2d[1][3])),
                     (int(corners_image_2d[0][7]), int(corners_image_2d[1][7])), (0, 0, 255), 2)
        if draw_rect:
            cv2.rectangle(image, (int(obj.x1), int(obj.y1)), (int(obj.x2), int(obj.y2)), (255, 255, 0), 2)
    cv2.imwrite(output_path, image)
