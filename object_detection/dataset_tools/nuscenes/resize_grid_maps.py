import glob
import sys
# sys.path.remove('/opt/mrtsoftware/release/lib/python2.7/dist-packages')
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import os

mypath = '/mrtstorage/datasets/nuscenes/grid_map/15cm_100m/v1.0-trainval_meta/samples/LIDAR_TOP'
outpath = '/mrtstorage/projects/grid_map_object_detection/nuscene_tfrecord/range_60m/samples'
i = 1

for file in glob.glob(mypath+"/*cartesian.png"):
  img = cv2.imread(file)
  img = img[133:534, 133:534]
  img = cv2.resize(img, (400, 400), interpolation=cv2.INTER_CUBIC)
  img_name = file.split('/')[-1]
  cv2.imwrite(os.path.join(outpath, img_name), img)
  print('Write', i, 'grid map')
  i += 1
