from sys import argv
from src.openCV.Detector import *

modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz"
# modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet101_v1_640x640_coco17_tpu-8.tar.gz"
# modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d4_coco17_tpu-32.tar.gz"

imagePath = "assets/federer.jpg"
videoPath = 0 # "videoPath"
threshold = 0.5

classFile = "src/config/coco.names"
detector = Detector()
#detector.readClasses(classFile)
#detector.downloadModel(modelURL)
#detector.loadModel()
# detector.predictImage(imagePath, threshold)
detector.testCam()

