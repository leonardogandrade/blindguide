from sys import argv
from src.openCV.Detector import *

modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz"

classFile = "src/config/coco.names"
detector = Detector()
detector.readClasses(classFile)
detector.downloadModel(modelURL)
