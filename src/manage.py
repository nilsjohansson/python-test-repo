from gluonPretrainedFasterRcnn import run as runRcnn
from gluonPretrainedSsdTutorial import run as runSsd
from gluonPretrainedYolo import run as runYolo
import mxnet as mx
import os
from matplotlib import pyplot as plt
from gluoncv import utils

script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
rel_path = "../testdata/parkingoutside2-rotated.jpg"
abs_file_path = os.path.join(script_dir, rel_path)

with open(abs_file_path, 'rb') as fp:
    str_image = fp.read()

image = mx.img.imdecode(str_image)

# runRcnn(abs_file_path)
# runYolo(abs_file_path)
class_IDs, scores, bounding_boxes, net_class_names, transformed_image = runSsd(image)

ax = utils.viz.plot_bbox(transformed_image, bounding_boxes[0], scores[0], class_IDs[0], class_names=net_class_names)
plt.show()