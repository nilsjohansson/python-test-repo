from gluoncv import model_zoo, data, utils
from mxnet.ndarray.ndarray import NDArray
from matplotlib import pyplot as plt
import numpy

def run(imageArray):
    print('SSD')
    # Need a way to keep this model locally, it now downloads every run
    
    net = model_zoo.get_model('ssd_512_resnet50_v1_voc', pretrained=True)

    # im_fname = utils.download('https://github.com/dmlc/web-data/blob/master/gluoncv/detection/street_small.jpg?raw=true',
    #                          path='street_small.jpg')

    x, img = data.transforms.presets.ssd.transform_test(imageArray, short=512)
    print('Shape of pre-processed image:', x.shape)

    class_IDs, scores, bounding_boxes = net(x)

    return class_IDs, scores, bounding_boxes, net.classes, img