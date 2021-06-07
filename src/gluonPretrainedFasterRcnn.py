from gluoncv import model_zoo, data, utils
from matplotlib import pyplot as plt

net = model_zoo.get_model('faster_rcnn_resnet50_v1b_voc', pretrained=True)

# im_fname = utils.download('https://github.com/dmlc/web-data/blob/master/gluoncv/detection/street_small.jpg?raw=true',
#                          path='street_small.jpg')

x, orig_img = data.transforms.presets.rcnn.load_test(r"E:\\Temp\3parkedcars-noWindowInterference.jpg")
print('Shape of pre-processed image:', x.shape)

class_IDs, scores, bounding_boxes = net(x)

ax = utils.viz.plot_bbox(orig_img, bounding_boxes[0], scores[0],
                         class_IDs[0], class_names=net.classes)
plt.show()