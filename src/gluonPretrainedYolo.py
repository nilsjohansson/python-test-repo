from gluoncv import model_zoo, data, utils
from matplotlib import pyplot as plt

def run(fname):
    net = model_zoo.get_model('yolo3_darknet53_voc', pretrained=True)

    x, orig_img = data.transforms.presets.yolo.load_test(fname, short=512)
    print('Shape of pre-processed image:', x.shape)

    class_IDs, scores, bounding_boxes = net(x)

    ax = utils.viz.plot_bbox(orig_img, bounding_boxes[0], scores[0],
                            class_IDs[0], class_names=net.classes)
    plt.show()