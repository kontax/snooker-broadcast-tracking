import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pylab
import _init_paths
import numpy as np
from videos.youtube_video import YoutubeVideo
from model.table_setup import TableSetup
from model.detection import SnookerDetector
from os import path
from config import cfg
from fast_rcnn.config import cfg as caffe_cfg


def transpose_image():
    print c
    w = cfg.snooker.width * cfg.multiplier
    h = cfg.snooker.height * cfg.multiplier
    newimg = np.zeros(shape=(h, w, 3))
    print "{0}".format(counter)
    # Transpose & clean
    # Turn into JSON
    # Send to RabbitMQ
    for x in xrange(len(newimg[0])):
        for y in xrange(len(newimg)):
            hom = c.dot(np.array([[x], [y], [1]]))
            newx = int(hom[0] / hom[2])
            newy = int(hom[1] / hom[2])
            newimg[y][x] = f[newy][newx]
    p = pylab.figure()
    for n, x in enumerate((f, newimg)):
        p.add_subplot(1, 2, n + 1)
        pylab.axis('off')
        pylab.tight_layout()
        pylab.imshow(x, cmap='Greys_r')
    pylab.show()


def transpose_snooker_objects():
    m = cfg.multiplier
    w = cfg.snooker.width * m
    h = cfg.snooker.height * m
    new_img = np.zeros(shape=(h, w, 3))
    fix, ax = plt.subplots(figsize=(12, 12))
    #ax.imshow(new_img, aspect='equal')

    for ind, img in enumerate((f, new_img)):
        fix.add_subplot(1, 2, ind + 1)
        plt.axis('off')
        plt.tight_layout()
        plt.imshow(img, aspect='equal')

    for ball in c.balls:
        print "{0}: ({1},{2}) | ({3},{4})".format(
            ball.colour, ball.x1, ball.y1, ball.x2, ball.y2)
        ax.add_patch(plt.Rectangle(
            (ball.x1, ball.y1),
            ball.x2 - ball.x1,
            ball.y2 - ball.y1,
            fill=False,
            edgecolor=ball.colour,
            linewidth=1))
    # plt.axis('off')
    for pocket in c.pockets:
        ax.add_patch(plt.Rectangle(
            (pocket.x1, pocket.y1),
            pocket.x2 - pocket.x1,
            pocket.y2 - pocket.y1,
            fill=False,
            edgecolor='orange',
            linewidth=1))

    plt.tight_layout()
    plt.draw()
    plt.show()


if __name__ == '__main__':

    # Use RPN for proposals
    caffe_cfg.TEST.HAS_RPN = True

    data_path = path.join('py-faster-rcnn', 'data', 'snooker')
    model_path = path.join('py-faster-rcnn', 'models', 'snooker_net')

    video = YoutubeVideo(url="https://www.youtube.com/watch?v=RIOi3YKtBcY")
    #video = YoutubeVideo(url="https://www.youtube.com/watch?v=irpfzXXPrX8")
    model = path.join(data_path, 'snooker.caffemodel')
    prototxt = path.join(model_path, 'snooker', 'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
    table_model = path.join(data_path, 'snooker_table.caffemodel')
    table_prototxt = path.join(model_path, 'snooker_table', 'snooker_table.pt')
    detector = SnookerDetector(config=cfg,
                               video=video,
                               model=model,
                               prototxt=prototxt,
                               table_model=table_model,
                               table_prototxt=table_prototxt)
    cleaner = TableSetup(cfg)

    detections = detector.detect()
    for image, detection in detections:
        table = cleaner.clean_predictions(detection)

