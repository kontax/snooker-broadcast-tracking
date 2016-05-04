from os import path

import _init_paths
import matplotlib.pyplot as plt
import numpy as np
import pylab

from config import cfg
from fast_rcnn.config import cfg as caffe_cfg
from model.detection import SnookerDetector
from model.table_setup import TableSetup
from videos.youtube_video import YoutubeVideo


def transpose_snooker_objects():
    m = cfg.multiplier
    w = cfg.snooker.width * m
    h = cfg.snooker.height * m
    new_img = np.full(shape=(h, w, 3), fill_value=(0, 128, 0))
    ax0.imshow(image)
    ax1.imshow(new_img, aspect='equal')

    for ball in table.balls:
        print "{0}: ({1},{2}) | ({3},{4})".format(
            ball.colour, ball.x1, ball.y1, ball.x2, ball.y2)
        ax1.add_patch(plt.Circle(
            (ball.x1, ball.y1),
            radius=(ball.x2 - ball.x1)/2,
            color=ball.colour,
            fill=True))

    plt.tight_layout()
    #plt.draw()
    plt.pause(0.05)
    plt.cla()


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
    table_factory = TableSetup(cfg)

    plt.ion()
    fix, (ax0, ax1) = plt.subplots(1, 2, sharey=True, figsize=(12, 12))

    detections = detector.detect()
    for image, detection in detections:
        table = table_factory.create_table(detection)
        #transpose_snooker_objects()

