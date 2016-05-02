import matplotlib.pyplot as plt
import _init_paths
from videos.youtube_video import YoutubeVideo
from model.detection import SnookerDetector
from os import path
from config import cfg
from fast_rcnn.config import cfg as caffe_cfg

if __name__ == '__main__':

    # Use RPN for proposals
    caffe_cfg.TEST.HAS_RPN = True

    video = YoutubeVideo(url="https://www.youtube.com/watch?v=RIOi3YKtBcY")
    model = path.join('py-faster-rcnn', 'data', 'snooker', 'snooker.caffemodel')
    prototxt = path.join('py-faster-rcnn', 'models', 'snooker_net', 'snooker',
                         'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
    detector = SnookerDetector(config=cfg,
                               video=video,
                               model=model,
                               prototxt=prototxt)

    x = detector.detect()
    counter = 0
    for f, d in x:
        print counter
        plt.imshow(f)
        plt.show()
        counter += 1
