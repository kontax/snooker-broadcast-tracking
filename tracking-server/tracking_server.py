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

    data_path = path.join('py-faster-rcnn', 'data', 'snooker')
    model_path = path.join('py-faster-rcnn', 'models', 'snooker_net')

    #video = YoutubeVideo(url="https://www.youtube.com/watch?v=RIOi3YKtBcY")
    video = YoutubeVideo(url="https://www.youtube.com/watch?v=irpfzXXPrX8")
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

    x = detector.detect()
    counter = 0
    for f, d in x:
        print counter
        plt.imshow(f)
        plt.show()
        counter += 1
