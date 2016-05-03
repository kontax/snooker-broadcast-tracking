import matplotlib.pyplot as plt
import _init_paths
import numpy as np
from videos.youtube_video import YoutubeVideo
from model.table_setup import TableSetup
from model.detection import SnookerDetector
from os import path
from config import cfg
from fast_rcnn.config import cfg as caffe_cfg

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

    x = detector.detect()
    counter = 0
    for f, d in x:
        c = cleaner.clean_predictions(d)
        print c
        newimg = np.zeros(shape=(1280, 720, 3))
        #print "{0} : {1}".format(counter, x)
        # Transpose & clean
        # Turn into JSON
        # Send to RabbitMQ
        for x in xrange(len(f)):
            for y in xrange(len(f[0])):
                hom = c.dot(np.array([[x], [y], [1]]))
                newx = int(hom[0]/hom[2])
                newy = int(hom[1]/hom[2])

                try:
                    newimg[y][x] = f[newy][newx]
                except:
                    continue

        plt.imshow(newimg)
        plt.imshow(f)

        plt.show()
        counter += 1
