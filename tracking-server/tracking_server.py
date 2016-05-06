from os import path

import _init_paths
import argparse
import cv2
import os
import sys

from config import cfg
from fast_rcnn.config import cfg as caffe_cfg
from model.caffe_model import CaffeModel
from model.detection import SnookerDetector
from model.table_setup import TableSetup
from server.messaging_server import MessagingServer
from videos.youtube_video import YoutubeVideo

caffe_modes = ['GPU', 'CPU']


def parse_args():

    parser = argparse.ArgumentParser(description="Snooker Tracking Server")
    parser.add_argument('--stream_link', dest='stream_link',
                        help='The Youtube video link to load and parse.',
                        default=None, type=str)
    parser.add_argument('--gpu', dest='gpu_device', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--caffe_mode', dest='caffe_mode',
                        help="Whether to use the 'GPU' (default) or 'CPU'",
                        choices=caffe_modes, default='GPU', type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()
    cfg.caffe_mode = args.caffe_mode
    cfg.gpu_device = args.gpu_device
    video = YoutubeVideo(args.stream_link)

    if video.width != 1280 or video.height != 720:
        raise StandardError("The video supplied needs to be 1280x720 for detection.")

    # Use RPN for proposals
    caffe_cfg.TEST.HAS_RPN = True

    # Set up the necessary objects for detection
    wd = os.getcwd()
    data_path = path.join(wd, 'py-faster-rcnn', 'data', 'snooker')
    model_path = path.join(wd, 'py-faster-rcnn', 'models', 'snooker_net')

    model = path.join(data_path, 'snooker.caffemodel')
    prototxt = path.join(model_path, 'snooker',
                         'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
    detection_model = CaffeModel(prototxt, model)

    model = path.join(data_path, 'snooker_table.caffemodel')
    prototxt = path.join(model_path, 'snooker_table', 'snooker_table.pt')
    table_model = CaffeModel(prototxt, model)

    messenger = MessagingServer(cfg.server)
    messenger.connect()
    messenger.send(" [x] Loaded video from {0}".format(video.video_source), "log")

    #video = YoutubeVideo(url="https://www.youtube.com/watch?v=RIOi3YKtBcY")
    #video = YoutubeVideo(url="https://www.youtube.com/watch?v=irpfzXXPrX8")
    #video = YoutubeVideo(url="https://www.youtube.com/watch?v=0uFQqiaMuH8")
    #video = YoutubeVideo(url="https://www.youtube.com/watch?v=8rtzUPY_t6U")
    table_factory = TableSetup(cfg)
    detector = SnookerDetector(
        config=cfg, video=video,
        detection_model=detection_model, table_model=table_model)

    # Run the detection algorithm
    detections = detector.detect()
    i = 0
    for image, detection in detections:

        # Convert the detection to a SnookerTable object if possible
        table = table_factory.create_table_test(detection, i)
        if table is None:
            continue
        print "Frame {0}".format(i)

        # Get the json and log details to send to the RabbitMQ server
        json = table.to_json()
        messenger.send(json, "json")
        messenger.send(json, "log")
        print "Sent JSON"

        # Convert the frame to send to the server
        img = cv2.imencode('.jpg', image)[1].tostring()
        messenger.send(img, "stream")
        print "Sent image byte array"
        i += 1

    print "Stream complete"
    messenger.disconnect()

