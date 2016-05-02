import numpy as np

import caffe
from fast_rcnn.nms_wrapper import nms
from fast_rcnn.test import im_detect


class SnookerDetector(object):
    """
    This class takes the SnookerVideo and CaffeModel details and outputs the object
    classifications and detections.
    """

    def __init__(self, config, video, model, prototxt):
        """
        Instantiates a new SnookerDetector, used to output the detections of a
        trained network.
        :param config: The configuration details within a Config class
        :param video: The SnookerVideo containing the video details to detect
        :param model: The CaffeModel file containing the model weights
        :param prototxt: The solver prototxt with the model architecture
        """
        self._config = config
        self._video = video
        self._model = model
        self._prototxt = prototxt
        self._classes = ('__background__',
                         'pocket', 'white', 'red', 'yellow', 'green',
                         'brown', 'blue', 'pink', 'black')

    @property
    def video(self):
        """Gets the SnookerVideo object to detect"""
        return self._video

    @video.setter
    def video(self, value):
        """Sets the SnookerVideo object to detect"""
        self._video = value

    def _get_detections(self, net, frame):
        """
        Run the network model on the specified frame, retrieving the detected
        bounding boxes for each of the possible classes.
        :param net: The Caffe network used to perform detection
        :param frame: The image to detect
        :return: A dictionary containing a horizontal stack of bounding boxes and
        confidence thresholds for each possible class, eg:
            detections['red'] = (x1, y1, x2, y2, 0.9)
        """
        cfg = self._config
        detections = {}

        # Get the detections from the network
        scores, boxes = im_detect(net, frame)

        # Loop through each class possible, except the background
        for i, c in enumerate(self._classes[1:]):
            i += 1

            # Extract the bounding boxes and scores from the detected boxes
            class_boxes = boxes[:, 4*i:4*(i + 1)]
            class_scores = scores[:, i]

            # Stack te results into a numpy horizontal stack for NMS detection
            results = np.hstack((class_boxes,
                                 class_scores[:, np.newaxis])).astype(np.float32)

            # Get the indices of the results that meet the non-maximum suppression
            # (NMS) threshold
            keep = nms(results, cfg.nms_threshold)
            results = results[keep, :]

            # Add a dictionary entry to the results where the confidence threshold
            # has been met
            detections[c] = np.where(results[:, -1] >= cfg.conf_threshold)[0]

        return detections

    def _clean(self, detections):
        pass

    def detect(self):
        """
        Run the detection network for each required frame in the SnookerVideo.
        :return: A generator with the detections made by the network.
        """
        cfg = self._config

        # Get the video generator to loop through
        stream = self.video().play_video()

        # We run the model every "detection_frame" frame
        detection_frame = cfg.detection_frame

        # Setup Caffe
        if cfg.caffe_mode == 'CPU':
            caffe.set_mode_cpu()
        else:
            caffe.set_mode_gpu()
            caffe.set_device(cfg.gpu_device)

        net = caffe.Net(self._prototxt, self._model, caffe.TEST)
        print 'Loaded {:s}'.format(self._model)

        # Warm up model
        im = 128 * np.ones((720, 1280, 3), dtype=np.uint8)
        for i in xrange(2):
            _, _ = im_detect(net, im)

        counter = 0
        for frame in stream:
            if counter % detection_frame == 0:
                detections = self._get_detections(net, frame)
                yield self._clean(detections)

            counter += 1
