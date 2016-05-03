import numpy as np

import caffe
from fast_rcnn.nms_wrapper import nms
from fast_rcnn.test import im_detect


class SnookerDetector(object):
    """
    This class takes the SnookerVideo and CaffeModel details and outputs the object
    classifications and detections.
    """

    def __init__(self, config, video, model, prototxt, table_model, table_prototxt):
        """
        Instantiates a new SnookerDetector, used to output the detections of a
        trained network.
        :param config: The configuration details within a Config class
        :param video: The SnookerVideo containing the video details to detect
        :param model: The CaffeModel file containing the model weights
        :param prototxt: The solver prototxt with the model architecture
        :param table_model: The weights for the table detection model
        :param table_prototxt: The solver prototxt for the table detection model
        """
        self._config = config
        self._video = video
        self._model = model
        self._prototxt = prototxt
        self._table_model = table_model
        self._table_prototxt = table_prototxt
        self._classes = ('__background__',
                         'pocket', 'white', 'red', 'yellow', 'green',
                         'brown', 'blue', 'pink', 'black')

        # Counter used for the detection frame
        self._counter = 0

        # Get the video generator to loop through
        self._stream = self.video.play_video()

        # Setup Caffe
        if config.caffe_mode == 'CPU':
            caffe.set_mode_cpu()
        else:
            caffe.set_mode_gpu()
            caffe.set_device(config.gpu_device)

        # Set up the model to detect the image of tables
        self._setup_table_net(table_prototxt, table_model)

        self._net = caffe.Net(prototxt, model, caffe.TEST)
        print 'Loaded {:s}'.format(model)

        # Warm up model
        im = 128 * np.ones((720, 1280, 3), dtype=np.uint8)
        for i in xrange(2):
            _, _ = im_detect(self._net, im)

    @property
    def video(self):
        """Gets the SnookerVideo object to detect"""
        return self._video

    @video.setter
    def video(self, value):
        """Sets the SnookerVideo object to detect"""
        self._video = value

    def _setup_table_net(self, prototxt, model):
        """
        Set up the simple network that outputs whether the image specified is valid
        for performing object detection on. Only those images that have the full
        table from the top cushion should be used.
        :param prototxt: The prototxt file outlining the model architecture
        :param model: The weights of the trained model
        :return:
        """
        net = caffe.Net(prototxt, model, caffe.TEST)

        transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
        transformer.set_transpose('data', (2, 0, 1))  # move image channels
        transformer.set_raw_scale('data', 255)  # rescale pixel numbers
        transformer.set_channel_swap('data', (2, 1, 0))  # swap from RGB to BGR

        # Reshape image to batch of 1, 3 channels, H: 85, W: 150
        net.blobs['data'].reshape(1, 3, 85, 150)

        self._table_net = net
        self._table_transformer = transformer

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
            class_boxes = boxes[:, 4 * i:4 * (i + 1)]
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
            indices = np.where(results[:, -1] >= cfg.conf_threshold)[0]
            detections[c] = results[indices, ]

        return detections

    def _check_validity(self, net, frame):
        """
        Check whether the specified frame is valid for performing detection on, ie.
        it contains an image containing the full table at the correct angle.
        :param net: The caffe network used for detection
        :param frame: The image to check
        :return: True if the image is valid, False otherwise
        """
        t = self._table_transformer

        # Pre-process the frame and set it as the data blob in the network
        transformed_frame = t.preprocess('data', frame)
        net.blobs['data'].data[...] = transformed_frame

        # Run a forward pass and get the result
        output = net.forward()
        output_prob = output['score'][0]

        return output_prob.argmax() == 1

    def _clean(self, detections):
        pass

    def detect(self):
        """
        Run the detection network for each required frame in the SnookerVideo.
        :return: A generator with the detections made by the network.
        """
        # We run the model every "detection_frame" frame
        detection_frame = self._config.detection_frame

        for frame in self._stream:
            if self._counter % detection_frame == 0:
                self._counter += 1

                # Check if we can use the image for detection, else skip the frame
                if not self._check_validity(self._table_net, frame):
                    continue
                detections = self._get_detections(self._net, frame)
                # detections = self._clean(detections)
                yield (frame, detections)

            self._counter += 1
