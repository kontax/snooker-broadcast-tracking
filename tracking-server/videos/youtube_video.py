import cv2
import pafy
from snooker_video import SnookerVideo


class YoutubeVideo(SnookerVideo):
    """
    This class is a derivation of the SnookerVideo class, which takes a YouTube url
    as a constructor parameter in order to stream the video.
    """

    def __init__(self, url):
        """
        Initializes a new YoutubeVideo
        :param url: The full URL, or 11 char video ID of the video
        """
        video = pafy.new(url)
        stream = video.getbestvideo(preftype='m4v', ftypestrict=False)
        (width, height) = stream.dimensions
        frame_rate = 30  # Guessing default frame rate for videos
        SnookerVideo.__init__(self, width, height, frame_rate)

        # Setup OpenCV
        cap = cv2.VideoCapture()
        res = cap.open(stream.url)

        if not res:
            raise ValueError("The specified URL cannot be read by OpenCV")

        self._stream = stream
        self._cap = cap

    @property
    def stream(self):
        """Gets the stream object containing the video data"""
        return self._stream

    def play_video(self):
        """
        Generator used to loop through the video frame by frame.
        :return: A generator containing a frame each pass as a numpy array.
        """
        cap = self._cap
        while cap.isOpened():
            ret, frame = cap.read()
            yield frame

