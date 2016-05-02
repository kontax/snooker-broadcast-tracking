class SnookerVideo(object):
    """
    This class represents a video of a snooker match, whether it be from a file, a
    stream or another source.
    """
    def __init__(self, video_source, width, height, frame_rate):
        self._width = width
        self._height = height
        self._frame_rate = frame_rate
        self._video_source = video_source
        return

    @property
    def width(self):
        """The width of the video in pixels"""
        return self._width

    @property
    def height(self):
        """The height of the video in pixels"""
        return self._height

    @property
    def frame_rate(self):
        """The number of frames the video plays every second"""
        return self._frame_rate

    @property
    def video_source(self):
        """The source of the video to play"""
        return self._video_source

    def play_video(self):
        """
        Play the video, generating a frame at a time until all frames are exhausted.
        :return: A generator looping through the video
        """
        raise NotImplementedError(
            "This method needs to be implemented by a derived class"
        )
