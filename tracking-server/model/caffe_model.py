class CaffeModel(object):
    """A wrapper for a caffe model including its weights and configuration file"""

    def __init__(self, prototxt, weights):
        """
        Initialize a new CaffeModel
        :param prototxt: The configuration file containing the network architecture
        :param weights: The trained weights of the model
        """
        self._prototxt = prototxt
        self._weights = weights

    @property
    def prototxt(self):
        """Gets the configuration file containing the network architecture"""
        return self._prototxt

    @property
    def weights(self):
        """Gets the trained weights of the model"""
        return self._weights
