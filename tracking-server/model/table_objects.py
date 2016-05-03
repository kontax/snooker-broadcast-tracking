import numpy as np


class TableObject(object):
    """
    Represents an object that has been detected on a snooker table, such as a ball
    or pocket
    """

    def __init__(self, bounding_box):
        """
        Instantiates a new TableObject
        :param bounding_box: The numpy array containing the x/y points and confidence
        from the neural network prediction
        """
        self._x1 = bounding_box[0]
        self._y1 = bounding_box[1]
        self._x2 = bounding_box[2]
        self._y2 = bounding_box[3]
        self._confidence = bounding_box[4]

    @property
    def x1(self):
        """The x value for the top left corner point"""
        return self._x1

    @property
    def y1(self):
        """The y value for the top left corner point"""
        return self._y1

    @property
    def x2(self):
        """The x value for the bottom right corner point"""
        return self._x2

    @property
    def y2(self):
        """The y value for the bottom right corner point"""
        return self._y2

    @property
    def confidence(self):
        """The confidence of the prediction made by the network"""
        return self._confidence

    def is_left(self, table_object):
        """Whether the specified object is fully to the left of this one"""
        return self.x2 < table_object.x1

    def is_right(self, table_object):
        """Whether the specified object is fully to the right of this one"""
        return self.x1 > table_object.x2

    def is_above(self, table_object):
        """Whether the specified object is fully above this one"""
        return self.y2 < table_object.y1

    def is_below(self, table_object):
        """Whether the specified object is fully below this one"""
        return self.y1 > table_object.y2

    def is_overlapping(self, table_object):
        """
        Whether the current table object overlaps the specified one
        :param table_object: The object to test overlapping
        :return: True if the objects overlap, false otherwise
        """
        return not self.is_above(table_object) and \
               not self.is_below(table_object) and \
               not self.is_left(table_object) and \
               not self.is_right(table_object)

    def transpose(self, matrix):
        """
        Given the transformation matrix specified, transform the current object to
        new coordinates.
        :param matrix: The transformation matrix
        """
        hom = matrix.dot(np.array([[self.x1], [self.y1], [1]]))
        self._y1 = float(hom[1] / hom[2])
        self._x1 = float(hom[0] / hom[2])

        hom = matrix.dot(np.array([[self.x2], [self.y2], [1]]))
        self._y2 = float(hom[1] / hom[2])
        self._x2 = float(hom[0] / hom[2])


class Pocket(TableObject):
    """Represents a corner pocket on a snooker table"""

    def __init__(self, bounding_box):
        super(Pocket, self).__init__(bounding_box)


class SnookerBall(TableObject):
    """Represents a ball on a snooker table"""

    def __init__(self, bounding_box, colour):
        """
        Instantiates a new SnookerBall object
        :param bounding_box: The numpy array containing the x/y points and confidence
        :param colour: The colour of the snooker ball
        """
        super(SnookerBall, self).__init__(bounding_box)
        self._colour = colour

    @property
    def colour(self):
        """The colour of the snooker ball"""
        return self._colour
