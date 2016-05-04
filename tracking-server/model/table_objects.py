import numpy as np


class PocketLocation(object):
    """Enum used to specify the location of a Pocket"""
    bottom_left, bottom_right, top_left, top_right = xrange(4)


class SnookerTable(object):
    """
    Represents a table containing the four corner pockets and the snooker balls on
    the table
    """

    def __init__(self, pockets, balls):
        """
        Instantiate a new SnookerTable
        :param pockets: The collection of four Pocket objects
        :param balls: The collection of SnookerBall objects
        """
        if len(pockets) > 4:
            raise ValueError("The number of pockets in the collection cannot "
                             "exceed 4. Please ensure the detections are correct.")

        self._pockets = pockets
        self._balls = balls

    @property
    def pockets(self):
        """Gets the Pocket objects outlining the corners of the table"""
        return self._pockets

    @property
    def balls(self):
        """Gets the SnookerBall objects on the table"""
        return self._balls


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

    @property
    def centre(self):
        """The centre of the bounding box of the object"""
        return self.x1 + (self.x2 - self.x1), self.y1 + (self.y2 - self.y1)

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

    def transpose(self, matrix, width, height):
        """
        Given the transformation matrix specified, transform the current object to
        new coordinates, including the width and height that the new object
        dimensions should be.
        :param matrix: The transformation matrix
        :param width: The width of the destination object
        :param height: The height of the destination object
        """
        hom = matrix.dot(np.array([[self.centre[0]], [self.centre[1]], [1]]))
        new_centre_y = float(hom[1] / hom[2])
        new_centre_x = float(hom[0] / hom[2])

        self._y1 = new_centre_y - height / 2
        self._x1 = new_centre_x + width / 2

        self._y2 = new_centre_y + height / 2
        self._x2 = new_centre_x - width / 2


class Pocket(TableObject):
    """Represents a corner pocket on a snooker table"""

    def __init__(self, bounding_box):
        super(Pocket, self).__init__(bounding_box)

    def transpose_and_reshape(self, matrix, location):
        """
        Given the transformation matrix specified, transform the current Pocket to
        new coordinates using the bounding box corners, and reshape it to a single
        pixel. The corner taken to transform depends on the location of the Pocket
        on the table (ie. top-left Pocket uses the bottom-right corner point).
        :param matrix: The transformation matrix
        :param location: The PocketLocation of the pocket on the table
        """
        x = 0
        y = 0
        if location == PocketLocation.bottom_left:
            x = self.x2
            y = self.y1
        if location == PocketLocation.bottom_right:
            x = self.x1
            y = self.y1
        if location == PocketLocation.top_left:
            x = self.x2
            y = self.y2
        if location == PocketLocation.top_right:
            x = self.x1
            y = self.y2

        hom = matrix.dot(np.array([[x], [y], [1]]))
        self._y1 = int(hom[1] / hom[2])
        self._x1 = int(hom[0] / hom[2])

        self._y2 = self.y1
        self._x2 = self.x1


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
