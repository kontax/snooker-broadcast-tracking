import numpy as np
from table_objects import Pocket
from table_objects import SnookerBall
from table_objects import SnookerTable


class TableSetup(object):
    """A factory class used to create a SnookerTable from network predictions"""

    def __init__(self, config):
        self._config = config
        self._transformation_matrix = np.zeros(shape=(3, 3), dtype=np.float32)
        self._object_counts = {
            "pocket": 4, "red": 15, "yellow": 1, "green": 1,
            "brown": 1, "blue": 1, "pink": 1, "black": 1, "white": 1}

    @property
    def transformation_matrix(self):
        """The transformation matrix used to view objects from a top-down view"""
        return self._transformation_matrix

    @staticmethod
    def _get_overlapped(collection):
        """
        Checks a collection of objects and returns all of those that overlap at all.
        (CONSIDER LOOKING AT INTERVAL TREE'S FOR THIS)
        :param collection: A TableObject collection to check for overlapping objects
        :return: The collection of overlapping objects as a collection of tuples
        """
        overlapped = []
        for i in range(0, len(collection)):
            for j in range(i, len(collection)):
                c1 = collection[i]
                c2 = collection[j]
                if np.array_equal(c1, c2):
                    continue
                if c1.is_overlapping(c2):
                    overlapped.append((c1, c2))

        return overlapped

    def _remove_overlapping_pockets(self, pockets):
        """
        Validates the collection of corner pockets, ensuring that there have been no
        more than 4 predictions, if there have then cleaning those that overlap to
        ensure only the one with the highest confidence interval remains.
        :param pockets: The collection of SnookerPocket objects
        :return: An updated collection of SnookerPocket objects without overlaps.
        """
        overlapping = self._get_overlapped(pockets)

        # If none overlap and we have 4 or less then nothing needs to be done
        if len(overlapping) == 0 and len(pockets) <= 4:
            return pockets

        original_count = len(pockets)

        # Otherwise we need to get the best fit
        for p1, p2 in overlapping:
            if p1.confidence >= p2.confidence:
                if p2 in pockets:
                    pockets.remove(p2)
            else:
                if p1 in pockets:
                    pockets.remove(p1)

        # If there are no changes then don't bother continuing
        if len(pockets) == original_count:
            return pockets

        # Recursively run the function in case multiple overlaps exist
        return self._remove_overlapping_pockets(pockets)

    @staticmethod
    def _sort_pockets(pockets):
        """
        Sorts the specified Pocket collection in order of top_left, bottom-left,
        bottom-right, top-right.
        :param pockets: The collection of Pocket objects to sort
        :return: A sorted collection of Pocket objects
        """
        # Sort Top to Bottom and extract
        pockets.sort(key=lambda x: x.y1)
        top_pockets = pockets[:2]
        bottom_pockets = pockets[2:]

        # Sort Left to Right inline
        top_pockets.sort(key=lambda x: x.x1)
        bottom_pockets.sort(key=lambda x: x.x1)

        top_left = top_pockets[0]
        top_right = top_pockets[1]
        bottom_left = bottom_pockets[0]
        bottom_right = bottom_pockets[1]

        return top_left, bottom_left, bottom_right, top_right

    def _set_transformation(self, pockets):
        """
        Sets the transformation matrix used for converting the region predictions to
        a top-down view with the same dimensions as a competition-sized snooker
        table. The scale depends on the multiplier value set in the config.
        Ref: http://math.stackexchange.com/a/339033
        :param pockets: The collection of Pocket objects.
        """
        # Only works with 4 pockets
        if len(pockets) != 4:
            return

        # Differentiate between the bottom/top and left/right pockets, as the order
        # of each is important for knowing which point transposes to which.

        top_left, bottom_left, bottom_right, top_right = self._sort_pockets(pockets)

        # Get the four points
        p1 = (bottom_left.x2, bottom_left.y1)
        p2 = (top_left.x2, top_left.y2)
        p3 = (top_right.x1, top_right.y2)
        p4 = (bottom_right.x1, bottom_right.y1)

        # Put three into a 3D matrix
        mat = np.array([[p1[0], p2[0], p3[0]],
                        [p1[1], p2[1], p3[1]],
                        [1, 1, 1]],
                       dtype=np.float32)

        # Using the last as a results vector
        v = np.array([[p4[0]],
                      [p4[1]],
                      [1]],
                     dtype=np.float32)

        # Solve the matrix equation
        s = np.linalg.inv(mat).dot(v)

        # Scale the original matrix by the coefficients in s to get the basis
        A = mat * np.transpose(s)

        # Perform the same on the destination coordinates, which are from (0,0) to
        # (w, h) where w and h are the width and height in mm of a snooker table by
        # the multiplier set in the config
        w = self._config.snooker.width * self._config.multiplier
        h = self._config.snooker.height * self._config.multiplier

        mat = np.array([[0, 0, w],
                        [h, 0, 0],
                        [1, 1, 1]],
                       dtype=np.float32)

        v = np.array([[w], [h], [1]], dtype=np.float32)
        s = np.linalg.inv(mat).dot(v)
        B = mat * np.transpose(s)

        # Invert B and multiply it by A to get the final matrix
        self._transformation_matrix = B.dot(np.linalg.inv(A))

    def _remove_extra_balls(self, colour, detections):
        """
        Occasionally the network will predict overlapping bounding boxes for balls
        where there is only one. This cannot be fully overcome here (especially for
        red balls where there are multiple on the table), however the best prediction
        is taken in each case.
        :param colour: The colour of the predicted snooker ball
        :param detections: The collection of detections made
        :return: The maximum allowed number of detections
        """
        max_allowed = self._object_counts[colour]
        if len(detections) <= max_allowed:
            return detections

        # Sort detections by confidence in descending order
        sorted_detections = detections[np.argsort(detections[:, 4])]

        # Loop through the sorted list until the maximum is reached, adding them
        # to the new collection
        output = []
        for i in xrange(max_allowed):
            output.append(sorted_detections[i])

        return output

    def _clean_predictions(self, detections):
        """
        Takes the detections that are made by the network and clean any based on the
        rules of the game, eg. Removing double counted colour balls.
        :param detections: The predictions made by the network
        :return: A tuple containing a collection of balls and pockets
        """
        # Extract predictions as TableObject's
        pockets = [Pocket(p) for p in detections['pocket']]
        ball_detections = [(key, value) for key, value
                           in detections.items()
                           if key != 'pocket']

        balls = []
        for colour, detections in ball_detections:
            detections = self._remove_extra_balls(colour, detections)
            for detection in detections:
                balls.append(SnookerBall(detection, colour))

        # Clean the pocket predictions
        pockets = self._remove_overlapping_pockets(pockets)

        return balls, pockets

    def create_table_test(self, detections, i):
        if i == 100:
            test = "here"
        return self.create_table(detections)

    def create_table(self, detections):
        """
        Creates a SnookerTable object containing a collection of Pocket and
        SnookerBall objects based on the detections made by the network. The bounding
        boxes of the TableObjects get transposed to a top-down view.
        :param detections: The predictions made by the neural network
        :return: A SnookerTable object populated with TableObjects
        """
        # Clean the predictions
        balls, pockets = self._clean_predictions(detections)

        # If that didn't work, continue and wait
        if len(pockets) > 4:
            return None

        # If no valid predictions have been made continue and wait
        if np.array_equal(
                self._transformation_matrix,
                np.zeros(shape=(3, 3), dtype=np.float32)) and len(pockets) < 4:
            return None

            # Set the transformation matrix
        self._set_transformation(pockets)

        # Transpose each ball
        for ball in balls:
            d = self._config.snooker.diameter * self._config.multiplier
            ball.transpose(self._transformation_matrix, d, d)

        # Transpose the pockets if there are 4
        if len(pockets) == 4:
            pockets = self._sort_pockets(pockets)
            for i in xrange(4):
                pockets[i].location = i
                pockets[i].transpose_and_reshape(self._transformation_matrix)

        return SnookerTable(pockets, balls)
