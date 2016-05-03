import numpy as np
from table_objects import Pocket
from table_objects import SnookerBall


class TableSetup(object):
    def __init__(self, config):
        self._config = config
        self._transformation_matrix = np.zeros(shape=(3, 3), dtype=np.float32)
        self._object_counts = {
            "pocket": 4, "red": 15, "yellow": 1, "green": 1,
            "brown": 1, "blue": 1, "pink": 1, "black": 1}

    @property
    def transformation_matrix(self):
        """The transformation matrix used to view objects from a top-down view"""
        return self._transformation_matrix

    def _get_overlapped(self, collection):
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

        # Otherwise we need to get the best fit
        for p1, p2 in overlapping:
            if p1.confidence >= p2.confidence:
                pockets.remove(p2)
            else:
                pockets.remove(p1)

        # Recursively run the function in case multiple overlaps exist
        return self._remove_overlapping_pockets(pockets)

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

        # The top and bottom pockets need to be differentiated as the bottom-left
        # corner is taken from the top pockets, and the top-right corner is taken
        # from the bottom pockets
        pockets.sort(key=lambda x: x.y1)
        top_pockets = pockets[:2]
        bottom_pockets = pockets[2:]
        top_pockets.sort(key=lambda x: x.x1)
        bottom_pockets.sort(key=lambda x: x.x1)

        # Get the four points
        p1 = (bottom_pockets[0].x2, bottom_pockets[0].y1)
        p2 = (top_pockets[0].x2, top_pockets[0].y2)
        p3 = (top_pockets[1].x2, top_pockets[1].y2)
        p4 = (bottom_pockets[1].x2, bottom_pockets[1].y1)

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
        self._transformation_matrix = A.dot(np.linalg.inv(B))

    def clean_predictions(self, detections):

        # Extract predictions as TableObject's
        pockets = [Pocket(p) for p in detections['pocket']]
        ball_detections = [(key, value) for key, value
                           in detections.items()
                           if key != 'pocket']

        balls = []
        for colour, detections in ball_detections:
            for detection in detections:
                balls.append(SnookerBall(detection, colour))

        # Clean the pocket predictions
        updated_pockets = self._remove_overlapping_pockets(pockets)

        # Set the transformation matrix
        self._set_transformation(updated_pockets)
        return self._transformation_matrix
