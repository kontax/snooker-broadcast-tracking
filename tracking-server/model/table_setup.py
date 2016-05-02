import numpy as np


class TableSetup(object):
    def __init__(self):
        self._object_counts = {
            "pocket": 4, "red": 15, "yellow": 1, "green": 1,
            "brown": 1, "blue": 1, "pink": 1, "black": 1}

    def _is_overlapped(self, object1, object2):
        bbox1 = object1[:4]
        bbox2 = object2[:4]



    def _has_overlapped(self, collection):
        """
        Check whether the objects in a collection are overlapping at all
        :param collection: The collection to check overlapping objects within
        :return: True if any objects overlap, false otherwise
        """
        for c1 in collection:
            for c2 in collection:
                if self._is_overlapped(c1, c2):
                    return True

        return False

    def _update_pockets(self, pockets):

        needs_updating = False

        # Check if there's any overlapping pockets, if so take care of them
        for p1 in pockets:
            for p2 in pockets:
                if not self._is_overlapped(p1, p2):
                    continue

        # If we've got the correct number or lower of pockets we're done
        if pockets.length() <= self._object_counts['pocket']:
            return pockets

    def clean_predictions(self, detections):
        pockets = detections['pocket']
        balls = [{key: value} for key, value
                 in detections.items()
                 if key != 'pocket']

        updated_pockets = self._update_pockets(pockets)
