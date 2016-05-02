import numpy as np


class TableSetup(object):
    def __init__(self):
        self._object_counts = {
            "pocket": 4, "red": 15, "yellow": 1, "green": 1,
            "brown": 1, "blue": 1, "pink": 1, "black": 1}

    def _is_overlapped(self, object1, object2):
        bbox1 = object1[:4]
        bbox2 = object2[:4]

        # x1 >= x1 and x2 <= x2 and y1 >= y1 and y2 <= y2
        # (P2.y <= P3.y && P1.y >= P4.y && P2.x >= P3.x && P1.x <= P4.x )
        return bbox1[3] <= bbox2[1] and \
            bbox1[1] >= bbox2[3] and \
            bbox1[2] >= bbox2[0] and \
            bbox1[0] <= bbox2[2]

    def _has_overlapped(self, collection):
        """
        Check whether the objects in a collection are overlapping at all
        (CONSIDER LOOKING AT INTERVAL TREE'S FOR THIS)
        :param collection: The collection to check overlapping objects within
        :return: True if any objects overlap, false otherwise
        """
        for c1 in collection:
            for c2 in collection:
                if self._is_overlapped(c1, c2):
                    return True

        return False

    def _update_pockets(self, pockets):

        needs_updating = self._has_overlapped(pockets)

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
