import unittest

import numpy as np

import homography_tracker


class FakeReIDManager:
    def __init__(self):
        self._gallery = {}
        self.last_extract_bbox = None

    def register(self, global_id, frame, bbox):
        self._gallery.setdefault(global_id, []).append(np.array([1.0], dtype=np.float32))

    def verify(self, gid_i, gid_j):
        return 1.0

    def extract_embedding(self, frame, bbox):
        self.last_extract_bbox = np.asarray(bbox, dtype=np.float32)
        return np.array([1.0], dtype=np.float32)

    def find_best_match(self, query_embedding, candidate_gids):
        if candidate_gids:
            return min(candidate_gids), 0.99
        return None, 0.0

    def get_gallery_ids(self):
        return sorted(self._gallery)


class MultiCameraTrackerReIDRegressionTests(unittest.TestCase):
    def test_new_person_does_not_reuse_active_gallery_id(self):
        reid = FakeReIDManager()
        tracker = homography_tracker.MultiCameraTracker(
            homographies=[np.eye(3), np.eye(3)],
            reid=reid,
        )
        frame = np.zeros((120, 120, 3), dtype=np.uint8)

        first_tracks = [
            np.array([[10, 10, 30, 60, 101, 0]], dtype=np.float32),
            np.empty((0, 6), dtype=np.float32),
        ]
        ids_after_first = tracker.update(first_tracks, frames=[frame, frame])
        self.assertEqual(ids_after_first[0][101], 1)
        self.assertEqual(reid.get_gallery_ids(), [1])

        second_tracks = [
            np.array([[70, 10, 90, 60, 202, 0]], dtype=np.float32),
            np.empty((0, 6), dtype=np.float32),
        ]
        ids_after_second = tracker.update(second_tracks, frames=[frame, frame])

        self.assertEqual(ids_after_second[0][202], 2)

    def test_reid_recovery_uses_original_bbox_not_projected_bbox(self):
        reid = FakeReIDManager()
        homography = np.array(
            [[2.0, 0.0, 5.0], [0.0, 1.5, 7.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        tracker = homography_tracker.MultiCameraTracker(
            homographies=[homography, np.eye(3)],
            reid=reid,
        )
        frame = np.zeros((200, 200, 3), dtype=np.uint8)

        tracker.lost[(1, 99)] = (7, np.array([0, 0, 20, 40], dtype=np.float32), 10)

        tracks = [
            np.array([[20, 30, 40, 90, 5, 0]], dtype=np.float32),
            np.empty((0, 6), dtype=np.float32),
        ]
        tracker.update(tracks, frames=[frame, frame])

        np.testing.assert_allclose(
            reid.last_extract_bbox,
            tracks[0][0][:4],
        )


if __name__ == "__main__":
    unittest.main()
