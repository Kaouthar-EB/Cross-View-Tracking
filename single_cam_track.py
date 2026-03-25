"""
single_cam_track.py
-------------------
Per-camera BoT-SORT wrapper using the official ultralytics .track() API.

Each instance owns its own YOLO model so tracker state is isolated per camera.

Anti-swap logic:
  1. Trajectory-based swap correction for general overlaps.
  2. Overlap-zone freezing: when two tracks overlap heavily, their IDs
     are frozen and their last known position OUTSIDE the overlap is
     memorised. When they separate, each track is re-assigned to its
     closest memorised exit position → prevents identity confusion
     in the clean zones after a prolonged occlusion.
"""

import os
import yaml
import numpy as np
from collections import defaultdict
from ultralytics import YOLO


# ── BoT-SORT config ────────────────────────────────────────────────────────────

def _write_botsort_cfg(conf: float, max_age: int) -> str:
    cfg = {
        "tracker_type":     "botsort",
        "track_high_thresh": 0.25,
        "track_low_thresh":  0.05,
        "new_track_thresh":  0.60,
        "track_buffer":      60,
        "match_thresh":      0.85,
        "fuse_score":        True,
        "gmc_method":        "sparseOptFlow",
        "proximity_thresh":  0.5,
        "appearance_thresh": 0.25,
        "with_reid":         False,
    }
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "_botsort_custom.yaml"
    )
    with open(path, "w") as f:
        yaml.dump(cfg, f)
    return path


# ── Geometry helpers ───────────────────────────────────────────────────────────

def _center(box):
    """Centre (cx, cy) d'une boîte [x1,y1,x2,y2]."""
    return np.array([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0])


def _iou(boxA, boxB):
    """IoU entre deux boîtes [x1,y1,x2,y2]."""
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0


# ── ByteTracker (BoT-SORT backend) ────────────────────────────────────────────

class ByteTracker:
    """
    Single-camera BoT-SORT wrapper with two layers of anti-swap protection:

    Layer 1 — trajectory matching:
        When two tracks overlap, compare their predicted next positions
        to the observed positions. If swapped distances are smaller,
        correct the IDs.

    Layer 2 — overlap-zone freezing:
        While two tracks are in heavy overlap (IoU > freeze_iou_thresh),
        memorise the last clean (non-overlapping) center for each track.
        When they exit the overlap, re-assign each track to the closest
        memorised exit center using the Hungarian algorithm.
        This fixes identity confusion in the clean zones AFTER a prolonged
        occlusion — the most common real-world failure case.

    Parameters
    ----------
    model              : YOLOv8 model name or path.
    conf               : Detection confidence threshold.
    max_age            : Frames to keep a lost track alive.
    classes            : Class indices to detect. Default [0] = person only.
    device             : 'cpu' or 'cuda:0'.
    history_len        : Past centers kept per track for trajectory prediction.
    overlap_iou_thresh : IoU above which Layer 1 (trajectory swap fix) fires.
    freeze_iou_thresh  : IoU above which Layer 2 (zone freezing) fires.
                         Should be >= overlap_iou_thresh.
    """

    def __init__(
        self,
        model: str = "yolov8n.pt",
        conf: float = 0.25,
        max_age: int = 60,
        classes: list = None,
        device: str = "cpu",
        history_len: int = 30,
        overlap_iou_thresh: float = 0.40,
        freeze_iou_thresh: float  = 0.50,
    ):
        self.model = YOLO(model)
        self.model.to(device)
        self.conf    = conf
        self.classes = classes if classes is not None else [0]
        self.device  = device
        self.cfg_path = _write_botsort_cfg(conf, max_age)

        self.history_len        = history_len
        self.overlap_iou_thresh = overlap_iou_thresh
        self.freeze_iou_thresh  = freeze_iou_thresh

        # track_id → list of past centers (Layer 1)
        self.track_history: dict[int, list] = defaultdict(list)
        # track_id → last known box
        self.last_boxes: dict[int, np.ndarray] = {}

        # Layer 2 state
        # frozen_pairs: frozenset({id_a, id_b}) → True  (pairs currently overlapping)
        self._frozen_pairs: set = set()
        # exit_anchors: track_id → last clean center (recorded just before overlap)
        self._exit_anchors: dict[int, np.ndarray] = {}

    # ── Layer 1: trajectory-based swap correction ──────────────────────────────

    def _predict_center(self, track_id: int) -> np.ndarray | None:
        hist = self.track_history[track_id]
        if not hist:
            return None
        if len(hist) < 2:
            return hist[-1]
        window   = hist[-min(5, len(hist)):]
        velocity = (window[-1] - window[0]) / max(len(window) - 1, 1)
        return window[-1] + velocity

    def _resolve_swaps(self, raw_tracks: np.ndarray) -> np.ndarray:
        if len(raw_tracks) < 2:
            return raw_tracks

        tracks = raw_tracks.copy()
        n = len(tracks)

        for i in range(n):
            for j in range(i + 1, n):
                if _iou(tracks[i, :4], tracks[j, :4]) < self.overlap_iou_thresh:
                    continue

                idI, idJ = int(tracks[i, 4]), int(tracks[j, 4])
                predI = self._predict_center(idI)
                predJ = self._predict_center(idJ)
                if predI is None or predJ is None:
                    continue

                cI = _center(tracks[i, :4])
                cJ = _center(tracks[j, :4])

                dist_normal  = np.linalg.norm(predI - cI) + np.linalg.norm(predJ - cJ)
                dist_swapped = np.linalg.norm(predI - cJ) + np.linalg.norm(predJ - cI)

                if dist_swapped < dist_normal:
                    tracks[i, 4], tracks[j, 4] = tracks[j, 4], tracks[i, 4]

        return tracks

    # ── Layer 2: overlap-zone freezing ────────────────────────────────────────

    def _update_freeze_state(self, tracks: np.ndarray) -> np.ndarray:
        """
        Detect pairs entering / staying in / leaving the overlap zone.

        - Entering  → record exit_anchor for each track (last clean position).
        - Staying   → do nothing (keep frozen IDs as-is).
        - Exiting   → re-assign IDs by nearest exit_anchor (Hungarian).
        """
        if len(tracks) < 2:
            self._frozen_pairs.clear()
            return tracks

        tracks = tracks.copy()
        n = len(tracks)

        current_pairs: set = set()

        for i in range(n):
            for j in range(i + 1, n):
                iou = _iou(tracks[i, :4], tracks[j, :4])
                pair = frozenset({int(tracks[i, 4]), int(tracks[j, 4])})

                if iou >= self.freeze_iou_thresh:
                    current_pairs.add(pair)

                    if pair not in self._frozen_pairs:
                        # Just entered overlap → record clean anchors
                        for idx in (i, j):
                            tid = int(tracks[idx, 4])
                            if tid not in self._exit_anchors:
                                self._exit_anchors[tid] = _center(tracks[idx, :4])

        # Pairs that just EXITED the overlap zone
        exited_pairs = self._frozen_pairs - current_pairs
        for pair in exited_pairs:
            ids_in_pair = list(pair)
            # Find the current track rows for these IDs
            rows = {int(tracks[k, 4]): k for k in range(n)}
            present = [tid for tid in ids_in_pair if tid in rows]

            if len(present) < 2:
                # One track is lost — nothing to re-assign
                for tid in present:
                    self._exit_anchors.pop(tid, None)
                continue

            # Build cost matrix: track_row × anchor_id
            anchors_available = [tid for tid in present if tid in self._exit_anchors]
            if len(anchors_available) < 2:
                for tid in present:
                    self._exit_anchors.pop(tid, None)
                continue

            # Cost matrix: rows = current track positions, cols = anchors
            observed_centers = np.array([_center(tracks[rows[tid], :4]) for tid in present])
            anchor_centers   = np.array([self._exit_anchors[tid]         for tid in present])

            # Simple 2×2 Hungarian (works for any size)
            cost = np.linalg.norm(
                observed_centers[:, None] - anchor_centers[None, :], axis=2
            )

            from scipy.optimize import linear_sum_assignment
            row_ind, col_ind = linear_sum_assignment(cost)

            # Build re-assignment map: current_tid → correct_tid
            remap = {}
            for r, c in zip(row_ind, col_ind):
                observed_tid = present[r]   # what the tracker thinks
                correct_tid  = present[c]   # what it should be (closest anchor)
                if observed_tid != correct_tid:
                    remap[observed_tid] = correct_tid

            # Apply remap (swap IDs pairwise)
            if remap:
                tid_to_row = {int(tracks[k, 4]): k for k in range(n)}
                for obs_tid, cor_tid in remap.items():
                    if obs_tid in tid_to_row and cor_tid in tid_to_row:
                        r1 = tid_to_row[obs_tid]
                        r2 = tid_to_row[cor_tid]
                        tracks[r1, 4], tracks[r2, 4] = tracks[r2, 4], tracks[r1, 4]
                        # Update map for subsequent iterations
                        tid_to_row[cor_tid] = r1
                        tid_to_row[obs_tid] = r2

            # Clean anchors for this pair
            for tid in present:
                self._exit_anchors.pop(tid, None)

        self._frozen_pairs = current_pairs
        return tracks

    # ── History update ─────────────────────────────────────────────────────────

    def _update_history(self, tracks: np.ndarray) -> None:
        for row in tracks:
            tid    = int(row[4])
            center = _center(row[:4])
            hist   = self.track_history[tid]
            hist.append(center)
            if len(hist) > self.history_len:
                hist.pop(0)
            self.last_boxes[tid] = row[:4].copy()

    # ── Public API ─────────────────────────────────────────────────────────────

    def update(self, frame: np.ndarray) -> np.ndarray:
        """
        Run detection + BoT-SORT + anti-swap (Layer 1 + Layer 2) on one frame.

        Returns (M, 6) array → [x1, y1, x2, y2, track_id, cls]
        Returns empty (0, 6) when no active tracks.
        """
        results = self.model.track(
            frame,
            persist=True,
            tracker=self.cfg_path,
            device=self.device,
            classes=self.classes,
            conf=self.conf,
            iou=0.30,
            verbose=False,
        )

        result = results[0]
        boxes  = result.boxes

        if boxes is None or boxes.id is None or len(boxes.id) == 0:
            return np.empty((0, 6), dtype=np.float32)

        xyxy   = boxes.xyxy.cpu().numpy()
        ids    = boxes.id.cpu().numpy()
        labels = boxes.cls.cpu().numpy()

        raw_tracks = np.column_stack([xyxy, ids, labels]).astype(np.float32)

        # Layer 1 — trajectory-based swap correction
        tracks = self._resolve_swaps(raw_tracks)

        # Layer 2 — overlap-zone freezing + exit re-assignment
        tracks = self._update_freeze_state(tracks)

        # Update trajectory history with corrected IDs
        self._update_history(tracks)

        return tracks

    @property
    def names(self):
        return self.model.names