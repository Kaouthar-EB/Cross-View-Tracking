"""
homography_tracker.py
---------------------
Cross-camera global ID assignment using foot-point projection,
Euclidean distance matching, AND appearance ReID verification.

WHY DISTANCE INSTEAD OF IoU:
- IoU compares bounding box overlap in pixel space.
- With oblique cameras, a person far away has their feet at the same
  pixel height as the head of someone nearby → boxes overlap randomly.
- Distance between projected FOOT POINTS on the ground plane is stable
  regardless of perspective distortion.

REID INTEGRATION:
- After geometry matching, appearance similarity is used to confirm or
  reject a proposed match between two cameras.
- When a new global_id is created, the crop is registered in the ReID gallery.
- At each frame, every active track's crop is registered to keep the gallery fresh.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


# ── Foot-point projection ──────────────────────────────────────────────────────

def _project_point(x, y, H):
    p = (H @ np.array([x, y, 1], dtype=np.float64)).flatten()
    return p[0] / p[2], p[1] / p[2]


def project_foot(bbox, H):
    """
    Project a bounding box into the reference frame using only its
    foot point (bottom-center, which lies on the ground plane z=0).
    Returns a reconstructed [x1, y1, x2, y2] in reference space.
    """
    x1, y1, x2, y2 = bbox[:4]

    foot_cx, foot_cy = (x1 + x2) / 2.0, float(y2)
    fc_x, fc_y = _project_point(foot_cx, foot_cy, H)

    bl_x, _ = _project_point(x1, y2, H)
    br_x, _ = _project_point(x2, y2, H)
    w2 = abs(br_x - bl_x)

    w_orig = max(x2 - x1, 1)
    h2 = w2 * ((y2 - y1) / w_orig)

    return [fc_x - w2 / 2, fc_y - h2, fc_x + w2 / 2, fc_y]


def modify_bbox_source(bboxes, H):
    """Project all bboxes via foot-point. Preserves columns beyond index 4."""
    if len(bboxes) == 0:
        return bboxes

    result = []
    for bbox in bboxes:
        proj = project_foot(bbox[:4], H)
        result.append(proj + list(bbox[4:]))
    return np.asarray(result, dtype=np.float32)


# ── Foot-point distance matching ───────────────────────────────────────────────

def _get_foot(bbox):
    """Bottom-center of a box."""
    return ((bbox[0] + bbox[2]) / 2.0, bbox[3])


def foot_distance_match(proj_i, proj_j, dist_thresh):
    """
    Match two sets of projected tracks by the Euclidean distance between
    their foot points using the Hungarian algorithm.

    Returns
    -------
    matches      : list of (idx_i, idx_j)
    unmatched_i  : list of unmatched indices in proj_i
    unmatched_j  : list of unmatched indices in proj_j
    """
    if len(proj_i) == 0 or len(proj_j) == 0:
        return [], list(range(len(proj_i))), list(range(len(proj_j)))

    feet_i = np.array([_get_foot(b) for b in proj_i])  # (N, 2)
    feet_j = np.array([_get_foot(b) for b in proj_j])  # (M, 2)

    cost = np.linalg.norm(feet_i[:, None] - feet_j[None, :], axis=2)

    row_ind, col_ind = linear_sum_assignment(cost)

    matches, matched_i, matched_j = [], set(), set()
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] <= dist_thresh:
            matches.append((r, c))
            matched_i.add(r)
            matched_j.add(c)

    unmatched_i = [i for i in range(len(proj_i)) if i not in matched_i]
    unmatched_j = [j for j in range(len(proj_j)) if j not in matched_j]

    return matches, unmatched_i, unmatched_j


# ── Multi-camera tracker ───────────────────────────────────────────────────────

class MultiCameraTracker:

    def __init__(
        self,
        homographies: list,
        dist_thresh: float = 60,
        id_memory: int = 45,
        reid=None,                    # ← NOUVEAU : instance de ReIDManager
        reid_thresh: float = 0.40,    # ← NOUVEAU : seuil cosine pour valider un match
    ):
        """
        Parameters
        ----------
        homographies : list of 3×3 matrices (camera 0 = reference = identity).
        dist_thresh  : max foot-point distance (pixels) to match two tracks.
        id_memory    : frames to remember a lost ID before discarding it.
        reid         : ReIDManager instance (optionnel — si None, désactivé).
        reid_thresh  : cosine similarity minimum pour confirmer un match inter-cam.
        """
        self.num_sources  = len(homographies)
        self.homographies = homographies
        self.dist_thresh  = dist_thresh
        self.id_memory    = id_memory
        self.reid         = reid          # ← NOUVEAU
        self.reid_thresh  = reid_thresh   # ← NOUVEAU

        self.next_id = 1
        self.ids  = [{} for _ in range(self.num_sources)]   # local_id → global_id
        self.age  = [{} for _ in range(self.num_sources)]   # local_id → frames seen
        self.lost = {}  # (cam, local_id) → (global_id, last_bbox, ttl)

    # ── Lost-buffer ────────────────────────────────────────────────────────────

    def _recall_lost(self, cam, local_id, proj_bbox):
        fx, fy = _get_foot(proj_bbox)
        best_gid, best_dist, best_key = None, float("inf"), None

        for (c, lid), (gid, last_bbox, ttl) in self.lost.items():
            if c != cam:
                continue
            lx, ly = _get_foot(last_bbox)
            d = np.sqrt((fx - lx) ** 2 + (fy - ly) ** 2)
            if d < self.dist_thresh and d < best_dist:
                best_gid, best_dist, best_key = gid, d, (c, lid)

        if best_key is not None:
            del self.lost[best_key]
        return best_gid

    def _tick_lost(self, active_ids):
        to_del = []
        for key, (gid, bbox, ttl) in self.lost.items():
            if gid in active_ids or ttl <= 1:
                to_del.append(key)
            else:
                self.lost[key] = (gid, bbox, ttl - 1)
        for k in to_del:
            del self.lost[k]

    # ── ReID helpers ───────────────────────────────────────────────────────────

    def _reid_verify(self, gid_i: int, gid_j: int) -> bool:
        """
        Retourne True si les deux global IDs ont une apparence compatible
        (ou si le ReID est désactivé / pas encore de gallery).
        """
        if self.reid is None:
            return True
        sim = self.reid.verify(gid_i, gid_j)
        return sim >= self.reid_thresh

    def _register_crops(self, tracks: list, frames: list) -> None:
        """
        Pour chaque track actif, enregistre le crop dans la gallery ReID
        en utilisant l'ID global déjà assigné.
        """
        if self.reid is None or frames is None:
            return

        for cam_idx, (trks, frame) in enumerate(zip(tracks, frames)):
            if frame is None:
                continue
            for row in trks:
                local_id = int(row[4])
                gid = self.ids[cam_idx].get(local_id)
                if gid is not None:
                    self.reid.register(gid, frame, row[:4])

    # ── Update ─────────────────────────────────────────────────────────────────

    def update(self, tracks: list, frames: list = None):
        """
        Parameters
        ----------
        tracks : list of (N_i, 6) arrays per camera — [x1,y1,x2,y2, id, cls]
        frames : list of raw BGR frames per camera (pour ReID). Optionnel.

        Returns
        -------
        self.ids — list of dicts local_track_id → global_id
        """
        proj_tracks = [
            modify_bbox_source(trks, self.homographies[i])
            for i, trks in enumerate(tracks)
        ]

        active_global_ids = set()

        for i in range(self.num_sources):
            for j in range(i + 1, self.num_sources):

                if len(proj_tracks[i]) == 0 and len(proj_tracks[j]) == 0:
                    continue

                matched_flags = {}

                matches, unmatches_i, unmatches_j = foot_distance_match(
                    proj_tracks[i], proj_tracks[j], self.dist_thresh
                )

                # ── Matched ────────────────────────────────────────────────
                for idx_i, idx_j in matches:
                    id_i = int(proj_tracks[i][idx_i][4])
                    id_j = int(proj_tracks[j][idx_j][4])

                    m_i   = self.ids[i].get(id_i)
                    m_j   = self.ids[j].get(id_j)
                    age_i = self.age[i].get(id_i, 0)
                    age_j = self.age[j].get(id_j, 0)

                    if m_i is not None and age_i >= age_j and not matched_flags.get(m_i):
                        # ── NOUVEAU : vérifier apparence avant d'accepter ──
                        if m_j is not None and m_i != m_j:
                            if not self._reid_verify(m_i, m_j):
                                # apparences incompatibles → traiter comme non-matché
                                unmatches_i.append(idx_i)
                                unmatches_j.append(idx_j)
                                continue
                        # ─────────────────────────────────────────────────
                        self.ids[j][id_j] = m_i
                        matched_flags[m_i] = True
                        active_global_ids.add(m_i)

                    elif m_j is not None and not matched_flags.get(m_j):
                        # ── NOUVEAU : vérifier apparence ──────────────────
                        if m_i is not None and m_i != m_j:
                            if not self._reid_verify(m_i, m_j):
                                unmatches_i.append(idx_i)
                                unmatches_j.append(idx_j)
                                continue
                        # ─────────────────────────────────────────────────
                        self.ids[i][id_i] = m_j
                        matched_flags[m_j] = True
                        active_global_ids.add(m_j)

                    else:
                        recovered = (
                            self._recall_lost(i, id_i, proj_tracks[i][idx_i]) or
                            self._recall_lost(j, id_j, proj_tracks[j][idx_j])
                        )
                        gid = recovered if recovered is not None else self.next_id
                        if recovered is None:
                            self.next_id += 1

                        self.ids[i][id_i] = gid
                        self.ids[j][id_j] = gid
                        matched_flags[gid] = True
                        active_global_ids.add(gid)

                    self.age[i][id_i] = age_i + 1
                    self.age[j][id_j] = age_j + 1

                # ── Unmatched i ────────────────────────────────────────────
                for idx_i in unmatches_i:
                    id_i = int(proj_tracks[i][idx_i][4])
                    m_i  = self.ids[i].get(id_i)

                    if m_i is None:
                        rec = self._recall_lost(i, id_i, proj_tracks[i][idx_i])

                        # ── NOUVEAU : ReID sur lost buffer ────────────────
                        if rec is None and self.reid is not None and frames is not None and frames[i] is not None:
                            emb = self.reid.extract_embedding(frames[i], proj_tracks[i][idx_i][:4])
                            if emb is not None:
                                rec, _ = self.reid.find_best_match(emb, self.reid.get_gallery_ids())
                        # ─────────────────────────────────────────────────

                        gid = rec if rec is not None else self.next_id
                        if rec is None:
                            self.next_id += 1
                        self.ids[i][id_i] = gid
                    else:
                        gid = m_i

                    matched_flags[gid] = True
                    active_global_ids.add(gid)
                    self.age[i][id_i] = self.age[i].get(id_i, 0) + 1
                    self.lost[(i, id_i)] = (gid, proj_tracks[i][idx_i], self.id_memory)

                # ── Unmatched j ────────────────────────────────────────────
                for idx_j in unmatches_j:
                    id_j = int(proj_tracks[j][idx_j][4])
                    m_j  = self.ids[j].get(id_j)

                    if m_j is None:
                        rec = self._recall_lost(j, id_j, proj_tracks[j][idx_j])

                        # ── NOUVEAU : ReID sur lost buffer ────────────────
                        if rec is None and self.reid is not None and frames is not None and frames[j] is not None:
                            emb = self.reid.extract_embedding(frames[j], proj_tracks[j][idx_j][:4])
                            if emb is not None:
                                rec, _ = self.reid.find_best_match(emb, self.reid.get_gallery_ids())
                        # ─────────────────────────────────────────────────

                        gid = rec if rec is not None else self.next_id
                        if rec is None:
                            self.next_id += 1
                        self.ids[j][id_j] = gid
                    else:
                        gid = m_j

                    matched_flags[gid] = True
                    active_global_ids.add(gid)
                    self.age[j][id_j] = self.age[j].get(id_j, 0) + 1
                    self.lost[(j, id_j)] = (gid, proj_tracks[j][idx_j], self.id_memory)

        self._tick_lost(active_global_ids)

        # ── NOUVEAU : mettre à jour la gallery ReID après chaque frame ────────
        self._register_crops(tracks, frames)
        # ──────────────────────────────────────────────────────────────────────

        return self.ids