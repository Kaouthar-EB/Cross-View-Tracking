import cv2
import numpy as np


# ── Homography helpers ─────────────────────────────────────────────────────────

def _project_point(x, y, H):
    """Project a single 2D point through homography H. Returns (x', y')."""
    p = (H @ np.array([x, y, 1], dtype=np.float64)).flatten()
    return p[0] / p[2], p[1] / p[2]


def apply_homography(uv, H):
    """Project an array of (u,v) points through H."""
    uv_ = np.zeros_like(uv, dtype=np.float64)
    for idx, (u, v) in enumerate(uv):
        uv_[idx] = _project_point(u, v, H)
    return uv_


def apply_homography_xyxy(xyxy, H):
    """
    Project bounding boxes using FOOT-POINT projection.

    WHY:
    A planar homography is only valid for z=0 (ground plane).
    Projecting all 4 corners of a box fails because the top corners
    (head, z≈1.7m) are NOT on the ground — H distorts them badly,
    producing huge stretched rectangles.

    HOW:
    1. Project only the BOTTOM-CENTER (foot point, z=0) — this is correct.
    2. Project the two bottom corners to estimate the width in cam2.
    3. Reconstruct the full box using the original aspect ratio.

    Result: a properly sized and positioned box in cam2 space.
    """
    if len(xyxy) == 0:
        return np.empty((0, 4), dtype=np.float32)

    result = []
    for (x1, y1, x2, y2) in xyxy:
        # -- Step 1: project foot center (bottom-center of box)
        foot_cx = (x1 + x2) / 2.0
        foot_cy = float(y2)
        fc_x, fc_y = _project_point(foot_cx, foot_cy, H)

        # -- Step 2: project bottom corners to get width in cam2
        bl_x, _ = _project_point(x1, y2, H)   # bottom-left
        br_x, _ = _project_point(x2, y2, H)   # bottom-right
        w2 = abs(br_x - bl_x)

        # -- Step 3: keep original aspect ratio for height
        w_orig = max(x2 - x1, 1)
        h_orig = y2 - y1
        h2 = w2 * (h_orig / w_orig)

        # -- Step 4: build box (foot point is at the bottom-center)
        result.append([
            fc_x - w2 / 2,   # x1
            fc_y - h2,        # y1
            fc_x + w2 / 2,   # x2
            fc_y,             # y2
        ])

    return np.array(result, dtype=np.float32)


# ── Drawing helpers ────────────────────────────────────────────────────────────

def draw_bounding_boxes(image, bounding_boxes, color=(0, 255, 0), thickness=2):
    for bbox in bounding_boxes:
        x1, y1, x2, y2 = np.intp(bbox)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)


def draw_matches(img1, kpts1, img2, kpts2, matches):
    vis = np.hstack([img1, img2])
    MAX_DIST_VAL = max([m.distance for m in matches])
    WIDTH = img2.shape[1]
    for src, dst, match in zip(kpts1, kpts2, matches):
        src_x, src_y = src
        dst_x, dst_y = dst
        dst_x += WIDTH
        color = (0, int(255 * (match.distance / MAX_DIST_VAL)), 0)
        vis = cv2.line(vis, (src_x, src_y), (dst_x, dst_y), color, 1)
    return vis


def color_from_id(gid):
    np.random.seed(int(gid))
    return np.random.randint(0, 255, size=3).tolist()


def draw_tracks(image, tracks, ids_dict, src, classes=None):
    """Draw bounding boxes with label 'id: X'. No trails."""
    vis = np.array(image)
    bboxes = tracks[:, :4]
    ids    = tracks[:, 4]

    for i, box in enumerate(bboxes):
        local_id  = int(ids[i])
        global_id = ids_dict.get(local_id)
        if global_id is None:
            continue

        color = color_from_id(global_id)
        x1, y1, x2, y2 = np.intp(box)

        # Box
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness=2)

        # Label background + text
        text = f"id: {global_id}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(vis, text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

    return vis