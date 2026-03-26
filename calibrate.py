"""
calibrate.py
------------
Compute the homography matrix by clicking corresponding ground-level
points on a SINGLE side-by-side window (cam1 left | cam2 right).

HOW TO USE
----------
- One window opens showing both cameras side by side.
- Click a ground point in CAM1 (left), then the SAME point in CAM2 (right).
- Repeat at least 4 times (8-12 is ideal, spread across the overlap area).
- Points must be ON THE GROUND (floor tiles, markings) — not on people.
- ENTER = compute and save | R = reset all points | Q = quit
"""

import numpy as np
import cv2
from ultralytics import YOLO
import utilities


# ── State ──────────────────────────────────────────────────────────────────────

state = {
    "pts1":   [],
    "pts2":   [],
    "frame1": None,
    "frame2": None,
    "width1": 0,
}

COLORS = [
    (0, 0, 255), (0, 165, 255), (0, 255, 255), (0, 255, 0),
    (255, 0, 0), (255, 0, 255), (128, 0, 255), (0, 128, 255),
    (0, 255, 128), (255, 128, 0),
]


def get_color(idx):
    return COLORS[idx % len(COLORS)]


def redraw(win):
    f1 = state["frame1"].copy()
    f2 = state["frame2"].copy()
    pts1, pts2 = state["pts1"], state["pts2"]

    # Draw complete pairs
    for i, (p1, p2) in enumerate(zip(pts1, pts2)):
        c = get_color(i)
        cv2.circle(f1, p1, 7, c, -1)
        cv2.putText(f1, str(i + 1), (p1[0] + 10, p1[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, 2)
        cv2.circle(f2, p2, 7, c, -1)
        cv2.putText(f2, str(i + 1), (p2[0] + 10, p2[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, 2)

    # Draw pending cam1 point (waiting for cam2)
    if len(pts1) > len(pts2):
        p1 = pts1[-1]
        c  = get_color(len(pts1) - 1)
        cv2.circle(f1, p1, 7, c, -1)
        cv2.putText(f1, str(len(pts1)), (p1[0] + 10, p1[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, 2)

    canvas = np.hstack([f1, f2])
    n = len(pts2)

    if len(pts1) > len(pts2):
        status = f"Point {len(pts1)} in CAM1 done -- now click the SAME spot in CAM2"
        sc = (0, 165, 255)
    elif n == 0:
        status = "Click a ground point in CAM1 (left image)"
        sc = (255, 255, 0)
    else:
        status = f"{n} pair(s) done -- click next in CAM1, or ENTER to compute"
        sc = (0, 255, 0)

    cv2.line(canvas, (state["width1"], 0),
             (state["width1"], canvas.shape[0]), (255, 255, 255), 2)
    cv2.putText(canvas, "CAM 1", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(canvas, "CAM 2", (state["width1"] + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(canvas, status, (10, canvas.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, sc, 2)

    cv2.imshow(win, canvas)


def on_click(event, x, y, flags, param):
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    win  = param
    w1   = state["width1"]
    pts1 = state["pts1"]
    pts2 = state["pts2"]

    if x < w1:                           # click in cam1
        if len(pts1) == len(pts2):       # previous pair complete
            pts1.append((x, y))
            print(f"  CAM1 point {len(pts1)}: ({x},{y}) -- now click same in CAM2")
    else:                                # click in cam2
        if len(pts1) > len(pts2):        # waiting for cam2
            pts2.append((x - w1, y))
            print(f"  CAM2 point {len(pts2)}: ({x-w1},{y}) -- pair {len(pts2)} done")

    redraw(win)


# ── Verification ───────────────────────────────────────────────────────────────

def verify_homography(video1, video2, H, detector):
    """
    Verification using FOOT POINTS only (circles), not projected boxes.

    Why circles instead of boxes:
    - When cam1 and cam2 have very different viewpoints (street-level vs overhead),
      projecting a tall/narrow cam1 box into cam2 gives reversed horizontal boxes
      because the aspect ratio is completely different between views.
    - The homography is only valid for z=0 (ground plane).
    - So we project only the FOOT POINT (bottom-center, z=0) and draw a circle.
    - If circles land on the correct people in cam2 -> homography is good.

    Green boxes = real detections in each camera
    Red circles = foot points of cam1 detections projected into cam2
    """
    video2.set(cv2.CAP_PROP_POS_FRAMES, 17)
    print("\n[Verification]")
    print("  Green boxes  = real detections in each camera")
    print("  Red circles  = cam1 foot points projected into cam2")
    print("  Good result: red circles land ON the green boxes in cam2")
    print("Press any key to advance frames, Q to quit.\n")

    while True:
        ret1, frame1 = video1.read()
        ret2, frame2 = video2.read()
        if not ret1 or not ret2:
            break

        results = detector(
            [frame1, frame2], device="cpu",
            classes=[0], conf=0.30, verbose=False,
        )

        pred1 = results[0].boxes.data.cpu().numpy()[:, :4]
        pred2 = results[1].boxes.data.cpu().numpy()[:, :4]

        # Green boxes in both cameras
        utilities.draw_bounding_boxes(frame1, pred1, color=(0, 255, 0))
        utilities.draw_bounding_boxes(frame2, pred2, color=(0, 255, 0))

        # Project only foot points from cam1 → cam2, draw as red circles
        h2, w2 = frame2.shape[:2]
        for (x1, y1, x2, y2) in pred1:
            foot_x = (x1 + x2) / 2.0
            foot_y = float(y2)
            p  = (H @ np.array([foot_x, foot_y, 1], dtype=np.float64)).flatten()
            fx = int(p[0] / p[2])
            fy = int(p[1] / p[2])
            if 0 <= fx < w2 and 0 <= fy < h2:
                cv2.circle(frame2, (fx, fy), 8, (0, 0, 255), -1)
                cv2.circle(frame2, (fx, fy), 8, (255, 255, 255), 2)

        vis = np.hstack([frame1, frame2])
        cv2.namedWindow("Verification", cv2.WINDOW_NORMAL)
        cv2.imshow("Verification", vis)

        if cv2.waitKey(0) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


# ── Main ───────────────────────────────────────────────────────────────────────

def main(opts):
    video1 = cv2.VideoCapture(opts.video1)
    assert video1.isOpened(), f"Could not open {opts.video1}"
    video2 = cv2.VideoCapture(opts.video2)
    assert video2.isOpened(), f"Could not open {opts.video2}"

    _, frame1 = video1.read()
    _, frame2 = video2.read()

    # Resize both to same height for side-by-side display
    h      = min(frame1.shape[0], frame2.shape[0])
    scale1 = h / frame1.shape[0]
    scale2 = h / frame2.shape[0]
    frame1 = cv2.resize(frame1, (int(frame1.shape[1] * scale1), h))
    frame2 = cv2.resize(frame2, (int(frame2.shape[1] * scale2), h))

    state["frame1"] = frame1
    state["frame2"] = frame2
    state["width1"] = frame1.shape[1]
    state["pts1"]   = []
    state["pts2"]   = []

    win = "Calibration"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_click, win)

    print("=" * 60)
    print("  CALIBRATION — click matching ground points")
    print("=" * 60)
    print("  1. Click ON THE GROUND in CAM1 (left)")
    print("  2. Click the SAME spot in CAM2 (right)")
    print("  3. Repeat >= 4 times (8-12 ideal)")
    print("  ENTER=compute  |  R=reset  |  Q=quit")
    print()

    redraw(win)

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == 13:   # ENTER
            if len(state["pts2"]) < 4:
                print(f"  Need at least 4 pairs, got {len(state['pts2'])}.")
                continue
            break
        elif key == ord("r"):
            state["pts1"].clear()
            state["pts2"].clear()
            print("  Reset — all points cleared.")
            redraw(win)
        elif key == ord("q"):
            cv2.destroyAllWindows()
            print("Aborted.")
            return

    cv2.destroyAllWindows()

    # Scale clicked coords back to original frame resolution
    src = np.float32([
        (x / scale1, y / scale1) for x, y in state["pts1"]
    ]).reshape(-1, 1, 2)
    dst = np.float32([
        (x / scale2, y / scale2) for x, y in state["pts2"]
    ]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    inliers = int(mask.sum()) if mask is not None else len(state["pts1"])
    print(f"\n  Homography computed — {inliers}/{len(state['pts1'])} inliers.")

    out_path = f"{opts.homography_pth}.npy"
    np.save(out_path, H)
    print(f"  Saved -> {out_path}\n")

    print("Running verification (foot-point projection)...")
    detector = YOLO("yolov8n.pt")
    detector.to("cpu")
    verify_homography(video1, video2, H, detector)

    video1.release()
    video2.release()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--video1",         type=str, required=True)
    p.add_argument("--video2",         type=str, required=True)
    p.add_argument("--homography-pth", type=str, required=True,
                   help="Output path (no .npy extension needed).")
    main(p.parse_args())