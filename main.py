import argparse

import cv2
import numpy as np

import homography_tracker
import utilities
from single_cam_track import ByteTracker

try:
    from reid_manager import ReIDManager
    _REID_IMPORT_ERROR = None
except Exception as exc:
    ReIDManager = None
    _REID_IMPORT_ERROR = exc


def _build_reid():
    if ReIDManager is None:
        if _REID_IMPORT_ERROR is not None:
            print(f"[WARN] ReID disabled: {_REID_IMPORT_ERROR}")
        return None

    try:
        return ReIDManager(
            max_embeddings_per_id=20,
            accept_thresh=0.40,
        )
    except Exception as exc:
        print(f"[WARN] ReID disabled: {exc}")
        return None


def main(opts):
    video1 = cv2.VideoCapture(opts.video1)
    assert video1.isOpened(), f"Could not open {opts.video1}"
    video2 = cv2.VideoCapture(opts.video2)
    assert video2.isOpened(), f"Could not open {opts.video2}"

    cam4_H_cam1 = np.load(opts.homography)
    cam1_H_cam4 = np.linalg.inv(cam4_H_cam1)
    homographies = [np.eye(3), cam1_H_cam4]

    trackers = [
        ByteTracker(
            model="yolov8n.pt",
            conf=opts.conf,
            max_age=opts.max_age,
            classes=[0],
            device="cpu",
        )
        for _ in range(2)
    ]

    global_tracker = homography_tracker.MultiCameraTracker(
        homographies,
        dist_thresh=opts.dist_thresh,
        id_memory=opts.max_age,
        reid=_build_reid(),
    )

    num_frames = int(min(
        video1.get(cv2.CAP_PROP_FRAME_COUNT),
        video2.get(cv2.CAP_PROP_FRAME_COUNT),
    ))

    video2.set(cv2.CAP_PROP_POS_FRAMES, 17)

    for _ in range(num_frames):
        frame1 = video1.read()[1]
        frame2 = video2.read()[1]

        if frame1 is None or frame2 is None:
            break

        frames = [frame1, frame2]
        tracks = [trackers[i].update(frames[i]) for i in range(2)]
        global_ids = global_tracker.update(tracks, frames=frames)

        out = [frame1.copy(), frame2.copy()]
        for i in range(2):
            if len(tracks[i]) > 0:
                out[i] = utilities.draw_tracks(
                    out[i], tracks[i], global_ids[i], i,
                    classes=trackers[i].names,
                )

        cv2.namedWindow("Vis", cv2.WINDOW_NORMAL)
        cv2.imshow("Vis", np.hstack(out))
        if cv2.waitKey(1) == ord("q"):
            break

    video1.release()
    video2.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video1", type=str, default="./epfl/cam1.mp4")
    parser.add_argument("--video2", type=str, default="./epfl/cam2.mp4")
    parser.add_argument("--homography", type=str, default="./my_matrix.npy")
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Detection confidence threshold.",
    )
    parser.add_argument(
        "--max-age",
        type=int,
        default=30,
        help="Frames to keep a lost track alive.",
    )
    parser.add_argument(
        "--dist-thresh",
        type=float,
        default=60,
        help="Max foot-point distance (pixels) to match same person across cameras.",
    )
    main(parser.parse_args())
