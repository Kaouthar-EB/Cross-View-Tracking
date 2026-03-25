"""
reid_manager.py
---------------
Appearance-based Re-Identification using OSNet (torchreid).

OSNet (Omni-Scale Network) is a lightweight CNN designed specifically
for person re-identification. It extracts a 512-dim feature vector
per person crop, which is far more discriminative than color histograms.

Install:
    pip install torchreid
    # or from source:
    git clone https://github.com/KaiyangZhou/deep-person-reid
    cd deep-person-reid && pip install -e .

The model weights are downloaded automatically on first run by torchreid
and cached in ~/.cache/torch/checkpoints/.

How it works:
1. For each tracked person, crop their bounding box from the frame.
2. Preprocess the crop (resize to 256x128, normalize).
3. Forward pass through OSNet → 512-dim L2-normalized embedding.
4. Maintain a gallery: global_id → list of embeddings.
5. verify(gid_i, gid_j) → cosine similarity to confirm inter-cam matches.
6. find_best_match(embedding, candidates) → recover lost IDs by appearance.
"""

import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import sys
from pathlib import Path

_LOCAL_TORCHREID_ROOT = Path(__file__).resolve().parent / "deep-person-reid"
if _LOCAL_TORCHREID_ROOT.exists():
    sys.path.insert(0, str(_LOCAL_TORCHREID_ROOT))

import torchreid


# ── OSNet input spec ───────────────────────────────────────────────────────────
_INPUT_H = 256
_INPUT_W = 128

_TRANSFORM = T.Compose([
    T.ToPILImage(),
    T.Resize((_INPUT_H, _INPUT_W)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]),
])


def _build_osnet(device: str) -> torch.nn.Module:
    """
    Build OSNet-x1.0 pretrained on Market-1501 + MSMT17 via torchreid.
    The model is downloaded automatically on first call.
    """
    model = torchreid.models.build_model(
        name="osnet_x1_0",
        num_classes=1000,   # will be ignored — we use features only
        pretrained=True,
    )
    model.eval()
    model.to(device)
    return model


# ── Embedding extraction ───────────────────────────────────────────────────────

def _extract_embedding(
    model: torch.nn.Module,
    frame: np.ndarray,
    bbox,
    device: str,
) -> np.ndarray | None:
    """
    Crop the person from the frame, run OSNet, return a 512-dim
    L2-normalized embedding as a numpy float32 vector.

    Returns None if the crop is too small.
    """
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

    # Clamp to frame boundaries
    h_f, w_f = frame.shape[:2]
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w_f, x2); y2 = min(h_f, y2)

    if (x2 - x1) < 16 or (y2 - y1) < 32:
        return None  # crop too small for meaningful ReID

    crop = frame[y1:y2, x1:x2]
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    tensor = _TRANSFORM(crop_rgb).unsqueeze(0).to(device)  # (1, 3, 256, 128)

    with torch.no_grad():
        feat = model(tensor)                               # (1, 512)
        feat = F.normalize(feat, p=2, dim=1)              # L2-normalize

    return feat.squeeze(0).cpu().numpy().astype(np.float32)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two L2-normalized vectors (= dot product)."""
    return float(np.dot(a, b))


def _mean_embedding(embeddings: list) -> np.ndarray:
    """Average pool a list of embeddings and re-normalize."""
    stacked = np.stack(embeddings, axis=0).mean(axis=0)
    norm = np.linalg.norm(stacked)
    return stacked / norm if norm > 1e-6 else stacked


# ── ReID Manager ───────────────────────────────────────────────────────────────

class ReIDManager:
    """
    OSNet-based Re-ID gallery manager.

    Parameters
    ----------
    device               : 'cpu' or 'cuda:0'.
    max_embeddings_per_id: Rolling gallery size per global ID.
    accept_thresh        : Cosine similarity threshold to accept a match.
                           OSNet embeddings are high-quality → use ~0.65.
    """

    def __init__(
        self,
        device: str = "cpu",
        max_embeddings_per_id: int = 20,
        accept_thresh: float = 0.65,
    ):
        self.device               = device
        self.max_embeddings_per_id = max_embeddings_per_id
        self.accept_thresh        = accept_thresh

        print("[ReIDManager] Loading OSNet-x1.0 ...")
        self._model = _build_osnet(device)
        print("[ReIDManager] OSNet ready.")

        # global_id -> list of 512-dim np.ndarray
        self._gallery: dict[int, list] = {}

    # ── Public API ─────────────────────────────────────────────────────────────

    def register(self, global_id: int, frame: np.ndarray, bbox) -> None:
        """
        Extract an OSNet embedding from the crop and add it to the gallery.
        Old embeddings are discarded when the buffer is full (rolling window).
        """
        emb = _extract_embedding(self._model, frame, bbox, self.device)
        if emb is None:
            return

        if global_id not in self._gallery:
            self._gallery[global_id] = []

        buf = self._gallery[global_id]
        buf.append(emb)
        if len(buf) > self.max_embeddings_per_id:
            buf.pop(0)

    def verify(self, gid_i: int, gid_j: int) -> float:
        """
        Return cosine similarity between gallery embeddings of gid_i and gid_j.
        Returns 1.0 (always accept) if either ID has no gallery entry yet.
        """
        emb_i = self._get_mean(gid_i)
        emb_j = self._get_mean(gid_j)

        if emb_i is None or emb_j is None:
            return 1.0  # no appearance info yet → trust geometry

        return _cosine_sim(emb_i, emb_j)

    def find_best_match(
        self,
        query_embedding: np.ndarray,
        candidate_gids: list[int],
    ) -> tuple[int | None, float]:
        """
        Among candidate_gids, return the one whose gallery embedding is
        most similar to query_embedding.

        Returns (best_gid, similarity) or (None, 0.0) if nothing passes
        accept_thresh.
        """
        best_gid = None
        best_sim = self.accept_thresh  # minimum bar

        for gid in candidate_gids:
            emb = self._get_mean(gid)
            if emb is None:
                continue
            sim = _cosine_sim(query_embedding, emb)
            if sim > best_sim:
                best_sim = sim
                best_gid = gid

        return best_gid, best_sim

    def extract_embedding(
        self, frame: np.ndarray, bbox
    ) -> np.ndarray | None:
        """Public wrapper so homography_tracker can get a raw embedding."""
        return _extract_embedding(self._model, frame, bbox, self.device)

    def get_gallery_ids(self) -> list[int]:
        return list(self._gallery.keys())

    # ── Internal ───────────────────────────────────────────────────────────────

    def _get_mean(self, gid: int) -> np.ndarray | None:
        buf = self._gallery.get(gid)
        if not buf:
            return None
        return _mean_embedding(buf)
