"""video face swapping: tracking, smoothing, and file IO around Faceswapper."""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Iterable, Iterator, Optional
import cv2
import numpy as np

# functions
from .blend import blend_back
from .data.alignment import FaceAligner
from .inference import FaceSwapper, _bbox_area
from .tracking import EMASmoother, LandmarkTracker

log = logging.getLogger(__name__)


class VideoFaceSwapper:
    def __init__(self, swapper: FaceSwapper, working_size: int = 128, detect_every: int = 5, smoothing_alpha: float = 0.5, blend_method: str = "alpha", det_size: int = 256,):
        self.swapper = swapper
        self.working_size = working_size
        self.detect_every = detect_every
        self.blend_method = blend_method
        # aligner at video working size, reusing the swapper detector.
        self.aligner = FaceAligner(detector=swapper.aligner.detector, output_size=working_size)
        self.tracker = LandmarkTracker()
        self.smoother = EMASmoother(alpha=smoothing_alpha)
        self._identity = None
        self._reset_state()

    def _reset_state(self) -> None:
        self._frame_idx = 0
        self._prev_gray: Optional[np.ndarray] = None
        self._prev_landmarks: Optional[np.ndarray] = None
        self.smoother.reset()

    def set_source(self, source_bgr: np.ndarray) -> bool:
        #Compute and cache the source identity once. Return False if no face
        self._identity = self.swapper.embed_identity(source_bgr)
        self._reset_state()
        return self._identity is not None

    def _landmarks_for_frame(self, frame_bgr: np.ndarray, gray: np.ndarray) -> Optional[np.ndarray]:
        # Detect every N frames, falls back to detection if tracking is lost. return None if no face
        need_detect = (self._frame_idx % self.detect_every == 0) or self._prev_landmarks is None
        if not need_detect and self._prev_gray is not None:
            tracked = self.tracker.track(self._prev_gray, gray, self._prev_landmarks)
            if tracked is not None:
                return tracked
            # tracking lost -> fall through to detection

        faces = self.aligner.detector.detect(frame_bgr)
        if not faces:
            return None
        largest = max(faces, key=lambda f: _bbox_area(f.get("bbox")))
        return np.asarray(largest["kps"], dtype=np.float32)

    def process_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        #Swap one frame. Return it unchanged if no source/target face
        if self._identity is None:
            raise RuntimeError("call set_source() before processing frames")

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        landmarks = self._landmarks_for_frame(frame_bgr, gray)
        if landmarks is None:
            self._frame_idx += 1
            self._prev_gray = gray
            return frame_bgr

        smoothed = self.smoother.update(landmarks)
        aligned = self.aligner.align_from_landmarks(frame_bgr, smoothed)
        swapped_aligned = self.swapper._swap_one(aligned.image, self._identity)
        result = blend_back(frame_bgr, swapped_aligned, aligned, method=self.blend_method)

        self._prev_landmarks = landmarks
        self._prev_gray = gray
        self._frame_idx += 1
        return result

    def process_frames(self, frames: Iterable[np.ndarray]) -> Iterator[np.ndarray]:
        for frame in frames:
            yield self.process_frame(frame)


def process_video(swapper: FaceSwapper, source_bgr: np.ndarray,in_path: str | Path, out_path: str | Path, working_size: int = 128,
                    detect_every: int = 5, smoothing_alpha: float = 0.5, blend_method: str = "alpha", det_size: int = 256,) -> dict:
    # swap a source identity into every frame of a video file

    vs = VideoFaceSwapper(swapper, working_size, detect_every, smoothing_alpha, blend_method)
    if not vs.set_source(source_bgr):
        raise ValueError("no face detected in the source image")

    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open video: {in_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    n, t0 = 0, time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(vs.process_frame(frame))
        n += 1
    cap.release()
    writer.release()

    elapsed = time.time() - t0
    achieved_fps = n / elapsed if elapsed > 0 else 0.0
    log.info("processed %d frames in %.1fs -> %.1f FPS (source fps %.1f)", n, elapsed, achieved_fps, fps)
    return {"frames": n, "seconds": elapsed, "processing_fps": achieved_fps, "source_fps": fps}


