""" tracking and smoothing for video face swap."""
from __future__ import annotations

from typing import Optional
import cv2
import numpy as np


class EMASmoother:
    """Exponential moving average over a stream of landmark arrays.
    state <- alpha * new + (1 - alpha) * state
    alpha=1 disables smoothing; default ~0.5 cause a little lag for a visibly steadier swap.
    """

    def __init__(self, alpha: float = 0.5):
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self.state: Optional[np.ndarray] = None

    def reset(self) -> None:
        self.state = None

    def update(self, value: np.ndarray) -> np.ndarray:
        if self.state is None:
            self.state = value.astype(np.float32).copy()
        else:
            self.state = self.alpha * value + (1.0 - self.alpha) * self.state
        return self.state.copy()


class LandmarkTracker:
    """Propagate 2D landmarks between frames with Lucas-Kanade optical flow.
    skip full detection on every frame; re-detect when track() returns None
    """

    def __init__(self, win_size: int = 21, max_level: int = 3, min_tracked: int = 4):
        self.lk_params = dict(
            winSize=(win_size, win_size),
            maxLevel=max_level,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
        )
        self.min_tracked = min_tracked

    def track(
        self, prev_gray: np.ndarray, gray: np.ndarray, prev_points: np.ndarray
    ) -> Optional[np.ndarray]:
        #Return new (5,2) landmark positions, or None if tracking is unreliable
        pts = prev_points.reshape(-1, 1, 2).astype(np.float32)
        new_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None, **self.lk_params)
        if new_pts is None or status is None:
            return None
        if int(status.sum()) < self.min_tracked:
            return None
        return new_pts.reshape(-1, 2)
    
