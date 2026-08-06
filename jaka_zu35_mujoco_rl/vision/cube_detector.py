"""Simulated color-based cube detector for the PandaPickEnv.

This module is a placeholder for a real YOLO / instance-segmentation vision
pipeline. It renders a simulated RGB image from the MuJoCo overhead camera,
thresholds the red cube by color, and back-projects the 2D image centre to a
3D world position using the known table height and camera intrinsics.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


class CubeDetector:
    """Detect a red cube in a simulated RGB image and estimate its 3D pose.

    Parameters
    ----------
    camera_position : np.ndarray
        (3,) world-frame position of the camera optical centre.
    camera_rotation : np.ndarray
        (3, 3) rotation matrix from camera frame to world frame. MuJoCo's
        camera frame is x-right, y-down, z-forward (into the scene).
    table_height : float
        Z-coordinate of the table surface in world frame (metres).
    cube_half_size : float
        Half side length of the cube (metres). Used only to validate the
        bounding box aspect ratio.
    fovy : float
        Vertical field of view of the camera in degrees.
    image_size : tuple[int, int]
        (height, width) of the rendered image.
    noise_std : float
        Standard deviation of Gaussian noise added to the estimated 3D position
        to simulate real detector uncertainty.
    """

    def __init__(
        self,
        camera_position: np.ndarray,
        camera_rotation: np.ndarray,
        table_height: float = 0.0,
        cube_half_size: float = 0.025,
        fovy: float = 45.0,
        image_size: tuple[int, int] = (480, 640),
        noise_std: float = 0.005,
    ) -> None:
        self.camera_position = np.asarray(camera_position, dtype=np.float64)
        self.camera_rotation = np.asarray(camera_rotation, dtype=np.float64)
        self.table_height = table_height
        self.cube_half_size = cube_half_size
        self.fovy = np.radians(fovy)
        self.image_size = image_size
        self.noise_std = noise_std

        self._camera_matrix = self._build_camera_matrix()

    def _build_camera_matrix(self) -> np.ndarray:
        """Build an OpenCV-style camera matrix from fovy and image size."""
        height, width = self.image_size
        fy = 0.5 * height / np.tan(self.fovy / 2.0)
        fx = fy  # square pixels
        cx = width / 2.0
        cy = height / 2.0
        return np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def detect(self, rgb_image: np.ndarray, rng: np.random.Generator | None = None) -> dict[str, Any]:
        """Run the cube detector on a rendered RGB image.

        Returns
        -------
        dict with keys:
            - detected: bool
            - position: np.ndarray | None, estimated 3D world position
            - confidence: float in [0, 1]
            - bbox: tuple[int, int, int, int] | None, (x, y, w, h) in image space
        """
        if rgb_image is None or rgb_image.size == 0:
            return {"detected": False, "position": None, "confidence": 0.0, "bbox": None}

        # Convert to HSV and threshold red. Red wraps around hue 0/180.
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
        mask1 = cv2.inRange(hsv, np.array([0, 120, 70]), np.array([10, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([170, 120, 70]), np.array([180, 255, 255]))
        mask = cv2.bitwise_or(mask1, mask2)

        # Clean up noise.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return {"detected": False, "position": None, "confidence": 0.0, "bbox": None}

        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        area = w * h
        if area < 100:
            return {"detected": False, "position": None, "confidence": 0.0, "bbox": None}

        # Aspect ratio sanity check: cube should be roughly square in image.
        aspect_ratio = float(min(w, h)) / max(w, h)
        if aspect_ratio < 0.3:
            return {"detected": False, "position": None, "confidence": 0.0, "bbox": None}

        confidence = min(1.0, area / 5000.0)

        # Image centre of the cube.
        u = x + w / 2.0
        v = y + h / 2.0
        pixel = np.array([u, v], dtype=np.float64)

        position = self._back_project(pixel)

        if rng is not None and self.noise_std > 0.0:
            position += rng.normal(0.0, self.noise_std, size=3)

        return {
            "detected": True,
            "position": position,
            "confidence": confidence,
            "bbox": (int(x), int(y), int(w), int(h)),
        }

    def _back_project(self, pixel: np.ndarray) -> np.ndarray:
        """Back-project an image pixel to the table plane z = table_height."""
        uv1 = np.array([pixel[0], pixel[1], 1.0], dtype=np.float64)
        ray_cam = np.linalg.solve(self._camera_matrix, uv1)
        ray_world = self.camera_rotation @ ray_cam
        if abs(ray_world[2]) < 1e-9:
            ray_world[2] = 1e-9
        scale = (self.table_height - self.camera_position[2]) / ray_world[2]
        position = self.camera_position + scale * ray_world
        return position
