"""Multi-camera calibration module."""

from pathlib import Path
from typing import List


class MultiCameraCalibration:
    """Class for multi-camera calibration."""

    def __init__(self):
        """Initialize the MultiCameraCalibration object."""
        self.images = []

    def read_images(self, image_path: str) -> List[str]:
        """
        Read all images from a given path.

        Args:
            image_path: Path to the directory containing images.

        Returns:
            List of image file paths.
        """
        path = Path(image_path)
        if not path.exists():
            raise ValueError(f"Path does not exist: {image_path}")

        if not path.is_dir():
            raise ValueError(f"Path is not a directory: {image_path}")

        # Get all image files (jpg, png, jpeg)
        image_extensions = {".jpg", ".jpeg", ".png"}
        image_files = [
            f for f in path.iterdir()
            if f.is_file() and f.suffix.lower() in image_extensions
        ]
        image_files = sorted(image_files)
        self.images = [str(f) for f in image_files]

        return self.images

    def print_images(self):
        """Print all loaded images."""
        print(f"\nLoaded {len(self.images)} images:")
        for idx, img_path in enumerate(self.images, 1):
            print(f"  {idx}. {img_path}")
