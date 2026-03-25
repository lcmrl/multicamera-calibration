"""Example script demonstrating the multicamera_calibration module."""

from multicamera_calibration import MultiCameraCalibration


def main():
    """Main function to demonstrate the MultiCameraCalibration class."""
    # Initialize the calibration object
    multicamera_calib = MultiCameraCalibration()

    # Path to images directory
    image_path = "assets/images/cam0"

    # Read and print images
    multicamera_calib.read_images(image_path)
    multicamera_calib.print_images()


if __name__ == "__main__":
    main()
