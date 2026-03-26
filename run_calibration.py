"""Example script demonstrating the multicamera_calibration module."""
import yaml

from multicamera_calibration import MultiCameraCalibration


def main():
    """Main function to demonstrate the MultiCameraCalibration class."""

    calib_file = "calibration.yaml"
    rig_file = "camera_rig.yaml"

    # Initialize the calibration object
    multicamera_calib = MultiCameraCalibration()

    # Read calibration and rig configuration files
    try:
        multicamera_calib.read_config(calib_file, rig_file)
        print("Calibration configuration loaded successfully")
        print(f"Calibration data: {multicamera_calib.calib}")
        print(f"Rig configuration: {multicamera_calib.rig}")
    except (FileNotFoundError, yaml.YAMLError) as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
