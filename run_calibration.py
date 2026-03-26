"""Example script demonstrating the multicamera_calibration module."""
import yaml

from multicamera_calibration import MultiCameraCalibration


def main():
    """Main function to demonstrate the MultiCameraCalibration class."""

    calib_file = "calibration.yaml"
    rig_file = "camera_rig.yaml"
    db_path = "db.db"
    image_path = "assets/images"
    output_path = "output"

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

    # Initialize the database
    multicamera_calib.database_initialization(db_path, image_path)

    # Run the reconstruction process
    multicamera_calib.reconstruct(output_path)


if __name__ == "__main__":
    main()
