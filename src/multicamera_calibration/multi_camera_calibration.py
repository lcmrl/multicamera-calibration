"""Multi-camera calibration module."""

import yaml


class MultiCameraCalibration:
    """Class for multi-camera calibration."""

    def __init__(self):
        """Initialize the MultiCameraCalibration object."""
        self.calib = None
        self.rig = None

    def read_config(self, calib_file: str, rig_file: str):
        """
        Read calibration and rig configuration files.

        Args:
            calib_file: Path to the calibration YAML file.
            rig_file: Path to the rig configuration YAML file.

        Raises:
            FileNotFoundError: If either file does not exist.
            yaml.YAMLError: If there's an error parsing the YAML files.
        """
        # Read calibration file
        try:
            with open(calib_file, 'r') as f:
                self.calib = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Calibration file not found: {calib_file}")
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing calibration file: {e}")

        # Read rig file
        try:
            with open(rig_file, 'r') as f:
                self.rig = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Rig file not found: {rig_file}")
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing rig file: {e}")
    
    def database_initialization(self):
        pass
    
    def run(self):
        self.database_initialization()
