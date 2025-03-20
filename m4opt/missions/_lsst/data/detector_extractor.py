"""
Extracts science detectors from `lsstCamSim.yaml`, excluding wavefront and guide sensors.
Saves results in CSV and DAT formats. Offsets and pixels are in millimeters (mm).

    LSSTCam : https://github.com/lsst/obs_lsst/blob/main/policy/lsstCamSim.yaml

    plate scale = 0.2 arcsec / pixel
    pixel_size = 0.01 mm per pixel
"""

import os

import numpy as np
import yaml
from astropy.table import Table


def read_yaml(file_path: str) -> dict:
    """
    Reads a YAML file and returns its content as a dictionary.
    Raises a FileNotFoundError if the file does not exist.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"YAML file '{file_path}' not found.")

    with open(file_path, "r") as file:
        return yaml.safe_load(file)


def load_camera_data(camera_yaml: str) -> dict:
    """
    Loads the complete camera configuration from a YAML file.
    """
    return read_yaml(camera_yaml)


def extract_science_detectors(camera_data: dict) -> list:
    """
    Extracts only the science detectors (ignoring wavefront and guide sensors).
    """
    science_detectors = []

    for det_name, det_info in camera_data.get("CCDs", {}).items():
        detector_type = det_info.get("detectorType")

        # Exclude non-science detectors (type != 0) and those containing "SG" or "SW"
        if detector_type != 0 or any(
            substring in det_name for substring in ["SG", "SW"]
        ):
            continue

        detector_offset = np.array(det_info.get("offset", [0, 0]))
        x_pixels, y_pixels = det_info.get("pixelSize", None)
        detector_id = det_info.get("id", "Unknown")
        physical_type = det_info.get("physicalType", "Unknown")
        serial = det_info.get("serial", "Unknown")

        # Append detector information
        science_detectors.append(
            {
                "detector_name": det_name,
                "detector_id": detector_id,
                "x_offset": detector_offset[0],
                "y_offset": detector_offset[1],
                "x_pixels": x_pixels,
                "y_pixels": y_pixels,
                "detector_type": detector_type,
                "physical_type": physical_type,
                "serial": serial,
            }
        )

    return science_detectors


def save_detectors_to_csv(detectors: list, filename: str = "science_detectors.csv"):
    """
    Saves detector information to a CSV file using astropy.table.Table.
    """
    if not detectors:
        print("No detectors found.")
        return

    table = Table(
        rows=detectors,
        names=[
            "detector_name",
            "detector_id",
            "x_offset",
            "y_offset",
            "x_pixels",
            "y_pixels",
            "detector_type",
            "physical_type",
            "serial",
        ],
    )
    table.write(filename, format="csv", overwrite=True)
    print(f"Saved CSV file: {filename}")


def save_detectors_to_dat(detectors: list, filename: str = "science_detectors.dat"):
    """
    Saves detector information to a .dat file using astropy.table.Table.
    """
    if not detectors:
        print("No detectors found.")
        return

    table = Table(
        rows=detectors,
        names=[
            "detector_name",
            "detector_id",
            "x_offset",
            "y_offset",
            "x_pixels",
            "y_pixels",
            "detector_type",
            "physical_type",
            "serial",
        ],
    )
    table.write(filename, format="ascii", overwrite=True)
    print(f"Saved DAT file: {filename}")


camera_yaml = "lsstCamSim.yaml"

try:
    camera_data = load_camera_data(camera_yaml)
    science_detectors = extract_science_detectors(camera_data)

    save_detectors_to_csv(science_detectors, "lsst_science_detectors.csv")
    save_detectors_to_dat(science_detectors, "lsst_science_detectors.dat")
except FileNotFoundError as e:
    print(e)
