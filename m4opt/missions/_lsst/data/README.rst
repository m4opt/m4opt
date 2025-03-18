Reproduced Data from lsstCamSim.yaml
====================================

The contents of this directory are reproduced from the following source:
`lsstCamSim.yaml <https://github.com/lsst/obs_lsst/blob/main/policy/lsstCamSim.yaml>`_.

1. Extracting data from `lsstCamSim.yaml` is computationally expensive.
   To optimize processing, we extract only relevant information, focusing on science detectors
   while excluding wavefront and guide sensors. The remaining 189 detectors are stored in an
   `astropy.table.Table` as a CSV file (`lsst_science_detectors.csv`), which contains the LSST camera geometry.

2. This CSV file is read by `../camera.py` to construct the LSST Field of View (FOV).


3. There are two types of detectors among the 189:
   - **ITL**: Pixel resolution of `[4071, 3999]`
   - **E2V**: Pixel resolution of `[4095, 4003]`

   The LSST features a plate scale of **0.2 arcsec/pixel** with a pixel size of **0.01 mm**.
