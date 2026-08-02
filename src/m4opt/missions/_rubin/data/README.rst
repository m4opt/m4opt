Reproduced Data from lsstCamSim.yaml
====================================

The contents of this directory are reproduced from the following source:
`lsstCamSim.yaml <https://github.com/lsst/obs_lsst/blob/main/policy/lsstCamSim.yaml>`_.

1. Data from `lsstCamSim.yaml` is read to extract relevant detector information.
   We focus only on **science detectors**, excluding wavefront and guide sensors during parsing.
   The resulting 189 detectors are loaded into an `astropy.table.Table` which contains the LSST camera geometry.

2. This table is directly consumed by the `../_camera.py` module to construct the LSST Field of View (FOV),
   based on the detector positions and sizes defined in the YAML file.
