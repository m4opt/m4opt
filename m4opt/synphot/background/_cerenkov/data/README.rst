The `../_energy_loss.py` and `../_refraction_index.py` scripts are a Python port of the MATLAB `Cerenkov` function from the MAATv2 package:
https://www.mathworks.com/matlabcentral/fileexchange/128984-astropack-maatv2


- ``suprasil_2a_refractive_index.csv``: Refractive index of Suprasil 2A (synthetic fused silica) as a function of wavelength.
- ``suprasil_2a_transmission.csv``: Transmission of Suprasil 2A (per 1 cm) as a function of wavelength.

These scripts include the following data files, derived from the same MAATv2 package:

For reference, see the original MATLAB repository:
https://github.com/EranOfek/AstroPack

The AE8 trapped electron flux model from the radiation belts is included.
For more details, see the NASA AE8/AP8 model as implemented in IRBEM:
 `NASA AE8/AP8, IRBEM <irbem:api/radiation_models>`

AE8 source repository: https://github.com/m4opt/aep8
