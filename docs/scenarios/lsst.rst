LSST
===

The Legacy Survey of Space and Time (LSST) at the Vera C. Rubin Observatory is
scheduled to begin in late 2025. The survey's cadence is being developed through
an iterative, community-driven process. The Rubin Observatory, also known as
the Vera C. Rubin Observatory, features an 8.4-meter aperture and a wide-field
view of about 9.6 square degrees. This ground-based telescope is designed to
conduct a deep, 10-year survey of the Southern sky. The Rubin Observatory will
use six photometric bands, with approximate limiting magnitudes (30 s
exposures, 5$\sigma$) as follows: :math:`sdssu \approx 23.9`, :math:`ps1__g \approx 25.0`,
:math:`ps1__r \approx 24.7`, :math:`ps1__i \approx 24.0`,
:math:`ps1__z \approx 23.3`, and :math:`ps1__y \approx 22.1`.

.. note::

   “Because non-standard observations cannot be guaranteed and no special
   processing will be supported at the start of operations, we do not recommend
   exposure times other than 30 s.” [#lssttooworkshop]_

:Field of View:            :math:`\pi \times 1.75^2 \approx 9.6 \,\mathrm{deg}^2`
:sensitive area:           "To be add"
:Filters:                  :math:`sdssu, ps1__g, ps1__r, ps1__i, ps1__z, ps1__y`
:Location:                 Chile
:Readout Overhead:         2 s  [#lsstdesign]_
:Maximum Slew Velocity:    6.3° s\ :sup:`-1`
:Maximum Slew Acceleration: "To be add"
:Exposure time:            30 s (standard); 120 s or 180 s (possible but not recommended)
:Filter Change:            120 s
:Slew and settle time:     15 s
:Time for one Trigger:     9 hr
:Magnitude limit:          :math:`ps1__r \sim 24.4\,\mathrm{mag}` (5$\sigma, 30 s)
:Maximum airmass:          2.5

In the azimuth direction, the telescope can move at 7 deg s\ :sup:`-1` over a
range of 360°, while in the elevation direction, it moves at 3.5 deg s\ :sup:`-1`
over a range of 90°.

.. [#lssttooworkshop] Source: `Rubin 2024 ToO Workshop Final Report <https://lssttooworkshop.github.io/images/Rubin_2024_ToO_workshop_final_report.pdf>`_
.. [#lsstdesign] Source: `Baseline Design of the LSST Telescope Mount Assembly <https://docushare.lsstcorp.org/docushare/dsweb/Get/Version-34972/BaselineDesignTMASPIE914518.pdf>`_
