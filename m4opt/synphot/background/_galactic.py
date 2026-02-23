import numpy as np
from astropy import units as u
from astropy.modeling.models import Linear1D
from synphot import SourceSpectrum, SpectralElement

from ...utils.typing_extensions import override
from .._extrinsic import ExtrinsicScaleFactor
from ._core import BACKGROUND_SOLID_ANGLE


class GalacticBackgroundScaleFactor(ExtrinsicScaleFactor):
    north_slope: float
    south_slope: float
    north_intercept: float
    south_intercept: float

    def __init__(self, north_slope, south_slope, north_intercept, south_intercept):
        super().__init__()
        self.north_slope = north_slope
        self.south_slope = south_slope
        self.north_intercept = north_intercept
        self.south_intercept = south_intercept

    @override
    def at(self, observer_location, target_coord, obstime):
        csc = 1 / np.sin(target_coord.galactic.b)
        return np.where(
            csc > 0,
            self.north_intercept + self.north_slope * csc,
            self.south_intercept - self.south_slope * csc,
        )


def GalacticBackground():
    """
    Diffuse Galactic ultraviolet background emission.

    Model the diffuse Galactic ultraviolet background using a piecewise
    cosecant model :footcite:`2014ApJS..213...32M`.

    The spectral energy distribution interpolates linearly between the GALEX
    filter wavelengths of 1539 and 2316 Å, and extrapolates linearly outside.

    References
    ----------
    .. footbibliography::

    Examples
    --------
    .. plot::
        :caption: Galactic difffuse emission in the GALEX bands. This should match Figure 7 of :footcite:`2014ApJS..213...32M`.
        :include-source: False

        from astropy.coordinates import EarthLocation, Galactic, SkyCoord
        from astropy.time import Time
        from astropy import units as u
        from matplotlib import pyplot as plt
        from m4opt.synphot.background import GalacticBackground
        from m4opt.synphot.background._core import BACKGROUND_SOLID_ANGLE
        from m4opt.synphot import observing
        import numpy as np


        lat = np.linspace(-90, 90, 100)
        wavelengths = [1539, 2316] * u.angstrom
        spectrum = GalacticBackground()

        target_coord = SkyCoord(0 * u.deg, lat * u.deg, frame=Galactic())
        # Observer location and obstime are arbitrary
        observer_location = EarthLocation(0 * u.m, 0 * u.m, 0 * u.m)
        obstime = Time("2024-01-01")

        with observing(observer_location, target_coord, obstime):
            flux_density = (spectrum(wavelengths[:, np.newaxis]) / BACKGROUND_SOLID_ANGLE).to(
                u.photon * u.cm**-2 * u.s**-1 * u.sr**-1 * u.angstrom**-1
            )

        ax = plt.axes()
        ax.set_xlim(-90, 90)
        ax.set_ylim(0, 5000)
        ax.set_xlabel("Galactic latitude (°)")
        ax.set_ylabel(f"UV background ({flux_density.unit:latex_inline}")
        for y, wavelength in zip(flux_density, wavelengths):
            ax.plot(lat, y, label=f"{wavelength:latex_inline}")
        ax.legend()

    .. plot::
        :caption: Spectral energy distribution model for Galactic diffuse emission, linearly interpolating between and extrapolating through the GALEX FUV and NUV bands.
        :include-source: False

        from astropy.coordinates import EarthLocation, Galactic, SkyCoord
        from astropy.time import Time
        from astropy import units as u
        from matplotlib import pyplot as plt
        from m4opt.synphot.background import GalacticBackground
        from m4opt.synphot.background._core import BACKGROUND_SOLID_ANGLE
        from m4opt.synphot import observing
        import numpy as np
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable
        from matplotlib.lines import Line2D


        galex_wavelengths = [1539, 2316] * u.angstrom
        lats = np.arange(-90, 100, 10) * u.deg
        wavelengths = np.linspace(1000, 3000) * u.angstrom
        spectrum = GalacticBackground()

        target_coord = SkyCoord(0 * u.deg, lats, frame=Galactic())
        # Observer location and obstime are arbitrary
        observer_location = EarthLocation(0 * u.m, 0 * u.m, 0 * u.m)
        obstime = Time("2024-01-01")

        with observing(observer_location, target_coord[:, np.newaxis], obstime):
            flux_density = (spectrum(wavelengths) / BACKGROUND_SOLID_ANGLE).to(
                u.photon * u.cm**-2 * u.s**-1 * u.sr**-1 * u.angstrom**-1
            )

        colormap = ScalarMappable(Normalize(vmin=0, vmax=90), plt.get_cmap("cividis"))
        fig, axs = plt.subplots(
            1, 3, figsize=(8, 3), width_ratios=(3, 3, 1), sharex=True, sharey=True
        )

        axs[0].set_title("Galactic latitude $b$ < 0°")
        axs[1].set_title("Galactic latitude $b$ > 0°")

        for ax, sign in zip(axs, [-1, 1]):
            keep = np.sign(lats) == sign
            for y, lat in zip(flux_density[keep], lats[keep]):
                ax.plot(wavelengths, y, color=colormap.to_rgba(sign * lat.value))
            ax.set_xlabel("Wavelength")

            ax_twin = ax.twiny()
            ax_twin.set_xlim(1000, 3000)
            ax_twin.set_xticks(galex_wavelengths.value)
            ax_twin.set_xticklabels(["FUV", "NUV"])
            ax_twin.grid()

        axs[0].set_ylim(0)
        axs[0].set_xlim(1000, 3000)
        axs[0].set_ylabel(f"Background ({flux_density.unit:latex_inline})")

        axs[-1].set_frame_on(False)
        plt.setp(
            axs[-1].xaxis.get_major_ticks() + axs[-1].yaxis.get_major_ticks(), visible=False
        )
        axs[-1].legend(
            [Line2D([], [], color=colormap.to_rgba(lat)) for lat in range(10, 100, 10)],
            [f"{lat}°" for lat in range(10, 100, 10)],
            mode="expand",
            title="$|b|$",
            loc="upper left",
            borderaxespad=0,
        )

        fig.tight_layout()
    """
    flux_values = [
        GalacticBackgroundScaleFactor(
            north_slope=185.1,
            south_slope=356.3,
            north_intercept=257.5,
            south_intercept=66.7,
        ),
        GalacticBackgroundScaleFactor(
            north_slope=133.2,
            south_slope=401.8,
            north_intercept=93.4,
            south_intercept=-205.5,
        ),
    ]
    wavelengths = ([1539, 2316] * u.angstrom).to(SourceSpectrum._internal_wave_unit)
    (d_wavelength,) = np.diff(wavelengths)
    murthy_photon_flux_density_unit = (
        u.photon * u.cm**-2 * u.s**-1 * u.sr**-1 * u.angstrom**-1
    )
    flux_unit = (1 * murthy_photon_flux_density_unit * BACKGROUND_SOLID_ANGLE).to(
        SourceSpectrum._internal_flux_unit
    )
    lhs, rhs = (
        SourceSpectrum(Linear1D(slope, intercept)) * SpectralElement(flux_value)
        for flux_value, slope, intercept in zip(
            flux_values[::-1],
            ([1, -1] * flux_unit / d_wavelength).to_value(
                SourceSpectrum._internal_flux_unit / SourceSpectrum._internal_wave_unit
            ),
            ([-1, 1] * wavelengths * flux_unit / d_wavelength).to_value(
                SourceSpectrum._internal_flux_unit
            ),
        )
    )
    return lhs + rhs
