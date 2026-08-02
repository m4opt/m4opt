import synphot
from astroquery.svo_fps import SvoFps


def bandpass_from_svo(filter_id: str) -> synphot.SpectralElement:
    """Look up a filter bandpass from the `SVO Filter Profile Service <http://svo2.cab.inta-csic.es/theory/fps/>`_.

    Parameters
    ----------
    filter_id
        The name of the filter.

    Returns
    -------
    :
        The filter transmission curve.

    Examples
    --------
    .. plot::

        from matplotlib import pyplot as plt
        from m4opt.synphot import bandpass_from_svo
        import numpy as np
        from astropy import units as u
        from astropy.visualization import quantity_support

        filter_id = "Palomar/ZTF.r"
        bandpass = bandpass_from_svo(filter_id)
        wavelength = np.arange(5000, 8000) * u.angstrom
        transmission = bandpass(wavelength)

        with quantity_support():
            ax = plt.axes()
            ax.plot(wavelength, transmission)
            ax.set_title(filter_id)
    """
    table = SvoFps.get_transmission_data(filter_id)
    return synphot.SpectralElement(
        synphot.Empirical1D,
        points=table["Wavelength"],
        lookup_table=table["Transmission"],
    )
