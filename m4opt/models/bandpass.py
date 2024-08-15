from astropy import units as u
from astropy.modeling import Model
from astropy.modeling.tabular import Tabular1D
from tynt import FilterGenerator

filter_generator = FilterGenerator()


class Bandpass(Model):
    """Bandpass transmission curve model

    Examples
    --------

    You can create a Bandpass model in several different ways. First, you can
    pass in your own data arrays:

    >>> import numpy as np
    >>> from astropy import units as u
    >>> from m4opt.models.bandpass import Bandpass
    >>> wavelengths = np.linspace(6000, 8000, 100)
    >>> transmission = 0.5 * np.exp(-(wavelengths - 7000)**2 / 500**2)
    >>> bp = Bandpass.from_table(wavelengths * u.Angstrom, transmission)
    >>> bp(7000 * u.Angstrom)
    0.49979598

    You can also look up a wide variety of builtin filters.

    >>> bp = Bandpass.from_name("Generic/Johnson.V")
    >>> bp(5555 * u.Angstrom)
    0.7657823492761526
    """

    n_inputs = 1
    n_outputs = 1
    input_units = {"x": u.Hz}
    return_units = {"y": u.dimensionless_unscaled}
    input_units_equivalencies = {"x": u.spectral()}

    @classmethod
    def from_name(cls, name: str):
        """Look up and return a builtin bandpass model by its name.

        Parameters
        ----------
        name : str
            Name of the filter.
            See :doc:`list of builtin filters <tynt:tynt/filters>`.
        """
        filt = filter_generator.reconstruct(name).table
        filt.input_units_equivalencies = cls.input_units_equivalencies
        return filt

    @classmethod
    def from_table(cls, *args, **kwargs):
        """Create a bandpass model from a lookup table.

        The arguments are the same as those for
        :class:`~astropy.modeling.tabular.Tabular1D`.
        """
        filt = Tabular1D(*args, **kwargs)
        filt.input_units_equivalencies = cls.input_units_equivalencies
        return filt
