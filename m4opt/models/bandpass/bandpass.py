from astropy.modeling.models import Tabular1D
from astropy import units as u
from astropy.units.physical import get_physical_type

from .core import BaseBandpass

# TODO: remove check below when tynt is released
# and add tynt to required dependencies
try:
    from tynt import FilterGenerator
    tynt_filters = FilterGenerator().available_filters()
except ImportError:
    tynt_filters = None


# TODO: turn on tox tests for tynt when that module is released
class Bandpass:
    """
    Bandpass models

    This is the generic model for telescope bandpasses or filters. These
    models take in wavelength or frequency as input, and return the
    filter transmittance as output.

    Examples
    --------

    You can create a Bandpass model in several different ways. First, you can
    pass in your own data arrays:

    >>> import numpy as np
    >>> from astropy import units as u
    >>> from m4opt.models.bandpass import Bandpass
    >>> waves = np.linspace(6000, 8000, 100)
    >>> transmission = 0.5 * np.exp(-(waves-7000)**2/500**2)
    >>> bp = Bandpass.from_table(waves* u.Angstrom, transmission)
    >>> bp(7000*u.Angstrom)
    <Quantity 0.49979598>

    Alternatively, if you have the optional module `tynt`, we can instantiate
    from `tynt`'s list of supported filters.

    --->from m4opt.models.bandpass import tynt_filters
    --->filter_name = tynt_filters[11]
    --->print(filter_name)
    'Generic/Johnson.V'
    --->bp_V = Bandpass.from_name(filter_name)
    --->bp_V(5555*u.Angstrom)
    <Quantity 0.76578238>

    Notes
    -----

    You can instantiate a generic Bandpass model from an `astropy` `1D
    model subclass`_. However, in order to be compatible
    with other `m4opt.models`, your model inputs must have spectral units and
    the output must be dimensionless.

    .. _`1D model subclass`: https://docs.astropy.org/en/stable/modeling/predef_models1D.html # noqa

    For example,

    ---> from astropy import units as u
    ---> gauss1d = Gaussian1D(amplitude=0.5, mean=6000 * u.Angstrom, stddev=500*u.Angstrom)
    ---> gauss1d.result.input_units_equivalencies = {'x': u.spectral()}

    will work with other `m4opt` models, but

    ---> gauss1d = Gaussian1D(amplitude=0.5, mean=6000 * u.erg, stddev=500 * u.erg)

    will not.
    """

    @classmethod
    def from_table(cls, points, lookup_table):
        """
        Initializes bandpass from user-defined table.

        points : 1D array of `astropy.Quantity` (wavelength or frequency)
        lookup_table : 1D array of bandpass response
        """

        unit_type = get_physical_type(points)
        if unit_type != "length" and unit_type != "frequency":
            raise AttributeError("Input array points does not have " +
                                 "appropriate wavelength or frequency " +
                                 "units.")

        result = Tabular1D(points, lookup_table*u.dimensionless_unscaled)
        result.input_units_equivalencies = (BaseBandpass.
                                            input_units_equivalencies)
        return result

    @classmethod
    def from_name(cls, filter_name):
        """
        Returns a bandpass for a filter specified by `filter_name`, using
        parameterizations from the `tynt <https://github.com/bmorris3/tynt>`_
        package.
        """
        if tynt_filters is not None:
            if filter_name not in tynt_filters:
                raise ValueError("Invalid filter {0}. ".format(filter_name) +
                                 "See 'm4opt.bandpass.tynt_filters' for " +
                                 "available options.")

            filter = FilterGenerator().reconstruct(filter_name)
            result = Tabular1D(filter.wavelength,
                               filter.transmittance*u.dimensionless_unscaled)
            result.input_units_equivalencies = (BaseBandpass.
                                                input_units_equivalencies)
            return result

        else:
            raise ImportError("optional dependency 'tynt' is not installed. "
                              "To install it, do `pip install "
                              "git+https://github.com/bmorris3/tynt.git@master`") # noqa
