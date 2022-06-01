from astropy.modeling import Model
from astropy.modeling.models import Tabular1D
from astropy import units as u
from astropy.units.physical import get_physical_type

from .core import Background

# TODO: remove check below when tynt is released
# and add tynt to required dependencies
try:
    from tynt import FilterGenerator
    has_tynt = True
    tynt_filters = FilterGenerator().available_filters()
except ImportError:
    has_tynt = False
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

    Finally, you can instantiate a Bandpass model from an
    `astropy.modeling.models` subclass and its associated input arguments.
    Note, however, Bandpass objects require wavelength inputs and dimensionless
    outputs.

    >>> from astropy.modeling.models import Box1D, Gaussian1D
    >>> bp_box = Bandpass.from_model(Box1D, amplitude=0.5, \
        x_0 = 600*u.nm, width = 200*u.nm))
    >>> bp_box(5500*u.Angstrom)
    <Quantity 0.5>

    >>> bp_gauss = Bandpass.from_model(Gaussian1D, amplitude=0.5, \
        mean=6000*u.Angstrom, stddev=500*u.Angstrom)
    >>> bp_gauss(5500*u.Angstrom)
    <Quantity 0.30326533>
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
        result.input_units_equivalencies = Background.input_units_equivalencies
        return result

    @classmethod
    def from_name(cls, filter_name):
        """
        Initializes bandpass from tynt package.
        """
        if has_tynt:
            if filter_name not in tynt_filters:
                raise ValueError("Invalid filter {0}. ".format(filter_name) +
                                 "See 'm4opt.bandpass.tynt_filters' for " +
                                 "available options.")

            filter = FilterGenerator().reconstruct(filter_name)
            result = Tabular1D(filter.wavelength,
                               filter.transmittance*u.dimensionless_unscaled)
            result.input_units_equivalencies = (Background.
                                                input_units_equivalencies)
            return result

        else:
            print("optional dependency 'tynt' is not installed. " +
                  "Will return 'None'")
            return None

    @classmethod
    def from_model(cls, modelclass, **kwargs):
        """
        Initializes bandpass from `astropy.modeling.Model` class
        given appropriate arguments.

        This function uses code from `synphot.spectrum.SourceSpectrum' under
        the terms of the synphot BSD 3-Clause License (see m4opt/licenses
        directory).

        """

        if not issubclass(modelclass, Model):
            raise ValueError("modelclass argument {0} ".format(modelclass) +
                             " is not a valid astropy Model class")

        modelname = modelclass.__name__
        if modelname not in cls._model_param_dict:
            raise ValueError("Model {0}".format(modelname) +
                             " is not supported.")

        modargs = {}
        for argname, val in kwargs.items():
            if argname in cls._model_param_dict[modelname]:
                argtype_req = cls._model_param_dict[modelname][argname]
                argtype = get_physical_type(val)

                # check for proper input units
                if argtype_req == 'wave':
                    if argtype != "length":
                        raise ValueError("Input argument {0}".format(argname) +
                                         " does not have length units.")
                    else:
                        modargs[argname] = val
                else:
                    # should be dimensionless
                    if argtype != "dimensionless":
                        raise ValueError("Input argument {0}".format(argname) +
                                         " should be dimensionless.")
                    else:
                        modargs[argname] = val * u.dimensionless_unscaled
            else:
                modargs[argname] = val * u.dimensionless_unscaled

        result = modelclass(**modargs)
        result.input_units_equivalencies = Background.input_units_equivalencies
        return result

    # taken and modified from `synphot.spectrum`
    # under terms of BSD 3-Clause license
    # (see m4opt/licenses directory)
    _model_param_dict = {
        'Box1D': {'amplitude': 'dimensionless', 'x_0': 'wave',
                  'width': 'wave'},
        'BrokenPowerLaw1D': {
            'amplitude': 'dimensionless', 'x_break': 'wave',
            'alpha_1': 'dimensionless', 'alpha_2': 'dimensionless'},
        'Const1D': {'amplitude': 'dimensionless'},
        'ConstFlux1D': {'amplitude': 'dimensionless'},
        'Empirical1D': {'points': 'wave',
                        'lookup_table': 'dimensionless'},
        'ExtinctionModel1D': {'points': 'wave',
                              'lookup_table': 'dimensionless'},
        'ExponentialCutoffPowerLaw1D': {
            'amplitude': 'dimensionless', 'x_0': 'wave',
            'x_cutoff': 'wave', 'alpha': 'dimensionless'},
        'Gaussian1D': {'amplitude': 'dimensionless', 'mean': 'wave',
                       'stddev': 'wave'},
        'GaussianAbsorption1D': {
            'amplitude': 'dimensionless', 'mean': 'wave',
            'stddev': 'wave'},
        'LogParabola1D': {
            'amplitude': 'dimensionless', 'x_0': 'wave',
            'alpha': 'dimensionless',
            'beta': 'dimensionless'},
        'Lorentz1D': {'amplitude': 'dimensionless', 'x_0': 'wave',
                      'fwhm': 'wave'},
        'RickerWavelet1D': {
            'amplitude': 'dimensionless', 'x_0': 'wave',
            'sigma': 'wave'},
        'MexicanHat1D': {
            'amplitude': 'dimensionless', 'x_0': 'wave',
            'sigma': 'wave'},
        'PowerLaw1D': {
            'amplitude': 'dimensionless', 'x_0': 'wave',
            'alpha': 'dimensionless'},
        'Trapezoid1D': {
            'amplitude': 'dimensionless', 'x_0': 'wave',
            'width': 'wave',
            'slope': 'dimensionless'}}
