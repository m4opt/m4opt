"""

Photon Source: Objects to hold source spectrum and calculate scale factors 
for photon rate evaluation

"""

# dependencies
import numpy as np
from astropy.coordinates import GeocentricTrueEcliptic, get_sun, SkyCoord
from astropy.modeling import Model, CompoundModel
from synphot.spectrum import SourceSpectrum
from astropy.table import QTable
import astropy.units as u
from synphot import Empirical1D, ConstFlux1D, GaussianFlux1D, PowerLawFlux1D
from synphot.units import PHOTLAM
from scipy.interpolate import RegularGridInterpolator

# all backgrounds here based on dorado-sensitivity/backgrounds
# source: https://github.com/nasa/dorado-sensitivity/blob/main/dorado/sensitivity/backgrounds.py

# TODO: should we change default behavior of GalacticBackground so that it returns one object
# instead of a CompoundModel?
# TODO: Should we refactor into a base PhotonSource, and have other backgrounds/sources inherit from this?

class PhotonSource(Model):
    """
    simple Photon Source Object

    Parameters
    ----------
    name : string
        Name of object. Used for accessing internal parameters in compound background model.
    spectrum : ``synphot.SourceSpectrum`` 
        Background Spectrum used for calculating source counts
    scale_factor : float
        Scaling for spectrum flux
    """

    # definitions required for astropy.Model
    n_inputs = 1 # wavelength
    n_outputs = 1 # flux

    def __init__(self, name="target", spectrum = None):
        self.spectrum = spectrum
        super().__init__()

        #goes after __init__() because reasons(?)
        self.name = name

    def valid(self):
        if self.spectrum is None:
            return False
        else:
            return True

    def set_spectrum(self, spectrum):
        self.spectrum = spectrum

    def set_scale_factor(self, scale_factor):
        self.scale_factor = scale_factor

    def evaluate(self, wavelength):
        if self.valid():
            return self.spectrum(wavelength)*self.scale_factor
        else:
            return RuntimeError("spectrum is not defined")

# Zodiacal Background
class ZodiacalBackground(Model):
    """
    Zodiacal Background Object

    Parameters
    ----------
    name : string
        Name of object. Used for accessing internal parameters in compound background model.
    spectrum : ``synphot.SourceSpectrum`` 
        Background Spectrum used for calculating source counts
    scale_factor : float
        Scale factor for zodical light; expressed as a ratio between
        desired value and set spectrum.
        See ``set_zodiacal_light_scale()`` for default estimation function,
        or ``set_scale_factor()`` for setting directly from user-provided value. 

    Notes
    -----
    ``default()`` initializes spectrum to the "high" zodiacal spectrum from Table 6.4 of 
    the STIS Instrument Manual. In the default case, ``set_zodiacal_light_scale()`` will 
    provide correct scale factor based on sky position and time.
    
    Otherwise, if user-defined spectrum is provided, ``set_scale_factor()`` should instead
    be used to directly set the appropriate scale factor.

    Examples
    --------
    TBD

    References
    ----------
    https://hst-docs.stsci.edu/stisihb/chapter-6-exposure-time-calculations/6-5-detector-and-sky-backgrounds
    """

    # definitions required for astropy.Model
    n_inputs = 1 # wavelength
    n_outputs = 1 # flux

    def __init__(self, name="zodical", spectrum = None, scale_fac = 1.):
        self.spectrum = spectrum
        self.scale_factor = scale_fac
        self.zodical_angular_dependence = None
        super().__init__()

        #goes after __init__() because reasons(?)
        self.name = name

    @classmethod
    def from_file(cls, path, name="zodical"):
        table = QTable.read(path)
        
        spectrum = SourceSpectrum(Empirical1D,
            points=table['wavelength'],
            lookup_table=table['surface_brightness'] * u.arcsec**2)

        return cls(name=name, spectrum=spectrum)
    
    @classmethod
    def default(cls):
        # TODO: change to use importlib.resources
        #with resources.path(data, 'stis_zodi_high.ecsv') as p:
            #table = QTable.read(p)
    
        table = QTable.read("../data/stis_zodi_high.ecsv")
        
        # "High" zodiacal light spectrum, normalized to 1 square arcsecond.
        high_zodiacal = SourceSpectrum(Empirical1D,
                                        points=table['wavelength'],
                                        lookup_table=table['surface_brightness'] * u.arcsec**2)
        
        return cls(spectrum=high_zodiacal)


    @classmethod
    def from_amplitude(cls, amp, name):
        spectrum = SourceSpectrum(ConstFlux1D, 
            amplitude= amp * PHOTLAM * u.steradian**-1 * u.arcsec**2)
        return cls(name=name, spectrum=spectrum)

    def valid(self):
        if self.spectrum is None:
            return False
        else:
            return True

    def set_spectrum(self, spectrum):
        self.spectrum = spectrum

    def set_scale_factor(self, scale_fac):
        self.scale_factor = scale_fac

    def evaluate(self, wavelength):
        if self.valid():
            return self.spectrum(wavelength)*self.scale_factor
        else:
            return RuntimeError("spectrum is not defined")


    def set_zodiacal_light_scale(self, coord, time):
        """
        Taken from dorado-sensitivity/backgrounds.py

        Get the scale factor for zodiacal light compared to "high" conditions.
        Estimate the ratio between the zodiacal light at a specific sky position
        and time, and its "high" value, by interpolating Table 6.2 of the STIS
        Instrument Manual.
        Parameters
        ----------
        coord : astropy.coordinates.SkyCoord
            The coordinates of the object under observation. If the coordinates do
            not specify a distance, then the object is assumed to be a fixed star
            at infinite distance for the purpose of calculating its helioecliptic
            position.
        time : astropy.time.Time
            The time of the observation.
        Returns
        -------
        float
            The zodiacal light scale factor.
        References
        ----------
        https://hst-docs.stsci.edu/stisihb/chapter-6-exposure-time-calculations/6-5-detector-and-sky-backgrounds
        """

        if self.zodical_angular_dependence is None:
            self.zodical_angular_dependence = self.get_zodi_angular_interp()

        obj = SkyCoord(coord).transform_to(GeocentricTrueEcliptic(equinox=time))
        sun = get_sun(time).transform_to(GeocentricTrueEcliptic(equinox=time))

        # Wrap angles and look up in table
        lat = np.abs(obj.lat.deg)
        lon = np.abs((obj.lon - sun.lon).wrap_at(180 * u.deg).deg)
        result = self.zodi_angular_dependence(np.stack((lon, lat), axis=-1))

        # When interp2d encounters infinities, it returns nan. Fix that up here.
        result = np.where(np.isnan(result), -np.inf, result)

        # Fix up shape
        if obj.isscalar:
            result = result.item()

        result -= self.zodi_angular_dependence([180, 0]).item()
        return u.mag(1).to_physical(result)

    @staticmethod
    def set_zodi_angular_interp():
        # Zodiacal light angular dependence from Table 16 of Leinert et al. (2017)
        # https://doi.org/10.1051/aas:1998105.
        
        # TODO: replace hardcoded path with importlib.resources
        #with resources.path(data, 'leinert_zodi.txt') as p:
            #table = np.loadtxt(p)
        table = np.loadtxt("../data/leinert_zodi.txt")
        lat = table[0, 1:]
        lon = table[1:, 0]  
        s10 = table[1:, 1:]

        # The table only extends up to a latitude of 75°. According to the paper,
        # "Towards the ecliptic pole, the brightness as given above is 60 ± 3 S10."
        lat = np.append(lat, 90)
        s10 = np.append(s10, np.tile(60.0, (len(lon), 1)), axis=1)

        # The table is in units of S10: the number of 10th magnitude stars per
        # square degree. Convert to magnitude per square arcsecond.
        sb = 10 - 2.5 * np.log10(s10 / 60**4)

        return RegularGridInterpolator([lon, lat], sb)


# Airglow Background
class AirglowBackground(Model):
    """
    Airglow Background Object

    Parameters
    ----------
    name : string
        Name of object. Used for accessing internal parameters in compound background model.
    spectrum : ``synphot.SourceSpectrum`` 
        Background Spectrum used for calculating source counts
        If not defined (i.e. 'None'), default airglow spectrum will be used.
    scale_factor : float
        Scale factor for airglow light; expressed as a ratio between
        desired value and set spectrum.
        See ``set_airglow_scale()`` for default estimation function,
        or ``set_scale_factor()`` for setting directly from user-provided value. 

    Notes
    -----
    ``default()`` initializes to the daytime airglow spectrum.
    In this case, ``set_airglow_scale()`` will provide correct scale factor based
    on whether it is daytime or not.
    Otherwise, if user-defined spectrum is provided, ``set_scale_factor()`` should instead
    be used to directly set the appropriate scale factor.

    Examples
    --------
    TBD

    """
    # definitions required for astropy.Model
    n_inputs = 1 # wavelength
    n_outputs = 1 # flux

    def __init__(self, name="airglow", spectrum = None, scale_fac = 1.):
        
        self.spectrum = spectrum
        self.scale_factor = scale_fac
        super().__init__()
        #goes after __init__() because reasons(?)
        self.name = name
        
    @classmethod
    def default(cls):
        """Airglow spectrum in daytime, normalized to 1 square arcsecond."""
        default_airglow = SourceSpectrum(GaussianFlux1D,
                            mean=2471 * u.angstrom,
                            fwhm=0.023 * u.angstrom,
                            total_flux=1.5e-15 * u.erg * u.s**-1 * u.cm**-2)
        return cls(spectrum=default_airglow)

    @classmethod
    def from_file(cls, path, name):
        table = QTable.read(path)
        
        spectrum = SourceSpectrum(Empirical1D,
            points=table['wavelength'],
            lookup_table=table['surface_brightness'] * u.arcsec**2)

        return cls(name=name, spectrum=spectrum)
    
    @classmethod
    def from_amplitude(cls, amp, name):
        spectrum = SourceSpectrum(ConstFlux1D, 
            amplitude= amp * PHOTLAM * u.steradian**-1 * u.arcsec**2)
        return cls(name=name, spectrum=spectrum)

    def valid(self):
        if self.spectrum is None:
            return False
        else:
            return True

    def set_spectrum(self, spectrum):
        self.spectrum = spectrum

    def set_scale_factor(self, scale_fac):
        self.scale_factor = scale_fac

    def evaluate(self, wavelength):
        if self.valid():
            return self.spectrum(wavelength)*self.scale_factor
        else:
            return RuntimeError("spectrum is not defined")

    def set_airglow_scale(self, night):
        """Get the scale factor for the airglow spectrum.
        Estimate the ratio between the airglow and its daytime value.
        Parameters
        ----------
        night : bool
            Use the "low" value if True, or the "average" value if False.
        Returns
        -------
        scale : float
            The airglow scale factor.
        References
        ----------
        https://hst-docs.stsci.edu/stisihb/chapter-6-exposure-time-calculations/6-5-detector-and-sky-backgrounds
        """
        self.scale_factor = np.where(night, 1e-2, 1)


# Galactic light background
class GalacticBackground(Model):
    """
    Galactic Background Object

    Parameters
    ----------
    name : string
        Name of object. Used for accessing internal parameters in compound background model.
    spectrum : ``synphot.SourceSpectrum`` 
        Background Spectrum used for calculating source counts
        If not defined (i.e. 'None'), default galactic spectrum will be used.
    scale_factor : float
        Scale factor for galactic light; expressed as a ratio between
        desired value and set spectrum.
        See ``set_galactic_scale()`` for default estimation function,
        or ``set_scale_factor()`` for setting directly from user-provided value. 

    Notes
    -----
    ``default()`` estimatea the Galactic diffuse emission based on the cosecant fits from
    Murthy (2014). In this case, there are two combined spectra and two scale factors. 
    ``set_galactic_scale()`` will provide correct scale factors based on galactic coordinate.

    Otherwise, if user-defined spectrum(s) is provided, ``set_scale_factor()`` should instead
    be used to directly set the appropriate scale factor(s).

    Examples
    --------
    TBD

    """
    # definitions required for astropy.Model
    n_inputs = 1 # wavelength
    n_outputs = 1 # flux

    def __init__(self, name="galactic", spectrum = None, default=0, scale_fac = 1.):
 
        self.spectrum = spectrum
        self.scale_factor = scale_fac
        self.default = default # used only in default case
        super().__init__()
        
        #goes after __init__() because reasons(?)
        self.name = name
        
    @classmethod
    def default(cls):
        galactic1 = SourceSpectrum(ConstFlux1D,
                                amplitude=1 * PHOTLAM * u.steradian**-1 * u.arcsec**2)


        galactic2 = SourceSpectrum(PowerLawFlux1D,
                                x_0=1528*u.angstrom,
                                alpha=-1,
                                amplitude=1 * PHOTLAM * u.steradian**-1 * u.arcsec**2)
        

        # technically returns CompoundModel
        return (cls(name='galactic1', spectrum=galactic1, default=1) + 
                cls(name='galactic2', spectrum=galactic2, default=2))

    @classmethod
    def from_file(cls, path, name):
        table = QTable.read(path)
        
        spectrum = SourceSpectrum(Empirical1D,
            points=table['wavelength'],
            lookup_table=table['surface_brightness'] * u.arcsec**2)

        return cls(name=name, spectrum=spectrum)
    
    @classmethod
    def from_amplitude(cls, amp, name):
        spectrum = SourceSpectrum(ConstFlux1D, 
            amplitude= amp * PHOTLAM * u.steradian**-1 * u.arcsec**2)
        return cls(name=name, spectrum=spectrum)    

    def valid(self):
        if self.spectrum is None:
            return False
        else:
            return True

    def set_spectrum(self, spectrum):
        self.spectrum = spectrum

    def set_scale_factor(self, scale_fac):
        self.scale_factor = scale_fac

    def evaluate(self, wavelength):
        if self.valid():
            return self.spectrum(wavelength)*self.scale_factor
        else:
            return RuntimeError("spectrum is not defined")
    
    def get_galactic_scales(self, coord):
        """Get the Galactic diffuse emission, normalized to 1 square arcsecond.
        Estimate the Galactic diffuse emission based on the cosecant fits from
        Murthy (2014).
        Parameters
        ----------
        coord : astropy.coordinates.SkyCoord
            The coordinates of the object under observation.
        Returns
        -------
        synphot.SourceSpectrum
            The Galactic diffuse emission spectrum, normalized to 1 square
            arcsecond.
        References
        ----------
        https://doi.org/10.3847/1538-4357/aabcb9
        """

        # NOTE: only works with default GalacticBackground, which returns compound model
        assert self.default == 1 or self.default == 2, "Function only works with objects instantiated from GalacticBackground.default()"

        b = SkyCoord(coord).galactic.b
        csc = 1 / np.sin(b)
        pos = (csc > 0)

        # Constants from Murthy (2014) Table 4.
        # Note that slopes for the Southern hemisphere have been negated to cancel
        # the minus sign in the Galactic latitude.
        fuv_a = np.where(pos, 93.4, -205.5)
        fuv_b = np.where(pos, 133.2, -401.8)

        fuv = fuv_a + fuv_b * csc

        if self.default == 1:
            return fuv
        else:
            nuv_a = np.where(pos, 257.5, 66.7)
            nuv_b = np.where(pos, 185.1, -356.3)
            nuv = nuv_a + nuv_b * csc
            
            # GALEX filter effective wavelengths in angstroms from
            # http://www.galex.caltech.edu/researcher/techdoc-ch1.html#3
            fuv_wave = 1528
            nuv_wave = 2271

            return (nuv - fuv) / (nuv_wave - fuv_wave)