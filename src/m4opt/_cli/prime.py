from .core import app


@app.command()
def prime():
    """Download and cache all dependencies that m4opt may use at runtime.

    Under normal operation, m4opt will download and cache various external
    data sources (for example, IERS Earth orientation data and Planck dust
    maps). If you need to run m4opt in an environment with no outbound Internet
    connectivity (for example, some computing clusters), you can run this
    command to download and cache the external data sources immediately.
    """
    from astropy.coordinates import EarthLocation

    from ..synphot.extinction._dust import dust_map

    EarthLocation.get_site_names()
    dust_map()
