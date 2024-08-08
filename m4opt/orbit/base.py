class Orbit:
    """Base class for an Earth satellite with a specified orbit."""

    @property
    def period(self):
        """The orbital period."""
        raise NotImplementedError

    def __call__(self, time):
        """Get the position and velocity of the satellite.

        Parameters
        ----------
        time : :class:`astropy.time.Time`
            The time of the observation.

        Returns
        -------
        coord : :class:`astropy.coordinates.SkyCoord`
            The coordinates of the satellite in the ITRS frame.

        Notes
        -----
        The orbit propagation is based on the example code at
        https://docs.astropy.org/en/stable/coordinates/satellites.html.

        """
        raise NotImplementedError
