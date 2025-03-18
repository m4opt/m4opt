from typing import Annotated

import healpy as hp
import numpy as np
import synphot
import typer
from astropy import units as u
from astropy.coordinates import ICRS, EarthLocation
from astropy.table import QTable
from astropy_healpix import HEALPix

from .. import missions
from ..fov import footprint_healpix
from ..synphot import DustExtinction, observing
from .core import app


@app.command()
def depth_map(
    schedule: Annotated[
        typer.FileBinaryRead,
        typer.Argument(help="Input filename for schedule", metavar="SCHEDULE.ecsv"),
    ],
    output: Annotated[
        typer.FileBinaryWrite,
        typer.Argument(help="Output FITS filename", metavar="OUTPUT.fits[.gz]"),
    ],
):
    """Generating a limiting magnitude map for an observing plan."""
    # Read schedule, keep only observing segments
    table = QTable.read(schedule, format="ascii.ecsv")
    table = table[table["action"] == "observe"]

    # Read arguments form schedule
    mission: missions.Mission = getattr(missions, table.meta["args"]["mission"])
    hpx = HEALPix(table.meta["args"]["nside"], frame=ICRS(), order="nested")
    bandpass = table.meta["args"]["bandpass"]
    snr = table.meta["args"]["snr"]
    footprints = footprint_healpix(
        hpx, mission.fov, table["target_coord"], table["roll"]
    )

    # Midpoint time of each observation
    times = table["start_time"] + 0.5 * table["duration"]

    # Evaluate the limiting magnitude for each observation, for each pixel.
    # Note that because each field may contain a different number of pixels,
    # we can't use Numpy array broadcasting. Instead, we repeat the entries
    # from the observer_location and time columns an appropriate number of
    # times.
    with observing(
        observer_location=EarthLocation.from_geocentric(
            *(
                np.concatenate(
                    [
                        np.tile(value, (len(footprint), 1))
                        for value, footprint in zip(
                            np.column_stack(
                                [
                                    coord.value
                                    for coord in table["observer_location"].geocentric
                                ]
                            ),
                            footprints,
                        )
                    ]
                ).T
                * table["observer_location"].geocentric[0].unit
            )
        ),
        target_coord=hpx.healpix_to_skycoord(np.concatenate(footprints)),
        obstime=times[0]
        + np.concatenate(
            [
                np.full(len(footprint), value)
                for value, footprint in zip(
                    (times - times[0]).to_value(u.s), footprints
                )
            ]
        )
        * u.s,
    ):
        limmag = mission.detector.get_limmag(
            snr,
            np.concatenate(
                [
                    np.full(len(footprint), value)
                    for value, footprint in zip(table["duration"].value, footprints)
                ]
            )
            * table["duration"].unit,
            synphot.SourceSpectrum(synphot.ConstFlux1D, amplitude=0 * u.ABmag)
            * DustExtinction(),
            bandpass,
        ).to_value(u.mag)

    # Find deepest limiting magnitude for each pixel
    table = (
        QTable(
            {
                "ipix": np.concatenate(
                    (np.arange(hpx.npix), np.concatenate(footprints))
                ),
                "limmag": np.concatenate((np.full(hpx.npix, -np.inf), limmag)),
            }
        )
        .group_by("ipix")
        .groups.aggregate(np.max)
    )
    table.sort("ipix")

    hp.write_map(
        output.name,
        table["limmag"],
        coord="C",
        nest=True,
        column_names=["LIMMAG"],
        column_units=["mag"],
        overwrite=True,
    )
