from contextlib import contextmanager
from itertools import accumulate, chain
from typing import Annotated, Iterable, cast

import numpy as np
import typer
from astropy import units as u
from astropy.coordinates import ICRS
from astropy.table import QTable
from astropy.time import Time
from astropy.visualization.units import quantity_support as _quantity_support
from astropy_healpix import HEALPix
from ligo.skymap.bayestar import rasterize
from ligo.skymap.io import read_sky_map
from ligo.skymap.plot.allsky import AutoScaledWCSAxes
from ligo.skymap.plot.poly import cut_prime_meridian
from ligo.skymap.postprocess import find_greedy_credible_levels
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.transforms import BlendedAffine2D
from matplotlib.typing import ColorType

from .. import missions
from ..fov import footprint, footprint_healpix
from ..utils.console import progress, status
from .core import app


@contextmanager
def quantity_support(*args, **kwargs):
    """Workaround for https://github.com/astropy/astropy/pull/17006."""
    with _quantity_support(*args, **kwargs):
        yield


@app.command()
@progress()
@quantity_support()
def animate(
    schedule: Annotated[
        typer.FileBinaryRead,
        typer.Argument(help="Input filename for schedule", metavar="SCHEDULE.ecsv"),
    ],
    output: Annotated[
        typer.FileBinaryWrite,
        typer.Argument(help="Output filename for animation", metavar="MOVIE.gif"),
    ],
    time_step: Annotated[
        u.Quantity[u.physical.time],
        typer.Option(
            parser=u.Quantity,
            help="Time step for evaluating field of regard",
        ),
    ] = "1 hour",
):
    with status("loading schedule"):
        table = QTable.read(schedule, format="ascii.ecsv")
        table["end_time"] = table["start_time"] + table["duration"]
        table = table[table["action"] == "observe"]
        deadline = table.meta["args"]["deadline"]
        delay = table.meta["args"]["delay"]
        nside = table.meta["args"]["nside"]
        skymap = table.meta["args"]["skymap"]
        visits = table.meta["args"]["visits"]

    with status("loading sky map"):
        hpx = HEALPix(nside, frame=ICRS(), order="nested")
        skymap_moc = read_sky_map(skymap, moc=True)
        probs = rasterize(skymap_moc["UNIQ", "PROBDENSITY"], hpx.level)["PROB"]
        event_time = Time(
            Time(skymap_moc.meta["gps_time"], format="gps").utc, format="iso"
        )

    with status("making animation"):
        with status("setting up axes"):
            fig = plt.figure()
            gs = fig.add_gridspec(2, 1, height_ratios=[4, 1])
            colormap = plt.get_cmap("Paired")
            assert isinstance(colormap, ListedColormap)
            (
                field_of_regard_color,
                averaged_field_of_regard_color,
                _,
                skymap_color,
                _,
                footprint_color,
                *_,
            ) = cast(Iterable[ColorType], colormap.colors)
            now_color = "black"
            footprint_alpha = 1 / visits

            ax_map = fig.add_subplot(gs[0], projection="astro hours mollweide")
            assert isinstance(ax_map, AutoScaledWCSAxes)
            transform = ax_map.get_transform("world")
            ax_map.add_artist(
                ax_map.legend(
                    [
                        Patch(facecolor=footprint_color, alpha=footprint_alpha),
                        Patch(facecolor="none", edgecolor=skymap_color),
                    ],
                    [
                        "observation footprints",
                        "90% credible region",
                    ],
                    loc="upper left",
                    borderaxespad=0,
                    bbox_to_anchor=[0, 1],
                    bbox_transform=fig.transFigure,
                )
            )
            ax_map.legend(
                [
                    Patch(facecolor=averaged_field_of_regard_color),
                    Patch(facecolor=field_of_regard_color),
                ],
                [
                    "averaged",
                    "instantaneous",
                ],
                loc="upper right",
                title="outside field of regard",
                borderaxespad=0,
                bbox_to_anchor=[1, 1],
                bbox_transform=fig.transFigure,
            )

            mission: missions.Mission = getattr(missions, table.meta["args"]["mission"])
            time_steps = (
                event_time
                + np.arange(
                    0,
                    (deadline + time_step).to_value(u.s),
                    time_step.to_value(u.s),
                )
                * u.s
            )

            cls = find_greedy_credible_levels(probs)
            ax_map.contour_hpx(cls, levels=[0.9], colors=[skymap_color], nested=True)
            ax_map.grid()

            observer_locations = mission.orbit(time_steps).earth_location

        with status("adding field of regard"):
            instantaneous_field_of_regard = np.logical_and.reduce(
                [
                    constraint(
                        observer_locations[:, np.newaxis],
                        hpx.healpix_to_skycoord(np.arange(hpx.npix)),
                        time_steps[:, np.newaxis],
                    )
                    for constraint in mission.constraints
                ],
                axis=0,
            )
            averaged_field_of_regard = np.logical_or.reduce(
                instantaneous_field_of_regard, axis=0
            )
            ax_map.contourf_hpx(
                averaged_field_of_regard,
                nested=True,
                levels=[-1, 0.5],
                colors=[averaged_field_of_regard_color],
                zorder=1.1,
            )

        with status("adding observation footprints"):
            footprint_patches = [
                [
                    plt.Polygon(
                        np.rad2deg(vertices),
                        transform=transform,
                        visible=False,
                        facecolor=footprint_color,
                        alpha=footprint_alpha,
                    )
                    for vertices in cut_prime_meridian(
                        np.column_stack(
                            (region.vertices.ra.rad, region.vertices.dec.rad)
                        )
                    )
                ]
                for region in footprint(
                    mission.fov, table["target_coord"], table["roll"]
                )
            ]
            for patch in chain.from_iterable(footprint_patches):
                ax_map.add_patch(patch)

            table["area"], table["prob"] = zip(
                *map(
                    lambda pixels: [len(pixels) * hpx.pixel_area, probs[pixels].sum()],
                    accumulate(
                        footprint_healpix(
                            hpx, mission.fov, table["target_coord"], table["roll"]
                        ),
                        lambda *args: np.unique(np.concatenate(args)),
                    ),
                )
            )

            ax_timeline = fig.add_subplot(gs[1])
            ax_timeline.set_xlim(0 * u.hour, deadline)
            ax_timeline.set_xlabel(f"time in hours since event at {event_time}")
            ax_timeline.set_ylim(0, 1)
            ax_timeline.yaxis.set_tick_params(
                color=skymap_color, labelcolor=skymap_color
            )
            ax_timeline.set_ylabel("probability", color=skymap_color)

            ax_area = ax_timeline.twinx()
            ax_area.yaxis.set_tick_params(
                color=footprint_color, labelcolor=footprint_color
            )
            ax_area.set_ylabel("area (deg$^2$)", color=footprint_color)

            ax_timeline.hlines(
                table["prob"],
                (table["start_time"] - time_steps[0]).to(u.hour),
                (table["end_time"] - time_steps[0]).to(u.hour),
                skymap_color,
            )
            ax_area.hlines(
                table["area"],
                (table["start_time"] - time_steps[0]).to(u.hour),
                (table["end_time"] - time_steps[0]).to(u.hour),
                footprint_color,
            )
            ax_area.set_ylim(0 * u.deg**2)

            now_line = ax_timeline.axvline(
                (time_steps[0] - event_time).to(u.hour),
                color=now_color,
            )
            now_label = ax_timeline.text(
                (time_steps[0] - event_time).to(u.hour),
                1,
                "now",
                ha="center",
                va="bottom",
                color=now_color,
                transform=BlendedAffine2D(ax_timeline.transData, ax_timeline.transAxes),
            )
            ax_timeline.axvspan(0 * u.hour, delay, color="lightgray")
            ax_timeline.text(
                delay,
                0.5,
                "delay",
                rotation=90,
                rotation_mode="anchor",
                horizontalalignment="center",
                verticalalignment="bottom",
                transform=BlendedAffine2D(ax_timeline.transData, ax_timeline.transAxes),
            )

        with status("rendering frames"):
            field_of_regard_artist = None

            def draw_frame(args):
                time, field_of_regard = args
                blit = [now_line, now_label]
                time_from_start = (time - event_time).to(u.hour)
                now_line.set_xdata([time_from_start])
                now_label.set_x(time_from_start)
                for patches, visible in zip(
                    footprint_patches, table["start_time"] <= time
                ):
                    for patch in patches:
                        if visible != patch.get_visible():
                            patch.set_visible(visible)
                            blit.append(patch)

                nonlocal field_of_regard_artist
                if field_of_regard_artist is not None:
                    blit.append(field_of_regard_artist)
                    field_of_regard_artist.remove()
                field_of_regard_artist = ax_map.contourf_hpx(
                    field_of_regard,
                    nested=True,
                    colors=[field_of_regard_color],
                    levels=[-1, 0.5],
                )
                blit.append(field_of_regard_artist)

                return blit

            FuncAnimation(
                fig,
                draw_frame,
                frames=zip(time_steps, instantaneous_field_of_regard),
                save_count=len(time_steps),
                blit=True,
            ).save(output.name)
