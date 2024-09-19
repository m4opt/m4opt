from typing import Annotated

import click
import typer

from .. import __version__, missions

app = typer.Typer(pretty_exceptions_show_locals=False)


def version_callback(value: bool):
    if value:
        print(__version__)
        raise typer.Exit()


@app.callback()
def version(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=version_callback,
            is_eager=True,
            help="Print version and exit.",
        ),
    ] = False,
):
    pass


class MissionParam(click.Choice):
    def __init__(self):
        choices = [name for name in missions.__all__ if name[0].islower()]
        super().__init__(choices)

    def convert(self, value, *args, **kwargs):
        if isinstance(value, missions.Mission):
            return value
        else:
            return getattr(missions, super().convert(value, *args, **kwargs))


MissionOption = Annotated[
    missions.Mission, typer.Option(click_type=MissionParam(), show_default=False)
]
