from typing import Annotated

import typer

from .. import __version__
from . import _types  # noqa: F401, import for side effects only

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
