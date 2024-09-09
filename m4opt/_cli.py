import typer

app = typer.Typer()


@app.command()
def prime():
    """Download and cache all dependencies that m4opt may use at runtime.

    Under normal operation, m4opt will download and cache various external
    data sources (for example, IERS Earth orientation data and Planck dust
    maps). If you need to run m4opt in an environment with no outbound Internet
    connectivity (for example, some computing clusters), you can run this
    command to download and cache the external data sources immediately.
    """
    from .models._extinction import dust_map

    dust_map()


@app.command()
def placeholder():
    """A placeholder for a future command"""
