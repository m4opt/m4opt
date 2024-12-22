import pytest
from typer.testing import CliRunner


@pytest.fixture
def run_cli():
    runner = CliRunner()

    def func(app, *args):
        result = runner.invoke(app, args, catch_exceptions=False)
        print(result.output)
        return result

    return func