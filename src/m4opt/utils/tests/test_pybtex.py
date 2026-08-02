import pytest
from sphinx.testing.util import SphinxTestApp


@pytest.mark.sphinx("html", testroot="pybtex")
def test_short_alpha(app: SphinxTestApp):
    app.build()
    result = (app.outdir / "index.html").read_text()
    assert (
        "Karl D. Bilimoria and Bong Wie. Time-optimal three-axis reorientation of a rigid spacecraft."
        in result
    )
    assert (
        "Igor Andreoni, Daniel A. Goldstein, Shreya Anand, et al. GROWTH on S190510g:"
        in result
    )
