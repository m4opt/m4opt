import sys

if sys.version_info >= (3, 12):
    from typing import override
else:
    # FIXME: requires Python >= 3.12
    from typing_extensions import override

__all__ = ("override",)
