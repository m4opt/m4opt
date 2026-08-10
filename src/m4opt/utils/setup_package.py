import numpy as np


def get_extensions():
    from setuptools import Extension

    return [
        Extension(
            "m4opt.utils._numpy",
            ["src/m4opt/utils/_numpy.c"],
            define_macros=[
                ("NPY_TARGET_VERSION", "NPY_2_0_API_VERSION"),
                ("NPY_NO_DEPRECATED_API", "NPY_2_0_API_VERSION"),
            ],
            include_dirs=[np.get_include()],
        )
    ]
