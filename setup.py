from sys import version_info

from setuptools import setup

setup(
    options={
        "bdist_wheel": {"py_limited_api": f"cp{version_info.major}{version_info.minor}"}
    }
)
