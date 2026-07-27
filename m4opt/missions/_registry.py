from .._plugin import pluginmanager
from . import _core

pluginmanager.add_hookspecs(_core)
pluginmanager.load_setuptools_entrypoints("m4opt")


def names() -> list[str]:
    return [name for name, _ in pluginmanager.list_name_plugin()]


def get(name: str) -> _core.Mission:
    return pluginmanager.get_plugin(name).register_mission()
