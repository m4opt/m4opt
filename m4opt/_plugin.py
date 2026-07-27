import pluggy

hookspec = pluggy.HookspecMarker(__package__)
hookimpl = pluggy.HookimplMarker(__package__)
pluginmanager = pluggy.PluginManager(__package__)
