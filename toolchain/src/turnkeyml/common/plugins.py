import pkgutil
import importlib


def discover():
    return {
        name: importlib.import_module(name)
        for _, name, _ in pkgutil.iter_modules()
        if name.startswith("turnkeyml_plugin_")
    }
