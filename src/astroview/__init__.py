"""Old name of `splosh` (renamed 2026-11-11): `import astroview` and `import astroview.X` give the splosh modules themselves (the same objects, so the
plugin registries are shared), and `python -m astroview` / the `astroview` command run splosh. Kept so that old scripts and plugins keep working."""
import importlib
import importlib.abc
import importlib.util
import sys

import splosh as _splosh


class _Alias(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, name, path=None, target=None):
        if name.startswith("astroview."):
            return importlib.util.spec_from_loader(name, self)
        return None

    def create_module(self, spec):
        return importlib.import_module("splosh" + spec.name[len("astroview"):])

    def exec_module(self, module):
        pass


sys.meta_path.insert(0, _Alias())
sys.modules[__name__] = _splosh
