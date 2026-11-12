"""Plugin discovery: the environment variable SPLOSH_PLUGINS and an installed entry point of the group `splosh.plugins` (a fake .dist-info directory on sys.path, no venv)."""
import importlib
import sys
import textwrap

import pytest

from splosh import plugins

TOY = textwrap.dedent('''
    from splosh import plugins
    plugins.register_field("toy_field_{tag}", lambda ctx: None, label="toy")
    plugins.register_preset("toy_preset_{tag}", {{"fov": 1.0}})
''')


def toys(loaded):
    """Only the toy plugins of these tests: any other installed plugin is not the subject here."""
    return [m for m in loaded if m.startswith("toyplug_")]


@pytest.fixture
def clean_registry(monkeypatch):
    """Isolate the registry and the module list; restore everything afterwards."""
    monkeypatch.setattr(plugins, "FIELDS", dict(plugins.FIELDS))
    monkeypatch.setattr(plugins, "PRESETS", dict(plugins.PRESETS))
    monkeypatch.setattr(plugins, "_loaded", {"done": False, "modules": [], "errors": []})
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.delenv("SPLOSH_PLUGINS", raising=False)
    before = set(sys.modules)
    yield
    for n in set(sys.modules) - before:
        if n.startswith("toyplug_"):
            del sys.modules[n]
    importlib.invalidate_caches()


def test_discovery_through_environment_variable(tmp_path, monkeypatch, clean_registry):
    (tmp_path / "toyplug_env.py").write_text(TOY.format(tag="env"))
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("SPLOSH_PLUGINS", "toyplug_env, toyplug_missing")
    assert toys(plugins.load_plugins(force=True)) == ["toyplug_env"]
    assert "toy_field_env" in plugins.FIELDS and "toy_preset_env" in plugins.PRESETS
    assert [n for n, _ in plugins.errors()] == ["toyplug_missing"]       # a broken plugin is reported, not fatal
    assert toys(plugins.load_plugins()) == ["toyplug_env"]                     # idempotent


def test_discovery_through_installed_entry_point(tmp_path, monkeypatch, clean_registry):
    (tmp_path / "toyplug_ep.py").write_text(TOY.format(tag="ep"))
    di = tmp_path / "toyplug_ep-0.0.1.dist-info"
    di.mkdir()
    (di / "METADATA").write_text("Metadata-Version: 2.1\nName: toyplug-ep\nVersion: 0.0.1\n")
    (di / "entry_points.txt").write_text("[splosh.plugins]\ntoy = toyplug_ep\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    assert "toyplug_ep" not in sys.modules
    assert toys(plugins.load_plugins(force=True)) == ["toyplug_ep"]
    assert "toy_field_ep" in plugins.FIELDS and "toy_preset_ep" in plugins.PRESETS
    assert plugins.errors() == []


def test_entry_point_and_env_var_naming_the_same_module_import_it_once(tmp_path, monkeypatch, clean_registry):
    (tmp_path / "toyplug_both.py").write_text(TOY.format(tag="both"))
    di = tmp_path / "toyplug_both-0.0.1.dist-info"
    di.mkdir()
    (di / "METADATA").write_text("Metadata-Version: 2.1\nName: toyplug-both\nVersion: 0.0.1\n")
    (di / "entry_points.txt").write_text("[splosh.plugins]\ntoy = toyplug_both:something\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("SPLOSH_PLUGINS", "toyplug_both")
    assert toys(plugins.load_plugins(force=True)) == ["toyplug_both"]
