"""register_adapters: the omarchy-mlx Muse adapter is found through OMARCHY_MLX_SERVE and registered before a rank loads a model."""
import sys

import pytest

from omarchy_cluster import adapters


def fake_serve(tmp_path, returns):
    pkg = tmp_path / "mlx_omarchy_muse"
    pkg.mkdir()
    (pkg / "__init__.py").write_text(
        "import os\n"
        "def register():\n"
        "    with open(os.path.join(os.path.dirname(__file__), 'called'), 'a') as f:\n"
        "        f.write('x')\n"
        "    return %r\n" % returns)
    return pkg


def test_serve_directory_from_the_environment_is_searched_and_register_is_called(tmp_path, monkeypatch):
    pkg = fake_serve(tmp_path, True)
    monkeypatch.setenv("OMARCHY_MLX_SERVE", str(tmp_path))
    monkeypatch.delitem(sys.modules, "mlx_omarchy_muse", raising=False)
    monkeypatch.setattr(sys, "path", list(sys.path))
    assert adapters.register_adapters() is True
    assert (pkg / "called").read_text() == "x"
    assert sys.path.count(str(tmp_path)) == 1
    adapters.register_adapters()  # a second rank load in the same process: the path is not added again
    assert sys.path.count(str(tmp_path)) == 1


def test_a_package_that_has_nothing_to_add_returns_false(tmp_path, monkeypatch):
    fake_serve(tmp_path, False)  # mlx-lm already provides a muse_spark module
    monkeypatch.setenv("OMARCHY_MLX_SERVE", str(tmp_path))
    monkeypatch.delitem(sys.modules, "mlx_omarchy_muse", raising=False)
    monkeypatch.setattr(sys, "path", list(sys.path))
    assert adapters.register_adapters() is False


def test_a_missing_package_is_not_an_error(monkeypatch):
    monkeypatch.delenv("OMARCHY_MLX_SERVE", raising=False)
    monkeypatch.setitem(sys.modules, "mlx_omarchy_muse", None)  # makes `import mlx_omarchy_muse` raise ImportError
    assert adapters.register_adapters() is False


def test_a_missing_dependency_inside_the_adapter_is_not_hidden(tmp_path, monkeypatch):
    pkg = tmp_path / "mlx_omarchy_muse"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("import a_module_that_does_not_exist_anywhere\n")
    monkeypatch.setenv("OMARCHY_MLX_SERVE", str(tmp_path))
    monkeypatch.delitem(sys.modules, "mlx_omarchy_muse", raising=False)
    monkeypatch.setattr(sys, "path", list(sys.path))
    with pytest.raises(ModuleNotFoundError, match="a_module_that_does_not_exist_anywhere"):
        adapters.register_adapters()
