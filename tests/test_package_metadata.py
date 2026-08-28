from importlib.metadata import version as distribution_version

import pdecert


def test_runtime_version_matches_installed_distribution():
    assert pdecert.__version__ == distribution_version("pdecert")
