import diagnost


def test_runtime_version_matches_package_version():
    assert diagnost.__version__ == "0.1.1"
