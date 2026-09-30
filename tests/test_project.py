def test_package_exposes_version() -> None:
    import fly_poker

    assert fly_poker.__version__ == "0.1.0"
