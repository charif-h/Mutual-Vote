import mutualvote


def test_version() -> None:
    assert isinstance(mutualvote.__version__, str)
