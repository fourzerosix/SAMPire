import pytest

from sampire import selftest


@pytest.mark.parametrize("name,fn", selftest.CHECKS, ids=[n for n, _ in selftest.CHECKS])
def test_check(name, fn):
    assert fn()
