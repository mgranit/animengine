import numpy as np
import pytest

from animengine import ease_in_out_cubic, linear, smoothstep


@pytest.mark.parametrize("ease", [linear, smoothstep, ease_in_out_cubic])
def test_endpoints_and_midpoint(ease):
    assert ease(0.0) == pytest.approx(0.0)
    assert ease(0.5) == pytest.approx(0.5)
    assert ease(1.0) == pytest.approx(1.0)


@pytest.mark.parametrize("ease", [smoothstep, ease_in_out_cubic])
def test_clamped_and_monotonic(ease):
    t = np.linspace(-0.5, 1.5, 201)
    values = ease(t)
    assert values.min() == pytest.approx(0.0)
    assert values.max() == pytest.approx(1.0)
    assert np.all(np.diff(values) >= -1e-12)
