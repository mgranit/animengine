import pytest

from animengine.platform import Window


@pytest.fixture(scope="session")
def require_gl():
    try:
        Window(16, 16, visible=False).close()
    except Exception as error:
        pytest.skip(f"No OpenGL context available: {error}")
