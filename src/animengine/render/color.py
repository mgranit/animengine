import re

import numpy as np

_HEX = re.compile(r"^#([0-9a-fA-F]{3}|[0-9a-fA-F]{4}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})$")


def to_rgba(value) -> np.ndarray:
    """Convert '#rgb[a]', '#rrggbb[aa]' or a 3/4-sequence of floats in [0, 1] to float32 RGBA."""
    if isinstance(value, str):
        match = _HEX.match(value)
        if match is None:
            raise ValueError(f"Invalid hex colour: {value!r}")
        digits = match.group(1)
        if len(digits) in (3, 4):
            digits = "".join(c * 2 for c in digits)
        channels = [int(digits[i:i + 2], 16) / 255.0 for i in range(0, len(digits), 2)]
    else:
        channels = [float(c) for c in value]

    if len(channels) == 3:
        channels.append(1.0)
    if len(channels) != 4:
        raise ValueError(f"Colour needs 3 or 4 channels, got {len(channels)}")
    return np.array(channels, dtype=np.float32)
