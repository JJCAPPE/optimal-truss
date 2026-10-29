"""Explicit geometry generators; generated geometry is still checked separately."""

import numpy as np

from .model import Truss


def warren(span=30.0, panels=3, raised_load=False, load_height=1.25):
    """Generate a Warren truss with equilateral reference panels.

    The exact version has horizontal chords and is NOT assignment compliant:
    its first interior bottom joint, where the load acts, is at support height.
    The modified version raises only that joint, retaining the connectivity and
    horizontal top chord. Its affected triangles are no longer equilateral.
    """
    if not isinstance(panels, int) or panels < 2:
        raise ValueError("panels must be an integer of at least two")
    if not np.isfinite(span) or span <= 0 or not np.isfinite(load_height):
        raise ValueError("Span must be positive and all dimensions finite")
    width = span / panels
    bottom = [[i * width, 0.0] for i in range(panels + 1)]
    top = [[(i + 0.5) * width, width * np.sqrt(3) / 2] for i in range(panels)]
    if raised_load:
        bottom[1][1] = load_height
    members = [[i, i + 1] for i in range(panels)]
    members += [[panels + 1 + i, panels + 2 + i] for i in range(panels - 1)]
    for i in range(panels):
        members += [[i, panels + 1 + i], [i + 1, panels + 1 + i]]
    name = "Modified Warren baseline" if raised_load else "Equilateral Warren (noncompliant reference)"
    return Truss(name, np.array(bottom + top), np.array(members), 0, panels, 1)
