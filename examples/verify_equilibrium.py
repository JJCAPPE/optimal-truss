"""Independent textbook-sized check, NOT the assigned course practice problem.

Run from the repository root: python examples/verify_equilibrium.py
Requires the package to be installed (python -m pip install -e .).
"""

import numpy as np

from optimal_truss.analysis import solve_equilibrium
from optimal_truss.model import Truss


# A 6-8-10 right triangle, pin at the left and roller at the right.
truss = Truss("Analytical verification triangle",
              nodes=[[0, 0], [8, 0], [0, 6]],
              members=[[0, 1], [0, 2], [1, 2]],
              pin=0, roller=1, load_joint=2)
loads = np.array([[0., 0.], [0., 0.], [4., -3.]])

# At the upper joint, diagonal direction is (0.8, -0.6):
# Fx: 4 + 0.8 F3 = 0 -> F3 = -5 oz (compression).
# Fy: -3 - F2 - 0.6 F3 = 0 -> F2 = 0.
# Right joint: -F1 - 0.8 F3 = 0 -> F1 = 4 oz (tension).
# Whole truss: 8 Ry - 6*4 = 0 -> Ry = 3; pin Fx=-4, pin Fy=0.
expected_members = np.array([4., 0., -5.])
expected_reactions = np.array([-4., 0., 3.])
members, reactions = solve_equilibrium(truss, loads)
np.testing.assert_allclose(members, expected_members, atol=1e-12)
np.testing.assert_allclose(reactions, expected_reactions, atol=1e-12)
print("Analytical comparison passed.")
print("Member forces (oz, + tension):", members)
print("Reactions (oz, pin-x/pin-y/roller-y):", reactions)
