# Mechanics and optimization method

## Equilibrium

For a member joining joints `i` and `j`, compute its length
`l = norm(x_j - x_i)` and direction `u = (x_j - x_i)/l`. Its column in the
equilibrium matrix contains `+u` at joint `i` and `-u` at joint `j`.
This sign convention makes positive member force tension.

The first J equations are horizontal equilibrium; the next J are vertical.
The last three columns represent pin-x, pin-y, and roller-y reactions.
`numpy.linalg.solve(A, -external_loads)` solves all forces together. It avoids
explicitly forming a matrix inverse. A condition-number check rejects singular
or severely ill-conditioned geometry. Matrix assembly is O(M), and the dense
linear solve is O(J³); these course trusses are small.

The connection matrix `C` has **J rows and M columns**, with exactly two ones
in each column. `M = 2J - 3` makes the equilibrium matrix square but does not
guarantee geometric stability. The validator also checks triangular growth,
crossings, and the actual matrix.

The main `solve` function applies a downward load at `load_joint`. The
`solve_equilibrium` function accepts an N-by-2 array of signed physical nodal
loads for verifying other load cases, including horizontal forces.

## Buckling and uncertainty

The supplied class fit is

```text
B_i = 3391.349 * l_i^(-2.012) oz-force, with fit band U = 1.52 oz-force.
```

Lengths must be in inches. The plot in the class data sheet gives coefficient
3391.349; the manual rounds it to 3391.3. No raw class measurements are bundled
or reconstructed from the graph.

Solve once under a unit downward load. Let `r_i` be the resulting member force
per unit applied load. Under load `W`, the member force is `F_i = r_i W`.
Only members with `r_i < 0` are candidates for compression buckling:

```text
W_i = B_i / |r_i|
W_failure = min(W_i over compressed members)
U_W = U / |r_critical|
```

Near ties are reported as multiple critical members; the largest propagated
band among these is retained. The code also computes both all-member envelopes:

```text
W_lower = min(max(B_i - U, 0) / |r_i|)
W_upper = min((B_i + U) / |r_i|)
```

This permits the first-failing member to change when strength changes. The fit
sheet describes a 95% fit band; it does **not** supply a joint probability model
or independent per-member standard deviations. Accordingly, the envelope is
not labeled a 95% confidence interval for whole-truss survival.

The cost is `sum(member lengths in inches) + 10 * number of joints`, in the
assignment's artificial currency. Load/cost uses nominal failure load.

## Feasibility

The validator reports every check, rather than returning only a yes/no flag:

- Simple, triangular, planar connectivity with `M = 2J - 3`.
- No crossing members, overlaps, zero lengths, or members passing through joints.
- A stable equilibrium matrix; a pin and vertical roller at the end supports.
- Support span 30–33 inches and no part below the support baseline.
- Every member 8.5–14 inches long.
- Load joint 0.5–1.5 inches above the supports and no joint directly underneath.
- Total member length no greater than 120 inches.
- A feasible allocation of all pieces to stock lengths 48, 48, and 24 inches.
- Nominal predicted capacity at least 30 oz-force.

Stock assignment uses descending-length backtracking, skipping equivalent
remaining-capacity states. It finds a real cutting assignment, not just a total
length check. A conservative kerf is charged once per piece when requested.
Joint/end allowances are outside the supplied joint-to-joint geometry model.
Nominal capacity does not verify a physical 60-second hold.

## Search

Differential evolution is a population-based numerical optimizer. It is
appropriate here because evaluations are inexpensive, geometry constraints can
be discontinuous, and the identity of the critical member can switch. There is
no training set or learned approximation to the statics equations.

Parameters include support span, each unsupported joint's horizontal position
as a fraction of span, and its height. The load-joint height uses the specified
0.5–1.5-inch bounds. Other height bounds follow from shortest graph distance to
a support times maximum member length; this is a geometric upper bound rather
than an arbitrary visual limit.

The initial population includes the feasible baseline, local perturbations,
and candidates across the coordinate bounds. Invalid candidates receive a
penalty. They can participate in the population's exploration, but they cannot
become the saved best feasible design. Every accepted incumbent passes the
independent validator, including stock packing.

The search preserves connectivity and load/support identities. It therefore
optimizes geometry within one topology; changing an input's members explores a
different topology in a separate run. Multiple seeds are independent restarts.
The final artifact is independently checked again. The metadata distinguishes
convergence from exhaustion of the generation budget and records the objective,
seed, bounds, function evaluations, and runtime.

Optimization records an initial incumbent and one real snapshot per generation.
The full trace supports progress plots, animations, and investigation of when
capacity improves. The same seed, inputs, and library versions reproduce the
numerical run; execution time and rendered-file metadata may vary.

## MATLAB interoperability

`design.mat` contains `C[J,M]`, `Sx[J,3]`, `Sy[J,3]`, `X[1,J]`, `Y[1,J]`, and
`L[2J,1]`. Reaction-column order is pin-x, pin-y, roller-y. `L` is the
equilibrium right-hand side: a downward physical load appears as a **positive**
entry in its vertical block. This follows the manual's equation rearrangement.
The tests reload the `.mat`, rebuild equilibrium from its contents, and compare
the resulting member forces and reactions with the Python output.
