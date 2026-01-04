"""Test module for polytope functionalities

List of tests:
- Test initialization of non-degenerate H-representation polytope
    - Test initializers gp.poly(A, b), gp.poly(A=a, b=b), gp.Polytope(A, b), gp.Polytope(A=a, b=b)
        - Should gp.poly(A, b=b) also be allowed?
        - Check that gp.poly(A, b, V=...), gp.poly(A=a, b=b, V=...) raise errors
        - Check initializer with A.shape = (m,), b.shape = (m, 1)  # NOTE: How should this be handled?
        - Check initializer with A.shape = (1, m), b.shape = (n, 1) is accepted
    - Check extensively for 1D, 2D, 3D
    - Check also 4D - 10D
    - Check if poly.n = A.shape[1]
    - Check if poly.dim = n for non-degenerate
    - Check if poly.is_degen = False for non-degenerate
    - ...

- Test initialization of degenerate H-representation polytope
    - Test initializers gp.poly(A, b), gp.poly(A=a, b=b), gp.Polytope(A, b), gp.Polytope(A=a, b=b)
        - Check that A_eq and b_eq are set after initialization  # NOTE: How do we want to handle this standardization? Do we 1) always convert 'degenerate' inequalities to equalities, or 2) always convert equalities to inequalities, or 3) keep both representations?
    - Test initializers gp.poly(A, b, A_eq=A_eq, b_eq=b_eq), ...
    - ...
    - Check that poly.dim < n for degenerate where poly.vol = 0
    - Check that poly.dim < n for degenerate where poly.vol = inf  # TODO: Check out dim = n - rank(A_eq), provided by Copilot
    - Check that poly.is_degen = True for degenerate

- Test initialization of empty polytope
    - Test initializers gp.poly(n=n), gp.Polytope(n=n) | No positional arguments provided
        - Check that gp.poly(3) raises an error (should be interpreted as V-representation with one vertex at [3])  # NOTE: Should it? Or should it just interpret it as a V-representation with one vertex at [3]? Do we require V to be a 2D array always? What about 1D arrays with one column/row?
    - Test initializers gp.poly(V=np.array([]).reshape(n, 0)) for empty V-representation
    - Test initializers gp.poly(A=np.array([]).reshape(0, n), b=np.array([]).reshape(0,)) for empty H-representation
    - Check that poly.is_empty = True
    - Check that poly.dim = 0
    - Check that poly.n = n

- Test containment of polytopes
    - Check that for vert in poly.verts: vert in poly equals True
    - Check that intermediate points are contained in poly, such that for vert1, vert2 in poly.verts: 0.5*(vert1 + vert2) in poly equals True
    - Check that points outside poly return False for containment check
    - Check that com of poly in poly equals True  # NOTE: Maybe we don't want to have com as a property, as we are always dealing with the centroid
    - Check that centroid of poly in poly equals True
    - Check that cheb_center of poly in poly equals True

- Test reduction of non-degenerate H-representation polytopes 
    - Check if poly.reduce() works, and removes redundant inequalities (for some examples)

- Test initialization of degenerate V-representation polytope from rays
    - Check that if V.shape = (n=2, 1), R.shape= (n=2, 1), and rays=[R] (to few rays to define a polytope/polyhedron: we need exactly two rays in 2D to define a polyhedral cone), an error is raised
    - Check that if V.shape = (n=2, 2), R_1.shape = (n=2, 2), R_2.shape = (n=2, 2), and rays=[R_1, R_2] (two sets of rays defining two different polyhedral cones), an error is raised: we cannot have two rays extending from two different vertices, as this would not define a (single, convex) polytope/polyhedron
    - V.shape = (n=2, 2), R_1.shape = (n=2, 1), R_2.shape = (n=2, 1), and rays=[R_1, R_2] (one ray per vertex, two vertices in total), a proper, degenerate polytope

- Test linear mapping of polytopes
    - Check that for a given polytope poly and a matrix M with compatible dimensions, M @ poly works as expected
    - Check that for a given polytope poly and a matrix M with compatible dimensions, poly @ M works is not allowed (should raise an error)  # NOTE: This should be the case, right? Since the vertices verts are column vectors, and thus poly @ M would not make sense
    - Check that for incompatible dimensions, a DimensionError is raised
    - Check that if A is singular, the linear mapping still works as expected
        - Check that the resulting polytope is degenerate

"""

import numpy as np
import scipy as sp
import pytest

import geopes as gp

@pytest.mark.parametrize("input, expected", [
    ((np.array([[1, 0], [0, 1], [-1, 0], [0, -1]]), np.array([1, 1, 1, 1])), ((np.array([[1, 0], [0, 1], [-1, 0], [0, -1]]), np.array([1, 1, 1, 1])), 2, False, 2))  # A.shape = (4, 2), b.shape = (4,)
])


def test_polytope_init_hrepr(input, expected) -> None:
    # FIXME: I have no idea if this actually works; I need to see this setup actually works...
    # Test H-representation initialization
    poly = gp.Polytope(*input)
    assert np.isclose(poly.A, expected[0][0]).all() and np.isclose(poly.b, expected[0][1]).all(), "Polytope H-representation initialization failed: attributes do not match input"
    assert poly.n == expected[1], "Polytope H-representation initialization failed: dimension n incorrect, should be 2"
    assert poly.is_degen == expected[2], "Polytope H-representation initialization failed: is_degen should be set to false"
    assert poly.dim == expected[3], "Polytope H-representation initialization failed: dimension dim incorrect, should be 2"


def test_polytope_init_vrepr() -> None:
    V = np.array([[0, 0], [1, 0], [1, 1], [0, 1]]); poly = gp.Polytope(V)
    # Test V-representation initialization
    assert np.array_equal(poly.V, V), "Polytope V-representation initialization failed: vertices do not match input"


def test_polytope_init_vrepr_degen_no_rays() -> None:
    V = np.array([[0, 0], [1, 0]]); poly = gp.Polytope(V)  # Two points defining a line segment
    # Test degenerate V-representation initialization (line segment)
    assert np.array_equal(poly.V, V), "Polytope degenerate V-representation initialization failed: vertices do not match input"
    assert poly.is_degen, "Polytope degenerate V-representation initialization failed: is_degen should be set to true"