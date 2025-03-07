"""This is for now the main class, as long as the structure of the project is not fixed (i.e., no package on PyPI is available yet. ).

© Bart Wolleswinkel, 2025. Distributed under a ... license.

### TODO: Check out package 'polytope', https://tulip-control.github.io/polytope/
### TODO: Check out package 'pypolycontain', https://pypolycontain.readthedocs.io/en/latest/index.html
### TODO: Check out repository 'pytope', https://github.com/heirung/pytope
### TODO: Check out paper "LazySets.jl: Scalable Symbolic-Numeric Set Computations" Juliacon (2021), https://juliacon.github.io/proceedings-papers/jcon.00097/10.21105.jcon.00097.pdf#page1
### TODO: Check out d(ouble)under/numerical Python methods "3.3.8. Emulating numeric types", https://docs.python.org/3/reference/datamodel.html
### TODO: Check out the repository "Remote Tube-based MPC for Tracking Over Lossy Networks" (which has multiple implementations of MRPI sets), https://github.com/EricssonResearch/Robust-Tracking-MPC-over-Lossy-Networks/blob/main/src/LinearMPCOverNetworks/utils_polytope.py
### TODO: Check out documentation 'Standard operators as functions', https://docs.python.org/3/library/operator.html
"""

from __future__ import annotations
from typing import Callable
import warnings

import numpy as np
import scipy as sp
import control as ct
import matplotlib.pyplot as plt
from numpy.typing import ArrayLike
import cvxpy as cvx  ### NOTE: I want to make this an optional dependency, as it's not strictly needed for any core functionality


class Polytope:
    """The polytope class which implements a polytope `poly` = {x ∈ ℝ^n | Fx ≤ g}.
    This class emulates a numerical type, and has a H-representation (half-space 
    representation) and V-representation (vertex representation).

    ### FIXME: We need to do proper renaming: as (A, B, C) describes a system, we must use (F, g) for the polytope.

    Methods
    -------
    __init__(F, g)
        Initialize the Polytope object.
    
    """

    def __init__(self, F: ArrayLike = None, g: ArrayLike = None):
        """Initialize a Polytope object (see class for description) based on either a 
        H-representation or a V-representation.

        ### FIXME: Maybe we should just make this a H-space representations? And make a method verts_to_poly instead?
        
        Parameters
        ----------
        F : ArrayLike
            The matrix F ∈ ℝ^{p x n} in the H-representation {x ∈ ℝ^n | Fx ≤ g}.
        g : ArrayLike
            The vector g ∈ ℝ^p in the H-representation {x ∈ ℝ^n | Fx ≤ g}.
        
        """
        ### FIXME: This is not implemented yet
        self.H_repr = True
        self.V_repr = False
        self._F = F
        self._g = g
        self._verts = None
        self._cheb_c = None
        self._cheb_r = None
        self._vol = None
        self.n = F.shape[1]  ### FIXME: Placeholder
        self.min_repr = None 
        self.is_empty = False  ### NOTE: A polytope can have zero volume but still be non-empty

    @property
    def F(self):
        if not self.H_repr:
            self._F, self._g = hrepr(self)
        return self._F
    
    @property
    def g(self):
        if not self.H_repr:
            (self._F, self._g), self.H_repr = hrepr(self), True
        return self._g
    
    @property
    def verts(self):
        if not self.V_repr:
            self._verts, self.V_repr = extreme(self), True
        return self._verts
    
    @property
    def cheb_c(self):
        """Compute the Chebyshev center of the polytope.
        
        """
        if self._cheb_c is None:
            ...
        raise NotImplementedError
    
    @property
    def cheb_r(self):
        """Compute the Chebyshev radius of the polytope.
        
        """
        if self._cheb_r is None:
            ...
        raise NotImplementedError
    
    @property
    def vol(self):
        """Compute the volume of the polytope.
        
        """
        if self._vol is None:
            ...
        raise NotImplementedError
    
    def __abs__(self) -> float:
        """Implements the magic method for the absolute value operator `abs` as the volume of the polytope.

        ### FIXME: How smart is it to implement this method? Since `P.vol` is already available.
        
        """
        return self.vol
    
    def __add__(self, other: Polytope | ArrayLike) -> Polytope:
        """Implements the magic method for the addition operator `+` as the Minkowski sum ⊕.

        ### FIXME: This does not actually work with ArrayLike, as that function calls `__array_ufunc__` instead.
        
        Parameters
        ----------
        other : Polytope | ArrayLike
            The other polytope or Numpy array to be added.
            
        Returns
        -------
        Polytope
            The Minkowski sum of the two polytopes.
        
        Raises
        ------
        DimensionError
            If the dimensions of the two polytopes or the vector do not match.

        """
        match other:
            case Polytope():
                if self.n != other.n:
                    raise DimensionError("The dimensions of the two polytopes do not match.")
                return mink_sum(self, other)
            case np.ndarray():
                raise NotImplementedError
            case _:
                raise TypeError("The other object is not a Polytope or Numpy array.")
            
    def __and__(self, other: Polytope) -> Polytope:
        """Implements the magic method for the bitwise AND operator `&` as the intersection operator ∩.

        """
        raise intersection(self, other)
            
    def __bool__(self) -> bool:
        """Implements the magic method for the `bool` operator as the non-empty operator.

        ### FIXME: This is also used in the package 'polytope' to check if the polytope has zero volume: however, I don't actually know if this is a good idea? As we already have the attribute `is_empty` for this purpose. If we implement this, we should also implement the `__len__` method, and the `__int__` method, ut again, this might lead to ambiguity.
        
        Returns
        -------
        bool
            True if the polytope has non-zero volume, False otherwise.
        
        """
        return self.vol > 0
            
    def __contains__(self, point: ArrayLike) -> bool:
        """Implements the magic method for the `in` operator as the `point` inclusion operator x ∈ P.

        ### TODO: Can we also make it such that we can check multiple points?
        
        Parameters
        ----------
        point : ArrayLike
            The point to be checked for inclusion.

        Returns
        -------
        bool
            True if the point is in the polytope, False otherwise.

        Raises
        ------
        DimensionError
            If the dimensions of the point does not match the polytope.

        """
        raise NotImplementedError
    
    def __int__(self) -> int:
        """Implements the magic method for the `int` operator as the number of vertices of the polytope.

        ### FIXME: How smart is it to implement this method? Since `P.verts.shape[0]` is already available.
        
        """
        return self.verts.shape[0]
    
    def __invert__(self) -> Polytope:
        """Implements the magic method for the bitwise NOT operator `~` as the complement of the polytope.

        ### FIXME: I don't know if I actually want this, feels a bit out of place...

        """
        try:
            self.complement = not self.complement
        except AttributeError:
            self.complement = True
        return self
            
    def __le__(self, other: Polytope) -> bool:
        """Implements the magic method `<=` as the subset operator P ⊆ Q.

        ### FIXME: Should we also add `__lt__` and `__ge__` and `__gt__`?
        
        Parameters
        ----------
        other : Polytope
            The other polytope to be compared with.
            
        Returns
        -------
        is_subset : bool
            True if the polytope is a (not necessarily proper) subset of the other polytope, False otherwise.
            
        """
        return is_subset(self, other)
    
    def __mul__(self, factor: float) -> Polytope:
        """Implements the magic method for the multiplication operator `*` as the scaling operator β * P.
        
        Parameters
        ----------
        factor : float
            The scaling factor β.

        Returns
        -------
        poly: Polytope
            The scaled polytope.

        """
        return scale(self, factor)
    
    def __neg__(self) -> Polytope:
        """Scale the polytope by a factor -1, i.e., the negation operator -P. Note that this is scaling around the origin.

        """
        return -1 * self
    
    def __pow__(self, power: int) -> Cube:
        """Implements the magic method for the power operator `**` as the Cartesian product V = S × ... × S (`power` times). Only implemented for 1-d polytopes (i.e., 1-cubes).
        
        Parameters
        ----------
        power : int
            The power of the Cartesian product.

        Returns
        -------
        poly : Cube
            The Cartesian product of the polytope with itself `power` times (an n-dimensionsal cube).

        """
        if self.n != 1:
            raise NotImplementedError("The power operator is only implemented for 1-d polytopes.")
        return power(self, power)
        
    def __str__(self) -> str:
        """Pretty print the polytope."""
        return pretty_print(self)
    
    def __repr__(self) -> str:
        """Debug print the polytope."""
        return f"{self.__class__.__name__}(A.shape={self._F.shape}, b.shape={self._g.shape}, verts.shape={self._verts.shape if self.V_repr else None}, n={self.n}, min_repr={self.min_repr}, is_empty={self.is_empty})"
    
    def __array_ufunc__(self, ufunc, method, other, *args, **kwargs) -> Polytope:
        """Magic method which gets called if the `other` object is a Numpy array. Depending on what operation to perform, we do different stuff.

        ### FIXME: Should we split `*args` instead, because I believe it's a tuple of fixed elements?

        Parameters
        ----------
        *args : tuple
            The arguments passed to the magic method.

        Returns
        -------
        poly : Polytope
            The result of the operation.
        
        """
        match ufunc.__name__:
            case 'add':
                return mink_sum(self, other)
            case 'matmul':
                return mat_mum(other, self)
            case _:
                raise NotImplementedError(f"ufunc is not implemented for Numpy array and operation '{ufunc.__name__}'")
            
    def contains(self, point: ArrayLike) -> bool:
        """Check if the point `point` is in the polytope.

        ### TODO: This method should be able to, efficiently, check multiple points at once.
        
        Parameters
        ----------
        point : ArrayLike
            The point to be checked for inclusion.
        
        Returns
        -------
        is_in : bool
            True if the point is in the polytope, False otherwise.
        
        """
        if point.shape[0] != self.n:
            raise DimensionError("The dimensions of the point do not match the polytope.")
        raise NotImplementedError
            
    def copy(self, type: str = 'deepcopy') -> Polytope:
        """Copies the polytope.
        
        Parameters
        ----------
        type : str
            The type of copy. Default is 'deepcopy'.
        
        Returns
        -------
        poly : Polytope
            The copied polytope.
        
        """
        match type:
            case 'deepcopy':
                raise NotImplementedError
            case 'copy':
                return self
            case _:
                raise ValueError(f"Unrecognized copy type '{type}'")
    
    def bbox(self, in_place: bool = False) -> Cube:
        """Compute the bounding box of the polytope.
        
        Parameters
        ----------
        in_place : bool
            Whether to perform the operation in place. Default is False.

        """
        raise NotImplementedError
    
    def bsphere(self) -> Sphere:
        """Compute the bounding sphere of the polytope.
        
        """
        raise NotImplementedError
    
    def sample(self, seed: int = None, method: str = 'rejection', dist: Callable = None) -> ArrayLike:
        """Sample a point from the polytope.
        
        """
        match method:
            case 'rejection':
                bbox = self.bbox()
                point = bbox.sample(seed, dist)
                while point not in self:
                    point = bbox.sample(seed, dist)
                return point
            case 'hit-and-run':
                raise NotImplementedError
            case _:
                raise ValueError(f"Unrecognized sampling method '{method}'")


class Zonotope(Polytope):
    """The zonotope class which implements a zonotope `zono` = {c + Gz ∈ ℝ^n | ‖z‖_∞ ≤ 1}."""

    def __init__(self, G: ArrayLike, c: ArrayLike):
        """Construct a zonotope with generator matrix `G` and center `c`.
        
        Parameters
        ----------
        G : ArrayLike
            The generator matrix G ∈ ℝ^{n x m} in the zonotope representation.
        c : ArrayLike
            The center of the zonotope.
        
        """
        self.G = G
        self.c = c
        # FIXME: Convert (G, c) into a halfspace representation (A, b)
        super().__init__(self.G + 1, self.c - 1)


class Cube(Polytope):
    """A n-cube is a special type of polytope, where the half-spaces are all aligned with some x_i-x_j plane.
    
    """

    def __init__(self, A = None, b = None):
        super().__init__(A, b)
        self.bounds = np.max(A, axis=0)  ### FIXME: Placeholder

    def sample(self) -> ArrayLike:
        """Sample a point from the cube, which is easier then from a general polytope, and therefore used by refection sampling.
        
        """
        raise NotImplementedError


def verts_to_poly(verts: ArrayLike) -> Polytope:
    """Convert vertices `verts` to a polytope.
    
    Parameters
    ----------
    verts : ArrayLike
        The vertices of the polytope.

    Returns
    -------
    poly : Polytope
        The polytope defined by the vertices.

    """
    raise NotImplementedError


def bounds_to_poly(lb: ArrayLike, ub: ArrayLike) -> Cube:
    """Convert lower and upper bounds `lb` and `ub` to a polytope.
    
    Parameters
    ----------
    lb : ArrayLike
        The lower bounds.
    ub : ArrayLike
        The upper bounds.   

    Returns
    -------
    poly : Polytope
        The polytope defined by the bounds.

    Raises 
    ------
    DimensionError
        If the dimensions of the lower and upper bounds do not match.
    
    """
    ### FIXME: It should be the case that both 'bounds_to_poly([-1, 1])' and 'bounds_to_poly([-1, 1], [-1, 1])' are valid
    raise NotImplementedError


def norm_to_poly(norm: float, n: int, p: float | str = 'inf') -> Polytope | Ellipsoid:
    """Convert the norm {x ∈ ℝ^n | ‖x‖_p ≤ norm} to a polytope (or, a Sphere, if the two-norm is selected).
    
    Parameters
    ----------
    norm : float
        The norm bound.
    n : int
        The dimension of the space.
    p : float | str
        The norm type. Default is 'inf' for the infinity norm.
        
    """
    match str(p):
        case '1':
            raise NotImplementedError
        case '2':
            warnings.warn("Returning a Sphere object instead of a Polytope object as the 2-norm was asked for.")
            return Sphere(np.zeros(n), norm)
        case 'inf':
            raise NotImplementedError
        case _:
            raise ValueError(f"The norm type '{p}' is not recognized.")


def vrepr(poly: Polytope) -> ArrayLike:
    """Convert a polytope `poly` from H-representation to V-representation.
    
    Parameters
    ----------
    poly : Polytope
        The polytope to be converted.

    Returns
    -------
    verts : ArrayLike
        The vertices of the polytope.

    """
    raise NotImplementedError


def hrepr(poly: Polytope) -> tuple:
    """Convert a polytope `poly` from V-representation to H-representation.
    
    Parameters
    ----------
    poly : Polytope
        The polytope to be converted.

    Returns
    -------
    (A, b) : tuple
        The matrix A ∈ ℝ^{p x n} and vector b ∈ ℝ^p in the H-representation {x ∈ ℝ^n | Ax ≤ b}.
    """
    raise NotImplementedError


def power(poly: Polytope, power: int) -> Cube:
    """Compute the Cartesian product of a polytope `poly` with itself `power` times, i.e., V = S × ... × S (`power` times).
    
    Parameters
    ----------
    poly : Polytope
        The polytope to be multiplied.
    power : int
        The power of the Cartesian product.
        
    Returns
    -------
    poly : Cube
        The Cartesian product of the polytope with itself `power` times (an n-dimensionsal cube).

    Examples
    --------
    Examples should be written in doctest format, and should illustrate how
    to use the function.

    >>> P = bounds_to_poly([-1, 1]) ** 3
    >>> print(repr(P))
    Cube(A.shape=(6, 3), b.shape=(6,), verts.shape=(8, 3), n=3)
    
    """
    raise NotImplementedError


def scale(poly: Polytope, factor: float, center: str = 'origin') -> Polytope:
    """Scale the polytope P = `poly` by a factor β = `factor` such that W = {β * x ∈ ℝ^n | x ∈ P}. Note that by default, the scaling is performed around the origin.

    ### FIXME: Should we make this a method of the polytope self instead? And have the method `P.scale(a, in_place=True)`?
    
    Parameters
    ----------
    poly : Polytope
        The polytope to be scaled.
    factor : float
        The scaling factor.
    origin : str
        The origin of the scaling. Default is 'origin'.
    
    Returns
    -------
    poly : Polytope
        The scaled polytope.
    
    """
    match center:
        case 'origin':
            raise NotImplementedError
        case 'cheb_c':
            raise NotImplementedError
        case _:
            raise ValueError(f"Unrecognized center '{center}'")
        

def mat_mum(array: ArrayLike, poly: Polytope) -> Polytope:
    """Implements the magic method for the matrix multiplication operator `@` as the linear transformation A P.

    ### FIXME: What if we want to implement P @ A? This should also be possible, right?
    
    """
    raise NotImplementedError


def extreme(poly: Polytope) -> ArrayLike:
    """Finds the extreme points of a polytope `poly`, i.e., the vertices `verts` of the polytope.

    ### FROM: polytope package
    ### FIXME: Should we move this to the class Polytope itself, i.e., a method rather then a function?
    
    Parameters
    ----------
    poly : Polytope
        The polytope to be converted.
    
    Returns
    -------
    verts : ArrayLike
        The vertices of the polytope.
    
    """
    raise NotImplementedError


def enum_int_points(poly: Polytope) -> ArrayLike:
    """Enumerate the integer points in a polytope `poly`, i.e., a lattice.
    
    ### TODO: Look into if this has cryptographic implementations? 
    
    """
    raise NotImplementedError


def convex_hull(points: ArrayLike) -> ArrayLike:
    """Compute the convex hull of a set of points `points`.
    
    Parameters
    ----------
    points : ArrayLike
        The points for which a convex hull is to be formed.
    
    Returns
    -------
    hull : ArrayLike
        The convex hull of the points.
    
    """
    raise NotImplementedError


def support(poly: Polytope, direction: ArrayLike) -> ArrayLike:
    """Compute the support function of a polytope `poly` in a given direction `direction`.

    Parameters
    ----------
    poly : Polytope
        The polytope.
    direction : ArrayLike
        The direction.
    
    Returns
    -------
    support : ArrayLike
        The support function of the polytope in the direction.

    References
    ----------
    [1] I. Kolmanovsky, E.G. Gilbert. (1998). "Theory and computation of disturbance invariant sets for discrete-time linear systems," Mathematical Problems in Engineering, vol. 4, pp. 317-367
    
    """
    raise NotImplementedError


def mink_sum(poly_1: Polytope, poly_2: Polytope) -> Polytope:
    """Compute the Minkowski sum `poly_1` ⊕ `poly_2` of two polytopes.
    
    Parameters
    ----------
    poly_1 : Polytope
        The first polytope.
    poly_2 : Polytope
        The second polytope.
    
    Returns
    -------
    poly : Polytope
        The Minkowski sum of the two polytopes.
    
    """
    raise NotImplementedError


def pont_diff(poly_1: Polytope, poly_2: Polytope) -> Polytope:
    """Compute the Pontryagin difference `poly_1` ⊖ `poly_2` of two polytopes.
    
    Parameters
    ----------
    poly_1 : Polytope
        The first polytope.
    poly_2 : Polytope
        The second polytope.
    
    Returns
    -------
    poly : Polytope
        The Pontryagin difference of the two polytopes.
    
    """
    raise NotImplementedError


def intersection(poly_1: Polytope, poly_2: Polytope) -> Polytope:
    """Compute the intersection V = `poly_1`, W = `poly_2`, V ∩ W = {x ∈ ℝ^n | x ∈ V, x ∈ W} of two polytopes (which 
    is guaranteed to be a convex polytope itself). How long can I actually make the line in this docstring? Does it matter for the annotation type hinting which is displayed? It seems like the breaks are actually automatic.
    
    Parameters
    ----------
    poly_1 : Polytope
        The first polytope.
    poly_2 : Polytope
        The second polytope.
    
    Returns
    -------
    poly : Polytope
        The intersection of the two polytopes.
    
    """
    raise NotImplementedError


def projection(points: Polytope | ArrayLike, proj: list | Subspace, keep_dims: bool = True) -> Polytope:
    """Compute the projection of a polytope or vector `points` onto a subspace `proj` as T = Proj(V, z).
    
    """
    raise NotImplementedError


def is_subset(poly_1: Polytope, poly_2: Polytope) -> bool:
    """Check if the polytope `poly_1` is a subset of the polytope `poly_2`, i.e., P ⊆ Q.
    
    """
    raise NotImplementedError


# ------------ SUBSPACE ------------


class Subspace():
    """A subspace class which implements a subspace `sub` = {x ∈ ℝ^n | x ∈ range(E)}, where E is the basis of the subspace.
    
    """

    def __init__(self, basis: ArrayLike):
        self.basis = basis
        self.dim = None   ### FIXME: Placeholder
        self.min_repr = None
        self.is_empty = None  ### NOTE: Here, 'empty' means the subspace just contains the zero vector, i.e., `self` = {0}

    @property
    def perp(self) -> Subspace:
        """Compute the orthogonal complement V^⊥ of the subspace V = `self`.
        
        """
        return Subspace(sp.linalg.null_space(self.basis.T).T)
    
    def __mod__(self, other: Subspace) -> QuotientSpace:
        """Implements the magic method `%` as the quotient space V / W of two subspaces V = `self` and W = `other`. Note that this requires that W ⊆ V.
        
        """
        return QuotientSpace(self, other)

    def __invert__(self) -> Subspace:
        """Implements the magic method `~` as the orthogonal complement V^⊥ of the subspace V = `self`.

        ### FIXME: Is `__invert__` the right method/name for this?
        
        """
        raise NotImplementedError
    
    def __repr__(self) -> str:
        """Debug print the subspace."""
        return f"{self.__class__.__name__}(basis.shape={self.basis.shape}, n={self.n}, dim={self.dim}, min_repr={self.min_repr}, is_empty={self.is_empty})"

    def reduce(self) -> Subspace:
        """Reduce the basis `basis` of the subspace to a minimal basis.
        
        """
        ### FIXME: Check for linear independence using a rank condition
        raise NotImplementedError
    
    def orthonormal(self, in_place: bool = True) -> None | Subspace:
        """Compute an orthonormal basis for the subspace `self`.

        ### FIXME: When should a method just act on itself, and when should it return a new object?
        
        """
        raise NotImplementedError
    

def QuotientSpace():
    """Class which implements a quotient space V / R = {V + r | r ∈ R}.

    ### FIXME: Should I also make the class `AffineSubset`, as the elements of the quotient space are affine subspaces?

    """

    def __init__(self, V: Subspace, R: Subspace):
        """Construct a quotient space V / R = {V + r | r ∈ R}.

        Parameters
        ----------
        V : Subspace
            The subspace V.
        R : Subspace
            The subspace R. Note that R ⊆ V.

        """
        raise NotImplementedError


def subs_add(subs_1: Subspace, subs_2: Subspace) -> Subspace:
    """Compute the addition, or *direct sum*, of two subspaces `subs_1` + `subs_2`.
    
    """
    return (subs_1.E + subs_2.E).reduce()


def angle(subs_1: Subspace, subs_2: Subspace) -> float:
    """Compute the angle between two subspaces `subs_1` and `subs_2`.
    
    """
    raise NotImplementedError


def quotient(subs_1: Polytope, subs_2: Polytope) -> QuotientSpace:
    """Compute the quotient space Q = V / W of two subspaces V = `subs_1` and W = `subs_2`. Note that this requires that W ⊆ V.
    
    """
    raise NotImplementedError


# ------------ ELLIPSOID ------------


class Ellipsoid():
    """The ellipsoid class which implements an ellipsoid `ell` = {x ∈ ℝ^n | (x - c)^T P^-1 (x - c) ≤ α}.
    
    """

    def __init__(self, P: ArrayLike, c: ArrayLike, alpha: float = 1):
        """Constructor for the ellipsoid class
        
        Parameters
        ----------
        P : ArrayLike
            The positive semi-definite matrix P ∈ ℝ^{n x n} in the ellipsoid representation.
        c : ArrayLike
            The center of the ellipsoid.
        alpha : float
            The scaling factor α.
        
        """
        self.P = P
        self.c = c
        self.alpha = alpha
        self.n = P.shape[0]
        self._R = None  ### NOTE: Radii of the ellipsoid, i.e., the semi-minor and major axis
        self._theta = None
        self._vol = None

    @property
    def theta(self) -> ArrayLike:
        """Compute the angle of the ellipsoid.
        
        """
        if self._theta is None:
            ...
        return self._theta
    
    @property
    def R(self) -> ArrayLike:
        """Compute the radii of the ellipsoid.
        
        """
        if self._R is None:
            ...
        return self._R
    
    @property
    def vol(self) -> float:
        """Compute the volume of the ellipsoid.
        
        """
        if self._vol is None:
            # FROM: https://math.stackexchange.com/questions/226094/measure-of-an-ellipsoid
            self._vol = (np.pi ** (self.n / 2) / np.math.gamma(self.n / 2 + 1)) / np.sqrt(np.linalg.det(self.P))
        return self._vol
    
    def bbox(self) -> Cube:
        """Compute the bounding box of the ellipsoid.
        
        """
        ### FROM: https://math.stackexchange.com/questions/3926884/smallest-axis-aligned-bounding-box-of-hyper-ellipsoid
        P_inv = np.linalg.inv(self.P)
        ### FIXME: Why is this the one that works? It seems to me that Q = Q / a should work, but it doesn't? Why does the sqrt mess thing up?
        lb, ub = self.c - self.a * np.sqrt(np.diag(P_inv)), self.c + self.a * np.sqrt(np.diag(P_inv))
        return bounds_to_poly(lb, ub)


class Sphere(Ellipsoid):
    """A sphere is a special type of ellipsoid where the matrix P is the identity matrix.
    
    """

    def __init__(self, c: ArrayLike, radius: float = 1):
        """Construct a sphere with center `c` and radius `radius`.
        
        Parameters
        ----------
        c : ArrayLike
            The center of the sphere.
        radius : float
            The radius of the sphere.
        
        """
        self.radius = radius
        super().__init__(np.eye(c.shape[0]), c, radius)


def normal_to_elps(mean: ArrayLike, Sigma: ArrayLike, a: float) -> Ellipsoid:
    """Convert a normal distribution N(`mean`, `Sigma`) to an ellipsoid of `a` times the standard deviation.
    
    Parameters
    ----------
    mean : ArrayLike
        The mean of the normal distribution.
    Sigma : ArrayLike
        The covariance matrix of the normal distribution. Must be positive semi-definite, i.e., Sigma ≽ 0 
    a : float
        The scaling factor of the standard deviation, i.e, the number of times the ellipsoid should include this.

    Returns
    -------
    elps: Ellipsoid
        The ellipsoid representing the normal distribution.

    Examples
    --------

    Consider the normal distribution with mean = 0 and Sigma = ... Now, if we want the 95% confidence interval, we note that this is equal to about 3 times the standard deviation. Therefore, we can write:
        
    """
    raise NotImplementedError


# ------------ CONTROL -------------


def mrpi(A: ArrayLike, B: ArrayLike, K: ArrayLike, W: Polytope, method: str = 'lazar', s_max: int = 10) -> Polytope:
    """Compute the minimal robustly positive invariant set F_∞ = ⊕_{i=0}^{∞} (A + B K)^i W. See also [1, Algorithm 1].

    References
    ----------
    [1] S.V. Rakovic, E.C. Kerrigan, K.I. Kouramas, D.G. Mayne. (2005, March). "Invariant approximations of the minimal robust positively Invariant set," IEEE Transactions on Automatic Control, vol. 50, no. 3, pp. 406-410
    [2] M.S. Darup, D. Teichrib. (2019, June). "Efficient computation of RPI sets for tube-based robust MPC," 2019 18th European Control Conference (ECC), pp. 325-330
    
    """
    match method:
        case 'rakovic':
            W_s = W.copy()
            for s_star in range(1, s_max + 1):
                W_s = A @ W_s
                if W_s <= W:
                    break
            func = lambda a, W, Z: a * Z <= W
            alpha_star, F = bisect(func, range=(0, 1), args=(W, W_s)), W.copy()
            for s in range(1, s):
                F += np.linalg.matrix_power(A, s) @ W
            F *= 1 / (1 - alpha_star)
            return F, s_star, alpha_star
        case 'darup':
            raise NotImplementedError
        case _:
            raise ValueError(f"Unrecognized method '{method}'")


def mpis(A: ArrayLike, X: Polytope, iter_max: int = 10) -> Polytope:
    """Compute the maximal invariant polytope contained in `X` under the dynamics `A`.

    References
    ----------
    [1] D. Janak, B. Açikrneşe. (2019, July). "Maximal Invariant Set Computation and Design for Markov Chains," 2019 American Control Conference (ACC), pp. 1244-1249

    """
    ### FIXME: Probably, this implementation is all wrong
    A_inv = pre_img(A)
    V = X.copy()
    for iter in range(iter_max):
        X_pre = np.linalg.matrix_power(A_inv, iter + 1) @ V
        if X_pre <= X:
            break
        V &= X_pre
    return V


def feas_reg_mpc() -> Polytope:
    """Compute the feasible region of a model predictive control (MPC) problem.
    
    """
    raise NotImplementedError


def max_ctrl_inv_subs(A: ArrayLike, B: ArrayLike, C: ArrayLike) -> Subspace:
    """Computes the maximal (A,B)-controlled invariant subspace `V_star` contained in the kernel of `C`, i.e., V_star ⊆ ker(C). Note that maximal here refers to the unique subspace `V_star` with the largest dimension, i.e., V* = max_{V is (A, B)-controlled invariant, V ⊆ ker(C)} dim(V). See [1, Algorithm 4.1.2].

    Parameters
    ----------
    A : ArrayLike
        The state matrix A.
    B : ArrayLike
        The input matrix B.
    C : ArrayLike
        The output matrix C.

    Returns
    -------
    V_star : Subspace
        The maximal (A,B)-controlled invariant subspace contained in the kernel of C.

    References
    ----------
    [1] G. Basile, G. Marro. (1992). "Controlled and conditioned invariants in linear system theory," Prentice Hall

    """
    raise NotImplementedError


def min_cond_inv_subs(A: ArrayLike, B: ArrayLike, C: ArrayLike) -> Subspace:
    """Computes the minimal (A,C)-conditioned invariant subspace `S_star` containing the image of `B`, i.e., S_star ⊇ Im(B). Note that minimal here refers to the unique subspace `S_star` with the smallest dimension, i.e., S* = max_{S is (A, C)-conditioned invariant, S ⊇ Im(B)} dim(S). See [1, Algorithm 4.1.1].

    Parameters
    ----------
    A : ArrayLike
        The state matrix A.
    B : ArrayLike
        The input matrix B.
    C : ArrayLike
        The output matrix C.

    Returns
    -------
    S_star : Subspace
        the minimal (A,C)-conditioned invariant subspace containing the image of B.

    References
    ----------
    [1] G. Basile, G. Marro. (1992). "Controlled and conditioned invariants in linear system theory," Prentice Hall

    """
    raise NotImplementedError


def max_reach_subs(A: ArrayLike, B: ArrayLike, C: ArrayLike) -> Subspace:
    """Computes the maximal reachability subspace R* = V* ∩ S*
    
    """
    raise NotImplementedError


# ------------ UTILS ------------


class DimensionError(Exception):
    """Exception raised for errors in the input dimensions.
    
    """

    def __init__(self, message: str):
        """Constructor for the exception.

        Attributes
        ----------
        message : str
            Explanation of the error.
        
        """
        self.message = message
        super().__init__(self.message)


def plot(obj: Polytope | Ellipsoid | Subspace, ax: plt.Axes = None) -> list[plt.Figure, plt.Axes]:
    """Method to plot either a polytope, a ellipsoid, or a subspace.
    
    """
    raise NotImplementedError


def bisect(func: Callable, range: tuple, args: tuple) -> float:
    """Bisection algorithm to find the root of a function `func` in a given `range` with `args`.
    
    """
    raise NotImplementedError


def pre_img(A: ArrayLike) -> ArrayLike:
    """Compute the pre-image of a matrix `A`. This is the inverse of the matrix if invertible, and the combination of the pseudo-inverse and the null-space if not.
    
    """
    raise NotImplementedError


def pretty_print(obj: Polytope | Ellipsoid | Subspace) -> str:
    """Pretty print the object `obj`. Used as a helper function for the `__str__` method, as this method can get really verbose.
    
    """
    return repr(obj)  ### FIXME: Placeholder


def is_in(obj_1: ArrayLike | Polytope | Ellipsoid | Subspace | cvx.Variable | cvx.atoms.affine.index.index, obj_2: Polytope | Ellipsoid | Subspace) -> bool | cvx.Constraint:
    """Check if a point `x` is in the object `obj`. If `x` is a cvxpy variable, return a constraint instead.
    
    """
    match obj_1:
        case cvx.Variable() | cvx.atoms.affine.index.index():
            match obj_2:
                case Polytope():
                    return obj_2.F @ obj_1 <= obj_2.g
                case Ellipsoid():
                    return cvx.norm(obj_1 - obj_2.c, obj_2.P) <= obj_2.alpha
                case Subspace():
                    ### FIXME: This should be some sort of equality constraint? Of should it be a polyhedron in half-space representation?
                    raise NotImplementedError
                case _:
                    raise ValueError(f"Unrecognized obj_2 type '{obj_2.__class__.__name__}'")
        case np.ndarray():
            return obj_1 in obj_2
        case Polytope() | Ellipsoid() | Subspace():
            return obj_1 <= obj_2
        case _:
            raise ValueError(f"Unrecognized obj_1 type '{obj_1.__class__.__name__}'")


def main():
    """Temporary file, to test some functionality
    
    """

    A = np.array([[1, 0], [0, 1]])
    b = np.array([1, 1])
    poly = Polytope(A, b)

    print(poly)
    print(repr(poly))

    # Here we can also write MPC
    X, U = cvx.Variable((2, 11)), cvx.Variable((2, 10))
    cost, const = [], []
    for k in range(10):
        const += [is_in(X[:, k + 1], poly)]
    

if __name__ == "__main__":
    main()