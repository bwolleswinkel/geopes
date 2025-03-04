"""This is for now the main class, as long as the structure of the project is not fixed (i.e., no package on PyPI is available yet. ).

© Bart Wolleswinkel, 2025. Distributed under a ... license.

"""

from __future__ import annotations
import warnings

import numpy as np
import scipy as sp
import control as ct
from numpy.typing import ArrayLike


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


class Polytope:
    """The polytope class which implements a polytope `poly` = {x ∈ ℝ^n | Ax ≤ b}.
    This class emulates a numerical type, and has a H-representation (half-space 
    representation) and V-representation (vertex representation).

    Methods
    -------
    __init__(A, b)
        Initialize the Polytope object.
    
    """

    def __init__(self, A: ArrayLike = None, b: ArrayLike = None):
        """Initialize a Polytope object (see class for description) based on either a 
        H-representation or a V-representation.

        ### FIXME: Maybe we should just make this a H-space representations? And make a method verts_to_poly instead?
        
        Parameters
        ----------
        A : ArrayLike
            The matrix A ∈ ℝ^{p x n} in the H-representation {x ∈ ℝ^n | Ax ≤ b}.
        b : ArrayLike
            The vector b ∈ ℝ^p in the H-representation {x ∈ ℝ^n | Ax ≤ b}.
        
        """
        ### FIXME: This is not implemented yet
        self.H_repr = True
        self.V_repr = False
        self._A = A
        self._b = b
        self._verts = None
        self._cheb_c = None
        self._cheb_r = None
        self._vol = None
        self.n = A.shape[1]  ### FIXME: Placeholder
        self.min_rep = None 
        self.is_empty = False  ### NOTE: A polytope can have zero volume but still be non-empty

    @property
    def A(self):
        if not self.H_repr:
            self._A, self._b = hrepr(self)
        return self._A
    
    @property
    def b(self):
        if not self.H_repr:
            (self._A, self._b), self.H_repr = hrepr(self), True
        return self._b
    
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
    
    def __add__(self, other: Polytope | ArrayLike) -> Polytope:
        """Implements the magic method for the addition operator `+` as the Minkowski sum ⊕.
        
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
            
    def __le__(self, other: Polytope) -> bool:
        """Implements the magic method `<=` as the subset operator P ⊆ Q.

        ### FIXME: Should we also add `__lt__` and `__ge__` and `__gt__`?
        
        Parameters
        ----------
        other : Polytope
            The other polytope to be compared with.
            
        Returns
        -------
        bool
            True if the polytope is a subset of the other polytope, False otherwise.
            
        """
        return is_subset(self, other)
    
    def __str__(self) -> str:
        """Pretty print the polytope."""
        return f"{self.__class__.__name__} in ℝ^{self.n}"  ### FIXME: Placeholder
    
    def __repr__(self) -> str:
        """Debug print the polytope."""
        return f"{self.__class__.__name__}(A.shape={self._A.shape}, b.shape={self._b.shape}, verts.shape={self._verts.shape if self.V_repr else None}, n={self.n})"
    
    def bbox(self) -> Cube:
        """Compute the bounding box of the polytope.
        
        """
        raise NotImplementedError
    
    def bsphere(self) -> Sphere:
        """Compute the bounding sphere of the polytope.
        
        """
        raise NotImplementedError


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


def scale(poly: Polytope, factor: float, center: str = 'origin') -> Polytope:
    """Scale the polytope P = `poly` by a factor β = `factor` such that W = {β * x ∈ ℝ^n | x ∈ P}. Note that by default, the scaling is performed around the origin.
    
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


class Subspace():
    """A subspace class which implements a subspace `sub` = {x ∈ ℝ^n | x ∈ range(E)}, where E is the basis of the subspace.
    
    """

    def __init__(self, E: ArrayLike):
        self.E = E

    def reduce(self) -> Subspace:
        """Reduce the basis E of the subspace to a minimal basis.
        
        """
        ### FIXME: Check for linear independence using a rank condition
        raise NotImplementedError


def subs_add(subs_1: Subspace, subs_2: Subspace) -> Subspace:
    """Compute the addition, or *direct sum*, of two subspaces `subs_1` + `subs_2`.
    
    """
    return (subs_1.E + subs_2.E).reduce()


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


def main():
    """Temporary file, to test some functionality
    
    """

    A = np.array([[1, 0], [0, 1]])
    b = np.array([1, 1])
    poly = Polytope(A, b)

    print(poly)
    print(repr(poly))
    

if __name__ == "__main__":
    main()