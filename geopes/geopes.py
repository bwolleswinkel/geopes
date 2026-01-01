"""This is for now the main class, as long as the structure of the project is not fixed (i.e., no package on PyPI is available yet. ).

© Bart Wolleswinkel, 2025. Distributed under a ... license.

### TODO: Check out package 'polytope', https://tulip-control.github.io/polytope/
### TODO: Check out package 'pypolycontain', https://pypolycontain.readthedocs.io/en/latest/index.html
### TODO: Check out repository 'pytope', https://github.com/heirung/pytope
### TODO: Check out paper "LazySets.jl: Scalable Symbolic-Numeric Set Computations" Juliacon (2021), https://juliacon.github.io/proceedings-papers/jcon.00097/10.21105.jcon.00097.pdf#page1
### TODO: Check out d(ouble)under/numerical Python methods "3.3.8. Emulating numeric types", https://docs.python.org/3/reference/datamodel.html
### TODO: Check out the repository "Remote Tube-based MPC for Tracking Over Lossy Networks" (which has multiple implementations of MRPI sets), https://github.com/EricssonResearch/Robust-Tracking-MPC-over-Lossy-Networks/blob/main/src/LinearMPCOverNetworks/utils_polytope.py
### TODO: Check out documentation 'Standard operators as functions', https://docs.python.org/3/library/operator.html
### TODO: Check out `pycvxset: A Python package for convex set manipulation`, https://arxiv.org/html/2410.11430v1
### TODO: Check out `MPT3` toolbox, https://www.mpt3.org/pmwiki.php/Main/HowTos
### TODO: Check out `PnPMPC-TOOLBOX', from `http://sisdin.unipv.it/pnpmpc/pnpmpc.php'
    - Check out e_MRPI to go from a Polytopic to an outer, simpler approximation
### TODO: Check out `Basic Properties of Convex Polytopes, Henk Martin
### TODO: Check out `contrained zonotopes', and `hybrid zonzotpes' (zonoLAB), https://github.com/ESCL-at-UTD/zonoLAB 
### TODO: Check out `Ellipsotopes', https://ieeexplore-ieee-org.tudelft.idm.oclc.org/stamp/stamp.jsp?arnumber=9832489

### TODO: Implement `Polyhedra' as a base class, and let `Polytope'` inherits from it.
### TODO: Implement `Cones' (also for ETC), https://en.wikipedia.org/wiki/Convex_cone
    - Gabriel: Hmm might be tricky, as there's many definition; he used quadratic cones, of the form x^T Q x < 0
    - Gabriel: I actually have some checks for (quadratic) cones, might be interesting to look at

### TODO: Check out `Ellipsoidal Calculus for Estimation and Control', https://link-springer-com.tudelft.idm.oclc.org/book/9780817636999
### TODO: Check out `Ellipsoidal Toolbox (ET)', https://www.mathworks.com/matlabcentral/fileexchange/21936-ellipsoidal-toolbox-et

### TODO: Gabriel: Check ellipsoidal containment with LMIs
### TODO: Allow for ellipsoids to be degenerate (also INFINITE measure, so ellipsoid), cite Ross's paper (check Gabriel STC)
    - Think about the matrix inverse, degeneracy of two types
### TODO: Gabriel: check out linear programming, simplex: Ax = b, x >= 0 (neither H- nor V-representation: called canonical form)
    - For degeneracy of polytope, maybe have a FLAG fro degeneracy, so that it keeps track
    - Maybe, just think of all the possible unboundedness, so no restrictions on that 

### TODO: Gabriel: Look at 'qhull' in Python

### TODO: Check out the YouTube video series from 'mathapptician' on convex polytopes: they also talk/give examples of facet and vertex enumeration algorithms
# FROM: https://www.youtube.com/watch?v=5LzsqZmwQiQ

### TODO: Look at conventions for 'set' class in python, on when operations should be methods of functions, or when they should be functions of the package

### TODO: Check out "Linear Matrix Inequalities in System and Control Theory," Boyd et al. (1994) specifically 5.2.1 Smallest invariant ellipsoid containing a polytope, and 5.2.2 Largest invariant ellipsoid contained in a polytope

### TODO: Check out "Algorithms for Ellipsoids," Pope (2008), written in Fortran

### TODO: Check out the software package 'polymake'
# FROM: https://polymake.org/doku.php/start

### TODO: Check out the Julia library "Polyhedra"
# FROM: https://juliapolyhedra.github.io/Polyhedra.jl/stable/generated/Minimal%20Robust%20Positively%20Invariant%20Set/  # nopep 8
# FROM: https://www.youtube.com/watch?v=YuxJtvgg6uc  # nopep8

### TODO: Check out the package pycddlib, definitly what we need
### TODO: Check out the cdd package on 'Komei's Software Page' and 'cdd, cddplus and cddlib homepage'
# FROM: https://pycddlib.readthedocs.io/en/stable/quickstart.html
# FROM: https://people.inf.ethz.ch/fukudak/soft/soft.html

### TODO: Check out `ELL_LIB`, a FORTRAN library for ellipsoidal calculus
# FROM: https://tcg.mae.cornell.edu/ELL_LIB/

### TODO: Check out the following video about maximal area inscribed rectangles in a ellipsoid, and about computing maximal points on an ellipsoid.
# FROM: # FROM: https://www.youtube.com/watch?v=1ZfKzv6Bi5k
# FROM: https://math.stackexchange.com/questions/995312/maximal-points-on-an-n-dimensional-ellipsoid (computing a maximal element p of an ellipsoid in a given direction)

### TODO: Check out the website 'zenodo' for having a doi for the package

### FIXME: What convention do we want for seperate constructors? Do we want to use classmethods, like pandas? So we can have:
    poly = geo.poly_from_verts(verts)  # More similar to numpy?
    poly = geo.verts_to_poly(verts)
    poly = geo.verts_2_poly(verts)
    poly = geo.verts2poly(verts)
    poly = geo.Polytope.from_verts(verts)  # @classmethod, used in pandas
which one is best?
# FROM: https://www.reddit.com/r/learnpython/comments/iy3rdl/classmethod_when_to_use_it/

### FIXME: Should we use abbreviations, and when? For instance, we can have:
    V_star = geo.control.min_inv_sub(...)  # NOTE: NumPy also uses the full name, i.e., `numpy.linalg.inv` and `numpy.random.rand`
    V_star = geo.ctrl.min_inv_subspace(...)  # Should we use abbreviations only for methods?

### FIXME: Should we also have functions as follows:
    X = geo.poly(F, g)  # Instead of `geo.Polytope(F, g)`
    X = geo.zono(G, c)  # Instead of `geo.Zonotope(G, c)`
    E = geo.ellipsoid(A, b)  # Instead of `geo.Ellipsoid(A, b)`
    # This is similar to how `numpy` does it, i.e., `np.array(...)`, or `control.ss(A, B, C, D)` instead of `control.StateSpace(A, B, C, D)`.

### FIXME: Should we name the package 'geopes' or 'pespy'? Geo stands for geometry, but could also be confused with geography. Examples would be:
    X = geo.poly(F, g)  # Geo for geometry
    X = pp.poly(F, g)  # Pes for polytope, ellipsoid, subspace

### FIXME: As Gabriel said, storing vertices in high dimesnions, e.g., 1000, is prohibitive as they grow exponentially (the number of them)

### TODO: In a `config.py` file, apart from `RTOL` and `ATOL`, we need to also have a function `set_print_options`, which sets the print options for numpy, scipy, and maybe also for cvxpy (if possible). Also, maybe add a `verbose` flag in there as well? And have everything use `config.VERBOSE`!

### TODO: For lazy properties, check out https://stackoverflow.com/questions/3012421/python-memoising-deferred-lookup-property-decorator, they have a `@lazy_property` decorator

### TODO: Also check out the walrus operator `:=`, https://realpython.com/python-walrus-operator/, which can be really clean for local variables and flags in loops

### TODO: Check out the following documentation on how to properly overwrite the ufunc method for numpy arrays, by setting `__array_ufunc__ = None` Also has a full list of operators and names!
# FROM: https://docs.scipy.org/doc/numpy-1.13.0/neps/ufunc-overrides.html

### TODO: For the `reduce` or `minimal` representation, where we remove redundant vertices or half-spaces, check out the following tow sources:
- # FROM: https://www.cs.mcgill.ca/~fukuda/soft/polyfaq/node23.html#polytope:Vredundancy2
- # FROM: https://www.cs.mcgill.ca/~fukuda/soft/polyfaq/node24.html

"""

from __future__ import annotations
from abc import ABC, abstractmethod
import copy
from typing import Any, Callable, Literal
import warnings

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt  # TODO: Make this an optional dependency perhaps?
from numpy.typing import ArrayLike
import cvxpy as cvx  ### NOTE: I want to make this an optional dependency, as it's not strictly needed for any core functionality


class ConvexRegion(ABC):
    """A base class for convex regions, which can be inherited by other classes such as Polytope, Ellipsoid, etc.
    
    """
    
    @property
    @abstractmethod
    def n(self):
        """The dimension of the space in which the convex region lives."""
        pass

    @property
    @abstractmethod
    def dim(self):
        """The dimension of the convex region itself, which can be different (i.e., equal or smaller) from the ambient space dimension."""
        pass

    @property
    @abstractmethod
    def vol(self):
        """The n-dimensional volume (Lebesgue measure) of the convex region."""
        pass


class Polytope(ConvexRegion):
    """The polytope class which implements a polytope `poly` = {x ∈ ℝ^n | Fx ≤ g}.
    This class emulates a numerical type, and has a H-representation (half-space 
    representation) and V-representation (vertex representation).

    ### FIXME: We need to do proper renaming: as (A, B, C) describes a system, we must use (F, g) for the polytope.

    Methods
    -------
    __init__(F, g)
        Initialize the Polytope object.
    
    """

    def __init__(self, A: ArrayLike | None, b: ArrayLike | None, verts: ArrayLike | None = None, *, A_eq: ArrayLike | None = None, b_eq: ArrayLike | None = None, rays: ArrayLike | None = None, min_repr: bool = True) -> None:
        """Initialize a Polytope object (see class for description) based on either a H-representation or a V-representation.

        ### NOTE: Using the `*` in the argument list forces the user to use keyword arguments for `F_ineq` and `g_ineq`, which makes it clear that these are not positional arguments.

        ### FIXME: Maybe we should just make this a H-space representations? And make a method verts_to_poly instead?

        ### FIXME: We should also allow for polyhedrons as 'degenerate' polytopes, i.e., those polytopes with infinite volume.

        ### FIXME: We should have three initializers: _init_H_repr (i.e., `gp.poly(A, b)`), _init_V_repr (`gp.poly(V, rays=R)), and _init_empty (`gp.poly(n=1)`). So either 2, 1, or 0 positional arguments, and the rest keyword arguments.
        
        Parameters
        ----------
        F : ArrayLike
            The matrix F ∈ ℝ^{p x n} in the H-representation {x ∈ ℝ^n | Fx ≤ g}.
        g : ArrayLike
            The vector g ∈ ℝ^p in the H-representation {x ∈ ℝ^n | Fx ≤ g}.

        Examples
        --------
        >>> import numpy as np
        >>> import geopes as gp

        import modules
        
        >>> A = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
        >>> b = np.array([1, 1, 1, 1])
        >>> V = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]])

        vertices representation

        >>> poly_1 = gp.poly(A, b)
        >>> poly_2 = gp.Polytope(V)
        >>> poly_3 = gp.poly(verts=V)
        
        Note that the above are all valid constructors for a polytope.

        Example of a link: `name of link<https://arxiv.org/abs/1803.08669>`__.
        
        """
        self.H_repr: bool = True   ### FIXME: Should this be `is_H_repr` instead?
        self.V_repr: bool = False   ### FIXME: Is this not redundant, as this will always coincide with self.verts being not None?
        self._A: ArrayLike | None = A
        self._g: ArrayLike | None = b
        self._F_eq: ArrayLike | None = None
        self._g_eq: ArrayLike | None = None
        self._verts: ArrayLike | None = None
        ### FIXME: Also look at the 'centroid', which is different from the Chebyshev center (see `pytope`)
        self._cheb_c: ArrayLike | None = None
        self._cheb_r: ArrayLike | None = None
        self._vol: float | None  = None
        self._com: ArrayLike | None = None   ### FIXME: Center of mass of the polytope. This should be called the centroid, right?
        self._n: int = A.shape[1]  ### FIXME: Placeholder
        self.is_degen: bool | None = None  ### FIXME: Not that we have two types of degeneracy, whenever self.vol = 0 (type I degeneracy), or when self.vol = np.inf (type II degeneracy). Should `self.is_degen` therefore be a flag or give two different values?
        ### FIXME: Also look into "Representation of unbounded polytopes" from wikipedia, about vertex representation with 'bounding rays', seems quite interesting. This is yet another representation of polytopes, which is neither H- nor V-representation.
        self.is_min_repr: bool | None = None  ### FIXME: This should also maybe be the flag `is_min_repr` instead?
        self.is_empty: bool | None = False  ### NOTE: A polytope can have zero volume but still be non-empty
        ### FIXME: This method should dispatch to either private method `_init_H_repr`` or `_init_V_repr`` based on the input arguments
        # TEMP: This is just a placeholder
        #
        args = (A, b)
        kwargs = {'F_eq': A_eq, 'g_eq': b_eq}
        #
        if len(args) == 2:
            if 'V' in kwargs:
                raise ValueError("Cannot provide both (F, g) and V.")
            ...
            ### FIXME: Where should we check that no ADDITIONAL keyword arguments are given, i.e., kwargs should be empty after we have 'used' all keywords, and we should raise an error otherwise. Should we do that here, in dispatch, or in the private methods?
        elif len(args) == 1:
            if 'F' in kwargs or 'g' in kwargs:
                raise ValueError("Cannot provide both V and F or g.")
            ...
        elif len(args) == 0:
            if 'F' in kwargs and 'g' in kwargs:
                if 'V' in kwargs:
                    raise ValueError("Cannot provide both (F, g) and V.")
                ...
            elif 'V' in kwargs:
                if 'F' in kwargs or 'g' in kwargs:
                    ### FIXME: I know this is redundant, but I want to keep the structure similar to above for now
                    raise ValueError("Cannot provide both V and F or g.")
                ...
        ### NOTE: We also need an 'empty' constructor, i.e., Polytope(), because for instance, for a Zonotope, do we also want to have lazy setters and getters? In that case, both the H- and V-representation are None, and only when the user tries to access them, we compute them.
        else:
            raise ValueError("Invalid arguments. Use Polytope(F, g) or Polytope(V=V).")
        
            

    def _init_H_repr(self, A: ArrayLike, b: ArrayLike, **kwargs):
        """Initialize the Polytope object based on a H-representation.
        
        ### NOTE: In all these constructors, we should always check that no ADDITIONAL keyword arguments are given, i.e., kwargs should be empty after we have 'used' all keywords, and we should raise an error otherwise.

        """
        raise NotImplementedError
    
    def _init_V_repr(self, V: ArrayLike, **kwargs):
        """Initialize the Polytope object based on a V-representation."""
        raise NotImplementedError

    @property
    def F(self):
        if not self.H_repr:
            self._A, self._g = hrepr(self)
        return self._A
    
    @property
    def g(self):
        if not self.H_repr:
            (self._A, self._g), self.H_repr = hrepr(self), True
        return self._g
    
    @property
    def verts(self):
        if not self.V_repr:
            self._verts, self.V_repr = extreme(self), True
        return self._verts
    
    @property
    def cheb_c(self):
        """Compute the Chebyshev center of the polytope."""
        if self._cheb_c is None:
            ...
        raise NotImplementedError
    
    @property
    def cheb_r(self):
        """Compute the Chebyshev radius of the polytope."""
        if self._cheb_r is None:
            ...
        raise NotImplementedError
    
    @property
    def n(self):
        """The dimension of the space in which the polytope lives.
        
        ### FIXME: Should return a value based on the representation of the polytope, i.e., if H-repr, then F.shape[1], if V-repr, then verts.shape[1].
         
        """
        return self._A.shape[1]  # FIXME: Placeholder
    
    @n.setter
    def n(self, value: int):
        warnings.warn("The dimension of the space 'n' is read-only and should not be set directly.", stacklevel=2, category=UserWarning)
        self._is_consitent = False
        self_n = value
    
    @property
    def dim(self):
        """The dimension of the polytope itself, which can be different (i.e., equal or smaller) from the ambient space dimension."""
        raise NotImplementedError
    
    @property
    def vol(self):
        """Compute the volume of the polytope."""
        if self._vol is None:
            ...
        raise NotImplementedError
    
    @property
    def edges(self) -> list[tuple[ArrayLike, ArrayLike]]:
        """Compute the edges of the polytope.
        
        ### FIXME: How do we want to represent edges? As a list of tuples of vertices? Or as a D x n x 2 Numpy array, where D is the number of edges, n is the dimension of the space, and 2 is the two endpoints of the edge?

        """
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
    
    def __hash__(self) -> int:
        """Implements the magic method for the hash operator `hash` as the hash of the polytope."""
        ### FIXME: This is directly from `pypolycontains`
        if self.hash_value is None:
            self.hash_value = hash(str(np.hstack([self.H, self.h])))
        return self.hash_value
    
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
    
    def __pow__(self, power: int) -> Box:
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
        return f"{self.__class__.__name__}(A.shape={self._A.shape}, A.dtype={self._A.dtype}, b.shape={self._g.shape}, verts.shape={self._verts.shape if self.V_repr else None}, n={self.n}, min_repr={self.is_min_repr}, is_empty={self.is_empty})"
    
    def __format__(self, format_spec: Any) -> str:
        if format_spec == 'fancy':
            return "\n------ FANCY ------\n"+ f"Polytope in ℝ^{self.n} with H-representation F ∈ ℝ^{{{self.F.shape[0]} x {self.F.shape[1]}}}, g ∈ ℝ^{self.g.shape[0]}, V-representation with {self.verts.shape[0] if self.V_repr else None} vertices, volume ...," + "\n------ FANCY ------" 
        elif format_spec == 'debug':
            raise NotImplementedError
        else:
            return super().__format__(format_spec)
    
    def __array_ufunc__(self, ufunc, method, other, *args, **kwargs) -> Polytope:
        """Magic method which gets called if the `other` object is a Numpy array. Depending on what operation to perform, we do different stuff.

        ### FIXME: Should we split `*args` instead, because I believe it's a tuple of fixed elements?

        ### FIXME: Actually, we should probably just have `__array_func__ = None` such that Python reverts back to the default behavior, see link below.
        # FROM: https://docs.scipy.org/doc/numpy-1.13.0/neps/ufunc-overrides.html

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
                return copy.deepcopy(self)
            case 'shallow':
                return copy.copy(self)
            case 'reference':
                return self
            case _:
                raise ValueError(f"Unrecognized copy type '{type}'")
    
    def bbox(self, in_place: bool = False) -> Box:
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
    
    def max(self) -> np.ndarray:
        """Returns the vertex which is 'furthest away' from either the center, the origin, or some other point. If multiple vertices are equally far, the one with smallest angle (going counterclockwise, in the positive horizontal direction) is returned.
        
        """
        raise NotImplementedError
    
    def sample(self, seed: int = None, method: str = 'rejection', density: Literal['uniform'] | Callable = 'uniform') -> ArrayLike:
        """Sample a point from the polytope."""
        match method:
            case 'rejection':
                ### TODO: Instead of just having rejection sampling from a uniform distribution, we can also actually have rejection sampling from any other distribution quite easily!
                ### FROM: https://en.wikipedia.org/wiki/Rejection_sampling
                bbox = self.bbox()
                point = bbox.sample(seed, density)
                while point not in self:
                    point = bbox.sample(seed, density)
                return point
            case 'hit-and-run':
                ### TODO: Check out "Convergence properties of hit–and–run samplers", Bélisle et al. (1998)
                raise NotImplementedError
            case _:
                raise ValueError(f"Unrecognized sampling method '{method}'")
            
    def to_zono(self) -> Zonotope:
        """Convert the polytope to a zonotope, if possible. Raises and error if the polytope is not a zonotope.

        Raises
        ------
        ValueError
            If the polytope cannot be converted to a zonotope, i.e., if it is not a zonotope.

        """
        raise NotImplementedError
    
    def to_graph(self):
        """Convert the polytope to a Hasse diagram"""
        ### FIXME: Maybe more explicit? Like `to_hasse_diagram`?
        raise NotImplementedError
    
    def to_H_repr(self, in_place: bool = True) -> None | Polytope:
        """Convert the polytope to H-representation, if it is not already in H-representation, and remove the V-representation.
        
        Parameters
        ----------
        in_place : bool
            Whether to perform the operation in place. Default is False.

        Returns
        -------
        poly : Polytope
            The polytope in H-representation.

        """
        raise NotImplementedError
    
    def to_V_repr(self, in_place: bool = True) -> None | Polytope:
        """Convert the polytope to V-representation, if it is not already in V-representation, and remove the H-representation.
        
        Parameters
        ----------
        in_place : bool
            Whether to perform the operation in place. Default is False.

        Returns
        -------
        poly : Polytope
            The polytope in V-representation.

        """
        raise NotImplementedError
    
    def plot(ax: plt.Axes = None, show: bool = True) -> plt.Axes | None:
        """Plot the polytope in 1D, 2D, or 3D.  

        ### FIXME: Should this be an external method as well? I.e., `geo.plot(poly)`? 
        ### TODO: Make it also such that the vertices of the polytope are numbered with the index of the `V` array!

        """
        ### FIXME: Do we want lazy imports?
        try:
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise ImportError("matplotlib is required for plotting polytopes.") from e
        raise NotImplementedError


class Zonotope(Polytope):
    """The zonotope class which implements a zonotope `zono` = {c + Gz ∈ ℝ^n | ‖z‖_∞ ≤ 1}.
    
    ### NOTE: A zonotope can also be constructed as the Minkowski sum of line segments of V. So we should also have a constructor method for that. Wait, maybe this is the same as G?

    ### FIXME: We should call the G-representation the generator representation

    """
    def __init__(self, G: ArrayLike, c: ArrayLike):
        """Construct a zonotope with generator matrix `G` and center `c`.
        
        Parameters
        ----------
        G : ArrayLike
            The generator matrix G ∈ ℝ^{n x m} in the zonotope representation.
        c : ArrayLike
            The center of the zonotope.
        
        """
        self.G_repr: bool = True
        self.G = G
        self.c = c  ### FIXME: Does a center differ from a centroid? If so, should we not use abbreviations?
        # FIXME: Convert (G, c) into a halfspace representation (A, b)
        super().__init__(self.G + 1, self.c - 1)


class Box(Zonotope):
    """A n-box is a special type of zonotope, where the half-spaces are all aligned with some x_i-x_j plane.

    ### FIXME: This is actually more often called a Box, see https://en.wikipedia.org/wiki/Hyperrectangle
    
    """

    def __init__(self, A = None, b = None):
        super().__init__(A, b)
        self.bounds = np.max(A, axis=0)  ### FIXME: Placeholder

    def sample(self) -> ArrayLike:
        """Sample a point from the box, which is easier then from a general polytope, and therefore used by rejection sampling.
        
        """
        raise NotImplementedError
    

class Cube(Zonotope):
    """A n-cube is a special type of zonotope, where all edges have the same length.

    ### FIXME: I don't know if we actually want to use this class.
    
    """

    def __init__(self, length: float, center: ArrayLike = None, rotation: ArrayLike = None):
        ...
        self.center = ...
        self.rotation = ...  ### FIXME: How do we want to represent rotations? As a matrix, or as a list of angles?
        raise NotImplementedError


def verts_to_poly(verts: ArrayLike) -> Polytope:
    """Convert vertices `verts` to a polytope.

    ### FIXME: Should these be class methods instead? I.e., `@classmethod` and move them to `class Polytope`, such that we can call `geo.Polytope().verts_to_poly(verts)`? Seems a bit more verbose, right? But again, this is really a different constructor method...

    ### FIXME: In the above cause, we should consider renaming `geo.Polytope().from_verts(verts)` instead, right?
    
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


def bounds_to_poly(lb: ArrayLike, ub: ArrayLike) -> Box:
    """Convert lower and upper bounds `lb` and `ub` to a polytope.

    ### FIXME: This should probably just be a wrapper for a `@classmethod` of Cube, i.e., `Cube.from_bounds(lb, ub)`, right? 
    
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

    ### FIXME: Should be called `norm_to_region` instead, as it can return both a Polytope and an Ellipsoid?
    
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
        

def convex_hull() -> ArrayLike[float]:
    """Compute the convex hull of a set of points"""
    raise NotImplementedError


def vrepr(poly: Polytope) -> ArrayLike:
    """Convert a polytope `poly` from H-representation to V-representation.

    ### FIXME: "However, the computational step from one of the main theorem’s descriptions of polytopes to the other—a “convex hull computation”—is often far from trivial. Essentially, there are three types of algorithms available: inductive algorithms (inserting vertices, using a so-called beneath-beyond technique), projection algorithms (known as Fourier–Motzkin elimination or double description algorithms), and reverse search methods (as introduced by Avis and Fukuda [AF92]). For explicit computations one can use public domain codes as the software package polymake [GJ00] that we use here, or sage [SJ05]; see also Chapters 26 and 67." (from "Basic Properties of Convex Polytopes", Henk Martin)
    
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


def enum_facets(verts: ArrayLike, rays: ArrayLike) -> ArrayLike:
    """Enumerate the facets of a polytope given its vertices and rays. Uses the Fourier-Motzkin elimination method. It leverages the libary `pycddlib`, which implements the double description method [1].
    
    Examples
    --------
    This is an example how the algorithm works.
    Step 0: V = {[1, 1], [2, 3]}  # Given vertices
    Step 1: As the polytope P is conv(V), it must hold that:
    lmd_1 + lmd_2 = 1
    lmd_1 * [1, 1] + lmd_2 * [2, 3] = [x_1, x_2]
    lmd_1, lmd_2 >= 0
    Step 2: Use the Fourier-Motzkin elimination to eliminate lmd_1 and lmd_2, resulting in the (in)equalities:
    2 * x_1 - x_2 == 1
    1 <= x_1 <= 2
    Step 3: Convert the equalities to inequalities (actually, we don't want to do that), resulting in the H-representation:
    A = [[-2,  1],
         [ 2, -1],
         [-1,  0],
         [ 1,  0]]
    b = [-1, 1, -1, 2]

    References
    ----------
    [1] Fukuda, K., Prodon, A. (1996). Double description method revisited. In: Deza, M., Euler, R., Manoussakis, I. (eds) Combinatorics and Computer Science. CCS 1995. Lecture Notes in Computer Science, vol 1120. Springer, Berlin, Heidelberg. https://doi.org/10.1007/3-540-61576-8_77
    
    """
    raise NotImplementedError


def enum_verts(A: ArrayLike, b: ArrayLike, A_eq: ArrayLike | None = None, b_eq: ArrayLike | None = None) -> ArrayLike:
    """Enumerate the vertices of a polytope given its half-space representation. Uses the Avis-Fukuda reverse search algorithm [1].
    
    References
    ----------
    [1] Avis, D., Fukuda, K. (1996). Avis, D., Fukuda, K. A pivoting algorithm for convex hulls and vertex enumeration of arrangements and polyhedra. Discrete Comput Geom 8, 295–313 (1992). https://doi.org/10.1007/BF02293050

    """
    raise NotImplementedError


def power(poly: Polytope, power: int) -> Box:
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


def translation(poly: Polytope, x: ArrayLike) -> Polytope:
    """Translate the polytope `poly` by a vector `x`, i.e., P = {x + y ∈ ℝ^n | y ∈ P}.

    ### FIXME: Naming convention: should it be 'translate' or 'translation'? And should it be 'intersection' or 'intersect'?
    
    """
    raise NotImplementedError


def rotation(poly: Polytope, angles: ArrayLike, center: str = 'origin') -> Polytope:
    """Rotated a polytope by angle θ_1, θ_2, ..., θ_n  = `angle` around the origin.

    ### FIXME: Maybe we should make this an `in_place` operation, as it's not really a new object? So add it as a method to the class?

    ### FIXME: Also check out the Givens rotation matrix, as `givens_rotation_matrix` in the package `polytope`.

    ### TODO: Make a helper method in `utils` which takes in a set of angles, and then returns a rotation matrix

    """
    raise NotImplementedError


def scale(poly: Polytope, factor: float, center: str = 'origin') -> Polytope:
    """Scale the polytope P = `poly` by a factor β = `factor` such that W = {β * x ∈ ℝ^n | x ∈ P}. Note that by default, the scaling is performed around the origin.

    ### FIXME: Should we make this a method of the polytope self instead? And have the method `P.scale(a, in_place=True)`?

    ### FIXME: Also maybe scaling just relative to a vector point
    
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
        

def cut_with_hyperplane(poly: Polytope, plane: None) -> tuple[Polytope, Polytope]:
    """Cuts a polytope with a hyperplane, resulting in two polytopes
    
    ### FIXME: 
    """
    return NotImplementedError


def find_longest_ray(poly: Polytope, center: str = 'mass') -> ArrayLike:
    """Find the longest ray in a polytope `poly`.
    
    """
    raise NotImplementedError
        

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


def normalize():
    """### FIXME: I don't know what this should do, see `MPT3` package for reference. https://people.ee.ethz.ch/~mpt/2/docs/refguide/mpt/@polytope/normalize.html
    
    """
    raise NotImplementedError


def distance(poly_1: Polytope, poly_2: Polytope, type: str = 'shortest') -> float:
    """Compute the Hausdorff distance between two polytopes `poly_1` and `poly_2`. This is the greatest of all the distances from a point in one set to the closest point in the other set.

    ### FIXME: Do we also want the shortest distance? Or the average distance? Seems like useful metrics, right? Look at `Hausdorff_distance` and `distance_polytopes` in the package `pypolycontains`.

    """
    match type:
        case 'shortest':
            raise NotImplementedError
        case 'hausdorff':
            raise NotImplementedError
        case 'average':
            raise NotImplementedError
        case 'centroid':
            raise NotImplementedError
        case _:
            raise ValueError(f"Unrecognized distance type '{type}'")


def enum_int_points(poly: Polytope) -> ArrayLike:
    """Enumerate the integer points in a polytope `poly`, i.e., a lattice.
    
    ### TODO: Look into if this has cryptographic implementations? 

    ### FIXME: Gabriel: Can be used for integer programming, combinatorial optimization (cutting plane methods)
    
    """
    raise NotImplementedError


def convex_hull(points: ArrayLike) -> ArrayLike:
    """Compute the convex hull of a set of points `points`.

    ### FIXME: Instead of an argument `points`, should it not also be `poly_1` and `poly_2`?
    
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

    ### FIXME: See also `support` from `polytope` and `pytope` packages.

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


def blaschke_sum(poly_1: Polytope, poly_2: Polytope) -> Polytope:
    """Compute the Blaschke sum `poly_1` # `poly_2` of two polytopes.

    ### FIXME: I don't know what this actually is.
    
    Parameters
    ----------
    poly_1 : Polytope
        The first polytope.
    poly_2 : Polytope
        The second polytope.

    Returns
    -------
    poly : Polytope
        The Blaschke sum of the two polytopes.

    References
    ----------
    [1] B. Grünbaum, V. Kaibel, V. Klee, G. M. Ziegler. (2003). "Convex Polytopes," Graduate Texts in Mathematics.

    """
    raise NotImplementedError


def subdirect_sum(poly_1: Polytope, poly_2: Polytope) -> Polytope:
    """Compute the subdirect sum `poly_1` ⊞ `poly_2` of two polytopes.

    ### FIXME: From "Basic Properties of Convex Polytopes", Henk Martin, I don't know what this actually is.

    Parameters
    ----------
    poly_1 : Polytope
        The first polytope.
    poly_2 : Polytope
        The second polytope.

    Returns
    -------
    poly : Polytope
        The subdirect sum of the two polytopes.
    """
    raise NotImplementedError


def direct_sum(poly_1: Polytope, poly_2: Polytope) -> Polytope:
    """Compute the direct sum `poly_1` + `poly_2` of two polytopes.

    ### FIXME: From "Basic Properties of Convex Polytopes", Henk Martin, I don't know what this actually is.

    Parameters
    ----------
    poly_1 : Polytope
        The first polytope.
    poly_2 : Polytope
        The second polytope.

    Returns
    -------
    poly : Polytope
        The direct sum of the two polytopes.
    """
    raise NotImplementedError


def mink_diff(poly_1: Polytope, poly_2: Polytope) -> Polytope:
    """Compute the Minkowski difference `poly_1` ⊖ `poly_2` of two polytopes.

    ### FIXME: Check that this name is actually correct
    
    Parameters
    ----------
    poly_1 : Polytope
        The first polytope.
    poly_2 : Polytope
        The second polytope.

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


def convex_hull(V: ArrayLike) -> ArrayLike:
    """Compute the convex hull of a set of points `V`.

    Parameters
    ----------
    V : ArrayLike
        A m x n array, where n is the dimension of the vector space and m is the number of points.
    method : str, default: 'scipy', options: {'scipy', 'cvxpy'}  ### FIXME: How to specify the options in the docstring?
        The method to compute the convex hull. Default is 'scipy'.

    Returns
    -------
    hull : ArrayLike
        The convex hull of the points.
    
    """
    raise NotImplementedError


def convex_union(poly_1: Polytope, poly_2: Polytope) -> Polytope:
    """Return the 'union' of two polytopes, i.e., the smallest polytope that contains both `poly_1` and `poly_2`. Note that a regular union `poly_1 ∪ poly_2` is possibly non-convex, and if so does not return a polytope.
    
    """
    raise NotImplementedError


def convex_set_diff(poly_1: Polytope, poly_2: Polytope, method: str = 'max_volume') -> Polytope:
    """Return the 'set difference' of two polytopes `poly_1` \\ `poly_2` = {x ∈ ℝ^n | x ∈ poly_1, x ∉ poly_2}, making sure this difference is convex.

    ### FIXME: This method might be really wonky, and a bit contrived, so we need to think if this makes any mathematical sense at all...
    
    """
    match method:
        case 'max_volume': 
            # NOTE: What this should do, is that it adds one hyperplane to the polytope poly_1, which intesect the most pertruding point of poly_2, and then optimize the normal vector (i.e., the angle) to maximize the volume of the resulting polytope
            raise NotImplementedError
        case 'exclude_verts':
            # NOTE: This should be a much more simple method, where we first of all exclude all the vertices of poly_1 which are inside poly_2, and then we keep removing vertices of poly_1 (which are closest to poly_2) until we have a convex polytope
            raise NotImplementedError
        case 'normal_to_cheb_c':
            # NOTE: Here, we also add on extra half plane constraint, but the normal vector is fixed as pointing to the center of the polygon; this could, in many cases, be the most simple and efficient method which also gives 'large' volume
            raise NotImplementedError
        case 'extend_half_spaces':
            # NOTE: In this method, we take all the halfspaces of poly_2 that 'run through' poly_1, and add the inverse inequality to poly_1, and then out of all these, we select the one with maximal volume
            raise NotImplementedError
        case _:
            raise ValueError(f"Unrecognized method '{method}'")


def projection(points: Polytope | ArrayLike, proj: list | Subspace, keep_dims: bool = True) -> Polytope:
    """Compute the projection of a polytope or vector `points` onto a subspace `proj` as T = Proj(V, z).

    ### FIXME: Look at `ray_shooting_hyperplanes` in the package `pypolycontains` for inspiration.

    ### FIXME: In `MPT3 Toolbox`, they also talk about projecting a point on the k-th facet (which is probably the fase?) of a polyhedron, https://www.mpt3.org/pmwiki.php/Main/HowTos
    
    """
    raise NotImplementedError


def is_subset(poly_1: Polytope, poly_2: Polytope) -> bool:
    """Check if the polytope `poly_1` is a subset of the polytope `poly_2`, i.e., P ⊆ Q.
    
    """
    raise NotImplementedError


def comp_cheb_ball(poly: Polytope) -> Sphere:
    """Compute the Chebyshev ball of a polytope `poly`.

    ### FIXME: Maybe make this a class-bound method?
    
    """
    raise NotImplementedError


def envelope(poly: Polytope) -> Polytope:
    """### FIXME: I don't know what this should do, see `polytope` package for reference.

    """
    raise NotImplementedError


def gen_rand_poly(n: int, n_verts: int, method: str = 'rand_hull') -> Polytope:
    """Generate a random polytope based on the method selected.
    
    Parameters
    ----------
    n : int
        The dimension of the vector space.
    n_verts : int
        The number of vertices of the polytope.
    method : str, options: ['rand_hull', 'rand_cube', 'rand_ball']   ### FIXME: How to specify the 'type-hinting' in the docstring?
        The method to generate the polytope.

    """
    raise NotImplementedError


def regular_poly(n: int, schlafli_symbols: tuple) -> Polytope:
    """Generate a regular polytope based on the Schläfli symbols. The polytope is centered around the Chebyshev center.
    
    """
    raise NotImplementedError


def is_adjacent(poly_1: Polytope, poly_2: Polytope) -> bool:
    """Check if two polytopes are adjacent, i.e., if they share a common facet.

    ### FIXME: Directly copied from `polytope` package

    ### NOTE: This should be handy for a `Partition` class, where we want to know which polytopes are adjacent to each other.
    
    """
    raise NotImplementedError


# ------------ SUBSPACE ------------


class Subspace(ConvexRegion):
    """A subspace class which implements a subspace `sub` = {x ∈ ℝ^n | x ∈ range(E)}, where E is the basis of the subspace.
    
    """

    def __init__(self, *args, basis: ArrayLike | None = None, n: int | None = None, reduce: bool = True, ortonormal: bool = False):
        #: Check the number of arguments
        match len(args):
            case 0:
                if n is not None and basis is None:
                    ### FIXME: Having the zero vector here is NOT correct! We should instead have an empty basis, as the zero vector CANNOT be a part of a basis (it not being linearly independent)!
                    basis = np.array([]).reshape((n, 0))
                elif n is not None and basis is not None:
                    raise ValueError("Cannot specify both 'n' and 'basis' when no positional arguments are given")
            case 1:
                if basis is not None or n is not None:
                    raise ValueError("Cannot specify 'n' or 'basis' when one positional argument 'basis' is given")
                basis = args[0]
        self._n: int = basis.shape[0]
        if ortonormal:
            basis = sp.linalg.orth(np.atleast_2d(basis)) if not np.allclose(basis, 0) else np.zeros((self.n, 1))
        ### FIXME: Instead to doing the reduce here, maybe we should change the self.basis setter to always do the reduction? And then to also handle the dimensionality based on that?
        elif reduce:
            basis = span(basis[:, np.newaxis] if basis.ndim == 1 else basis)
        self.basis: ArrayLike = np.atleast_2d(basis)
        self._n: int = self.basis.shape[0]
        self._dim: int | None = (self.basis.shape[1] if not np.allclose(self.basis, 0) else 0) if reduce else None
        self.is_min_repr: bool | None = True if reduce else None
        self.is_trivial: bool | None = np.allclose(self.basis, 0) if reduce else None

    def __getattr__(self, name: str) -> Any | AttributeError:
        if name == 'is_empty':
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'. Use the attribute 'is_trivial' to check if the subspace only contains the zero vector.")
        else:
            self.__getattribute__(name)

    @property
    def n(self):
        return self._n
    
    @property
    def dim(self):
        return self._dim
    
    @property
    def vol(self):
        return 0 if self.n != self.dim else np.inf

    @property
    def perp(self) -> Subspace:
        """Compute the orthogonal complement V^⊥ of the subspace V = `self`.
        
        """
        if self.is_trivial:
            return Subspace(np.eye(self.n))
        else:
            return Subspace(sp.linalg.null_space(self.basis.T)) if self.dim < self.n else Subspace(n=self.n)
    
    def copy(self, type: str = 'deepcopy') -> Subspace:
        """Copies the subspace.
        
        Parameters
        ----------
        type : str
            The type of copy. Default is 'deepcopy'.
        
        Returns
        -------
        subs : Subspace
            A copy of `self`.
        
        """
        match type:
            case 'deepcopy':
                return copy.deepcopy(self)
            case 'copy':
                return self
            case _:
                raise ValueError(f"Unrecognized copy type '{type}'")
            
    __array_ufunc__ = None
    
    def __add__(self, other: Subspace) -> Subspace:
        """Implements the magic method `+` as the (Mikowski) addition, also known as the direct sum, of two subspaces V = `self` and W = `other`.
        
        References
        ----------
        [1] M. A. Massoumnia, "A geometric approach to failure detection and identification in linear systems," Ph.D. dissertation, Massachusetts Institute of Technology, Cambridge, MA, 1986. Available: https://ntrs.nasa.gov/citations/19860014486

        """
        if self.n != other.n:
            raise DimensionError(f"Cannot add subspaces of different dimensions: {self.n} and {other.n}")
        else:
            return subs_add(self, other)
        
    def __radd__(self, other: Subspace) -> Subspace:
        return self.__add__(other)
    
    def __iadd__(self, other: Subspace) -> Subspace:
        self.basis = subs_add(self, other).basis
        return self
    
    def __and__(self, other: Subspace) -> Subspace:
        """Implements the magic method `&` as the intersection V ∩ W of two subspaces V = `self` and W = `other`.
        
        """
        if self.n != other.n:
            raise DimensionError(f"Cannot compute intersection of subspaces of different dimensions: {self.n} and {other.n}")
        else:
            return subs_intersection(self, other)
    
    def __eq__(self, other: Subspace) -> bool:
        return np.linalg.matrix_rank(self.basis) == np.linalg.matrix_rank(np.hstack((self.basis, other.basis))) == np.linalg.matrix_rank(other.basis)
    
    def __matmul__(self, other: ArrayLike) -> Subspace:
        raise NotImplementedError
    
    def __rmatmul__(self, other: ArrayLike) -> Subspace:
        # TEMP
        #
        print(other)
        #
        if other.ndim != 2:
            raise DimensionError(f"Subspace mapping requires a 2D array, got array with ndim={other.ndim}")
        elif other.shape[1] != self.n:
            raise DimensionError(f"Matrix of incompatible dimensions: {self.n} and {other.shape[0]}")
        return Subspace(other @ self.basis)
    
    def __imatmul__(self, other: ArrayLike) -> Subspace:
        ### FIXME: Here I need to check what happens if the dimensionality becomes zero. Better yet, I need to make it such that the setter self.basis is changed to handle this case properly! Such that it always does reduction and sets the proper attributes.
        self.basis = self.basis @ other
        return self
    
    def __mod__(self, other: Subspace) -> QuotientSpace:
        """Implements the magic method `%` as the quotient space V / W of two subspaces V = `self` and W = `other`. Note that this requires that W ⊆ V.
        
        """
        return QuotientSpace(self, other)
    
    def __getitem__(self, key: int | slice) -> ArrayLike:
        """Implements the magic method for indexing `[]` to get the basis vectors of the subspace.
        
        Parameters
        ----------
        key : int | slice
            The index or slice of the basis vectors to get.
        
        Returns
        -------
        basis_vec : ArrayLike
            The basis vector(s) at the given index or slice.
        
        """
        if self.is_trivial:
            raise IndexError("Cannot index basis vectors of a trivial subspace")
        return self.basis[:, key]
    
    def __setitem__(self, key: int | slice, value: ArrayLike) -> None:
        """Implements the magic method for indexing `[]` to set the basis vectors of the subspace.
        
        Parameters
        ----------
        key : int | slice
            The index or slice of the basis vectors to set.
        value : ArrayLike
            The basis vector(s) to set at the given index or slice.
        
        """
        self.basis[:, key] = value
        self.reduce()

    def __delitem__(self, key: int | slice) -> None:
        """Implements the magic method for indexing `[]` to delete the basis vectors of the subspace.
        
        Parameters
        ----------
        key : int | slice
            The index or slice of the basis vectors to delete.
        
        """
        self.basis = np.delete(self.basis, key, axis=1)
        self.reduce()

    def __iter__(self):
        """Implements the magic method for iteration over the basis vectors of the subspace.
        
        Yields
        ------
        basis_vec : ArrayLike
            The next basis vector in the subspace.
        
        """
        for i in range(self.basis.shape[1]):
            yield self.basis[:, i]

    def __invert__(self) -> Subspace:
        """Implements the magic method `~` as the orthogonal complement V^⊥ of the subspace V = `self`.

        ### FIXME: Is `__invert__` the right method/name for this?
        
        """
        raise NotImplementedError
    
    def __str__(self) -> str:
        ### FIXME: Placeholder
        print(self.__repr__(), end='')
        return ", with basis:\n" + str(self.basis)
    
    def __repr__(self) -> str:
        """Debug print the subspace."""
        return f"{self.__class__.__name__}(basis.shape={self.basis.shape}, n={self.n}, dim={self.dim}, min_repr={self.is_min_repr}, is_trivial={self.is_trivial})"

    def reduce(self) -> Subspace:
        """Reduce the basis `basis` of the subspace to a minimal basis.
        
        """
        ### FIXME: Check for linear independence using a rank condition
        self.basis = span(self.basis) if not self.is_trivial else np.zeros((self.n, 1))
        self.is_trivial = np.allclose(self.basis, 0)
        self._dim = self.basis.shape[1] if not self.is_trivial else 0
        self.is_min_repr = True
        return self
    
    def orthonormal(self, in_place: bool = True) -> None | Subspace:
        """Compute an orthonormal basis for the subspace `self`.

        ### FIXME: When should a method just act on itself, and when should it return a new object?
        
        """
        basis = sp.linalg.orth(self.basis) if not self.is_trivial else np.zeros((self.n, 1))
        if in_place:
            self.basis = basis
        else:
            return Subspace(basis)
    

class AffineSubset(ConvexRegion):
    """Class which implements an affine subspace `aff_sub` = {x ∈ ℝ^n | x = x_0 + v, v ∈ V}, where V is a subspace and x_0 is a point in the affine subspace.
    
    """

    def __init__(self, point: ArrayLike, basis: ArrayLike):
        """Construct an affine subspace `aff_sub` = {x ∈ ℝⁿ | x = x₀ + v, v ∈ V}, where V is a subspace and x_0 is a point in the affine subspace.

        Parameters
        ----------
        point : ArrayLike
            A point x_0 in the affine subspace.
        basis : ArrayLike
            The basis of the subspace V.
        
        """
        raise NotImplementedError
    

class QuotientSpace:
    """Class which implements a quotient space V / R = {V + r | r ∈ R}, where R is a subspace of V, and V + r is an affine subspace.

    ### FIXME: Should I also make the class `AffineSubset`, as the elements of the quotient space are affine subspaces?

    ### FIXME: Gabriel: Look op Grassmannian | one-to-one with orthogonal matrices, lots of interplay

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
    

def span(a: ArrayLike) -> ArrayLike:
    """Compute a basis spanned by the matrix `a` by removing linearly dependent columns"""
    if a.shape[1] == 0:
        return a
    else:
        basis = [a[:, 0]]
        for i in range(1, a.shape[1]):
            if np.linalg.matrix_rank(np.column_stack((*basis, a[:, i]))) > len(basis):
                basis.append(a[:, i])
        return np.column_stack(basis)
    


def subs_add(subs_1: Subspace, subs_2: Subspace) -> Subspace:
    """Compute the addition, or *direct sum*, of two subspaces `subs_1` + `subs_2`"""
    new_basis = span(np.hstack((subs_1.basis, subs_2.basis)))
    return Subspace(new_basis)


def subs_intersection(subs_1: Subspace, subs_2: Subspace) -> Subspace:
    """Compute the intersection of two subspaces `subs_1` ∩ `subs_2`"""
    intersection_basis = sp.linalg.null_space(np.row_stack((subs_1.perp.basis.T, subs_2.perp.basis.T)))
    return Subspace(intersection_basis)


def inv_map(A: ArrayLike, subs: Subspace) -> Subspace:
    """Compute the inverse map of a subspace `V` under the linear transformation `A`, i.e., A^{-1} V"""
    return Subspace(sp.linalg.null_space(subs.perp.basis.T @ A))


def angle(subs_1: Subspace, subs_2: Subspace) -> float:
    """Compute the angle between two subspaces `subs_1` and `subs_2`.
    
    """
    raise NotImplementedError


def quotient(subs_1: Polytope, subs_2: Polytope) -> QuotientSpace:
    """Compute the quotient space Q = V / W of two subspaces V = `subs_1` and W = `subs_2`. Note that this requires that W ⊆ V.
    
    """
    raise NotImplementedError


# ------------ ELLIPSOID ------------


class Ellipsoid(ConvexRegion):
    """The ellipsoid class which implements an ellipsoid `ell` = {x ∈ ℝ^n | (x - c)^T P^-1 (x - c) ≤ α}.

    ### NOTE: According to "Constrained zonotopes: A new tool for set-based estimation and fault detection," there is also the definition {Qx + c | ‖x‖_2 ≤ a}. Ohh nevermind, this is for DEGENERATE ellipsoids, where P is not invertible. But this is super important, as this is another representation!

    References
    ----------
    [1] Alexander, A., Valyi, I. (1997). "Ellipsoidal Calculus for Estimation and Control," ...
    
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
    
    def bbox(self) -> Box:
        """Compute the bounding box of the ellipsoid.
        
        """
        ### FROM: https://math.stackexchange.com/questions/3926884/smallest-axis-aligned-bounding-box-of-hyper-ellipsoid
        P_inv = np.linalg.inv(self.P)
        ### FIXME: Why is this the one that works? It seems to me that Q = Q / a should work, but it doesn't? Why does the sqrt mess thing up?
        lb, ub = self.c - self.a * np.sqrt(np.diag(P_inv)), self.c + self.a * np.sqrt(np.diag(P_inv))
        return bounds_to_poly(lb, ub)
    

    def sample(self, seed: int = None, in_ellps: bool = True) -> ArrayLike:
        """Sample a point from the ellipsoid according to the normal distribution.

        Parameters
        ----------
        seed : int
            The random seed.
        in_ellps : bool
            Whether to sample from inside the ellipsoid. Default is True. If False, we do not use the truncated normal distribution, but the normal distribution.
        
        """
        raise NotImplementedError


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


def normal_to_ellps(mean: ArrayLike, Sigma: ArrayLike, a: float) -> Ellipsoid:
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
    ellps: Ellipsoid
        The ellipsoid representing the normal distribution.

    Examples
    --------

    Consider the normal distribution with mean = 0 and Sigma = ... Now, if we want the 95% confidence interval, we note that this is equal to about 3 times the standard deviation. Therefore, we can write:
        
    """
    raise NotImplementedError


def ellps_to_poly(ellps: Ellipsoid) -> Polytope:
    """Create an inner or outer polytypic approximation of an ellipsoid `ellps`.

    ### FIXME: Look at "Algorithms for Polyhedral Approximation of Multidimensional Ellipsoids", https://www-sciencedirect-com.tudelft.idm.oclc.org/science/article/pii/S0196677499910313
    ### FIXME: Look at "Ellipsotopes: Uniting Ellipsoids and Zonotopes for Reachability Analysis and Fault Detection", https://arxiv.org/pdf/2108.01750
    
    """
    raise NotImplementedError


def poly_to_ellps(poly: Polytope) -> Ellipsoid:
    """Create an inner or outer ellipsoidal approximation of a polytope `poly`.

    ### FIXME: Look at "MOSEK, 11.6 Inner and outer Löwner-John Ellipsoids", https://docs.mosek.com/latest/dotnetfusion/case-studies-ellipsoids.html
    ### FIXME: Look at this note, https://www.mi.uni-koeln.de/opt/wp-content/uploads/2015/12/CO2015-16-VL19.pdf
    
    """
    raise NotImplementedError


def lyap_to_ellps(A: ArrayLike, Q: ArrayLike) -> Ellipsoid:
    """Convert a Lyapunov function V(x) = x^T Q x to an ellipsoid.

    ### FIXME: This is just a placeholder, but do something with ellipsoids and level sets of Lyapunov functions

    """
    raise NotImplementedError


def actuator_sat_reach(A: ArrayLike, B: ArrayLike, U: Polytope) -> Ellipsoid:
    """Calculate an under-approximation of the maximal reachable set of a linear system with actuator saturation.
    
    References
    ---------
    [1] S.H. Kafash, J. Giraldo, C. Murguia, A.A. Cardenas, J. Ruths. (2018, June). "Constraining Attacker Capabilities Through Actuator Saturation," 2018 Annual American Control Conference (ACC), Milwaukee, USA, pp. 986-991

    """
    raise NotImplementedError


# ------------ CONTROL -------------


def mrpi(A: ArrayLike, B: ArrayLike, K: ArrayLike, W: Polytope, method: str = 'lazar', s_max: int = 10) -> Polytope:
    """Compute the minimal robustly positive invariant set F_∞ = ⊕_{i=0}^{∞} (A + B K)^i W. See also [1, Algorithm 1].

    ### FIXME: Also look at `eps_MRPI` in the package `pytope`. There's several algorithms there

    ### FIXME: Also look at `calculate_maximum_admissible_output_set` from `pytope`, from "LinearSystemswith StateandControlConstraints: The TheoryandApplicationof Maximal OutputAdmissible Sets"

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


def reach_in_n_steps(A: ArrayLike, B: ArrayLike, X: Polytope, U: Polytope, n: int) -> Polytope:
    """Compute the reachable set of a linear system in `n` steps.

    """
    raise NotImplementedError


def feas_reg_mpc() -> Polytope:
    """Compute the feasible region of a model predictive control (MPC) problem.
    
    """
    raise NotImplementedError


def max_ctrl_inv_subs(A: ArrayLike, B: ArrayLike, C: ArrayLike) -> Subspace:
    """Computes the maximal (A,B)-controlled invariant subspace `V_star` contained in the kernel of `C`, i.e., V_star ⊆ ker(C). Note that maximal here refers to the unique subspace `V_star` with the largest dimension, i.e., V* = max_{V is (A, B)-controlled invariant, V ⊆ ker(C)} dim(V). See [1, Algorithm 4.1.2].

    ### NOTE: Gabriel: these are fixed-point algorithms

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


def max_output_nulling_subs(A: ArrayLike, B: ArrayLike, C: ArrayLike, D: ArrayLike = None) -> Subspace:
    """Computes the maximal output nulling subspace

    ### NOTE: These are fixed-point algorithms
    
    """
    raise NotImplementedError


def partition(poly: Polytope) -> list[Polytope]:
    """Partition a polytope `poly` into smaller polytopes. This is useful for example in model predictive control (MPC) to create a finite set of polytopes that can be used to approximate the reachable set, or for explicit MPC. We might want to make the class `Partition` for this. 

    ### FIXME: Check out "An algorithmic approach to convex fair partitions of convex polygons," Campillo et all (2024)

    """
    raise NotImplementedError


# ------------ UTILS ------------


class DimensionError(ValueError):
    """Exception raised for errors in the input dimensions.
    
    """

    def __init__(self, message: str):
        """Constructor for the exception.

        Attributes
        ----------
        message : str
            Explanation of the error.
        
        """
        self.message = "dimension of first object does not match dimension of second object"
        super().__init__(self.message)


class IncompatibleTypes(TypeError):
    """Exception raised for errors in the input types.
    
    """

    def __init__(self, message: str):
        """Constructor for the exception.

        Attributes
        ----------
        message : str
            Explanation of the error.
        
        """
        self.message = "incompatible types for operation"
        super().__init__(self.message)


def plot(obj: Polytope | Ellipsoid | Subspace, ax: plt.Axes = None) -> list[plt.Figure, plt.Axes]:
    """Method to plot either a polytope, a ellipsoid, or a subspace.

    ### FIXME: This should probably be a class-bound method for each of the classes?
    
    """
    raise NotImplementedError


def bisect(func: Callable, range: tuple, args: tuple) -> float:
    """Bisection algorithm to find the root of a function `func` in a given `range` with `args`"""
    raise NotImplementedError


def pre_img(A: ArrayLike) -> ArrayLike:
    """Compute the pre-image of a matrix `A`. This is the inverse of the matrix if invertible, and the combination of the pseudo-inverse and the null-space if not.

    ### FIXME: Maybe we should make this a function `inv_map(A, V)` instead?
    
    """
    raise NotImplementedError


def pretty_print(obj: Polytope | Ellipsoid | Subspace) -> str:
    """Pretty print the object `obj`. Used as a helper function for the `__str__` method, as this method can get really verbose.

    Examples
    --------
    Example of how the matrix is printed based on whether it is a V-representation or an H-representation, and how format specifiers can be used.

    >>> import geopy as gp
    >>> import numpy as np
    >>> A, b = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]]), np.array([1, 1, 1, 1])
    >>> p_1 = gp.poly(A, b)
    >>> print(p_1)
    Polytope in ℝ^2
    [[ 1,  0]  |    [[1]
     [ 0,  1]  |     [1]
     [-1,  0]  x <=  [1]
     [ 0, -1]] |     [1]]
    >>> V = np.random.rand(10, 10)
    >>> p_2 = gp.poly(V)
    >>> print(f"{p_2:fancy}")
    Polytope in ℝ^10
    / [ 0.00] [ 1]      [ 0.01] \\
    | [ 2.00] [ 2]      [ 0.50] |
    <    ... , ..., ...,   ...  >
    \\ [-5.31] [-4]      [-0.10] /

    """
    return f"{obj.__class__.__name__} in ℝ^{obj.n}"


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


def main() -> None:

    A = np.array([[1, 0], [0, 1]])
    b = np.array([1, 1])
    poly = Polytope(A, b)

    print(poly)
    print(f"Poly: {poly:fancy}")
    print(repr(poly))

    A = np.array([[0, 0, 0], [1, 2, -1], [0, 0, 1]])
    # A = np.zeros((3, 3))
    subs_1 = Subspace(np.array([[1, 1, 0], [0, 2, 0]]).T)
    subs_2 = Subspace(n=3)
    subs_3 = Subspace(np.array([1, 1, 2]))
    subs_4 = subs_1 + subs_2
    subs_5 = subs_1 + subs_3
    subs_full = Subspace(np.eye(3))
    subs_6 = subs_full & subs_3
    subs_7 = inv_map(A, subs_3)

    subs_5.orthonormal()

    print(np.linalg.eigvals(A))

    print(subs_1)
    print(subs_2)
    print(subs_3)
    print(subs_4)
    print(subs_5)
    print(subs_full)
    print(subs_6)
    print(subs_7)
    print(subs_1 == subs_4)
    print(subs_6 == subs_3)
    print(inv_map(A, subs_3))
    print(A @ subs_7 == subs_3)

    print(subs_1[0])
    print([e for e in subs_1])
    del subs_1[0]
    print(subs_1)
    print(subs_2[0])

    # Here we can also write MPC
    X, U = cvx.Variable((2, 11)), cvx.Variable((2, 10))
    cost, const = [], []
    for k in range(10):
        const += [is_in(X[:, k + 1], poly)]

    # TODO: Implement the following:
    if False:
        A, b = poly.to_ineq()  # We can also have variations on this, but return -> tuple
        v_1, v_2, v_3 = subs  # FIXME: I don't know if we want this, but this is also a possibility: actually, numpy supports this, so... we can have the following:
        v_1, v_2 = subs.basis  # NOTE: Here we are relying on the row-major order of numpy arrays
    

if __name__ == "__main__":
    main()