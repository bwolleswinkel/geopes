GeoPES documentation
===================================

GeoPES is a Python library that implements polytopes, ellipsoids, and subspaces and 
defines mathematical operations on them. The package is designed to work well with control
applications in mind. The coding itself is made to feel as *pythonic* as possible.

======================
Numeric type emulation
======================

The package is easy to use and makes extensive use of the operators :code:`+`, :code:`*`, :code:`in`, :code:`@`, etc.::

   import geopes as geo
   impost numpy as np

   # Create a polytope from a H-representation
   F, g = np.array([[-1, 2], [0, 1], [3, 2], [2, 4]]), np.array([1, 2, 5, -7])
   W = geo.Polytope(F, g)

   # Perform some operation
   Z = -W  # 'Invertion', i.e. Z = {-w | w ∈ W}
   Y = W + Z  # Minkowski sum W ⊕ Z = {w + z | w ∈ W, z ∈ Z}
   Y = W - Z  # Pontryagin difference W ⊖ Z
   Y = g + Y  # Addition with vectors is also supported
   Y = 1.1 * Y  # Scaling
   Y = A @ Y  # Linear transformation
   Y = W & Z  # Intersection W ∩ Z

   D, alpha = Z.copy(), 0.9
   for _ in range(3):
      D = Z & (alpha * (A @ D))  # Z ∩ (α * A D)


Check out the :doc:`usage` section for further information, including
how to :ref:`installation` the project.

.. note::

   This project is a work-in-progress.

Contents
--------

.. toctree::

   usage
   api
   conventions
   examples
