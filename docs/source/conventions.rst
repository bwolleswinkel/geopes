Conventions
===========

Just like many other packages, GeoPES uses some conventions. We stick to the conventions from Numpy, which uses **row-major ordering**. This means that, e.g., a list of vertices ``verts`` with :math:`m` vertices :math:`\boldsymbol{x} \in \mathbb{R}^{n}` will be a Numpy array such that ``verts.shape == (m, n)``. This is so we can use the following idiom::

   for vert in poly.verts:
      print(vert)

-----------
Time-series
-----------

To make matters a bit worse, ...