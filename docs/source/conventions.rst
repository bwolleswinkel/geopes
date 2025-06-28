Conventions
===========

Just like many other packages, GeoPES uses some conventions. We stick to the conventions from Numpy, which uses **row-major ordering**. This means that, e.g., a list of vertices ``verts`` with :math:`m` vertices :math:`\boldsymbol{x} \in \mathbb{R}^{n}` will be a Numpy array such that ``verts.shape == (m, n)``. This is so we can use the following idiom::

   for vert in poly.verts:
      print(vert)

-----------
Time-series
-----------

To make matters a bit worse, for timeseries data, such as

.. math:: \boldsymbol{X} = \begin{bmatrix} \vert & \vert & & \vert \\ \boldsymbol{x}_{0} & \boldsymbol{x}_{1} & \cdots & \boldsymbol{x}_{N} \\  \vert & \vert & & \vert \end{bmatrix},

we use **column-major ordering**, as this is the convention in the Python ``control`` library. This will occurs when we use, for instance::

   import control as ct
   _, _, X = ct.forced_response(sys, t, u, return_X=True)

----------
Developers
----------

For developers, there are several conventions to follow:

Docstrings
^^^^^^^^^^

For docstring, we follow mainly the NumPy docstring convention, as defined here: `NumPy docstring <https://numpydoc.readthedocs.io/en/latest/format.html>`_. We do have these added restrictions:

* The first line of the docstring should directly follow the ``"""`` symbol, i.e.:
   
   .. code-block:: python

     def my_function():
         """This is the first line of the docstring, immediately after the triple quotes.

         Here we have the last line of the docstring, which is always followed by an empty line. 

         """
         pass

   The reason that we have this, is that if we collapse the docstring in our IDE, we can still see the first line of the docstring, which is often a short description of the function. Otherwise, we just see the ``"""`` symbol, which is not very informative.
   .. image:: images/collapsed_docstring_still_readable.png
      :alt: Example of a collapsed docstring that is still readable
      :align: left
.. FIXME: I don't actually know if I want this? Because if the start of the docstring is very long, there is no way to collapse it in the IDE... so we might want to start on a newline anyway?

* The last line of the docstring should be empty, i.e. there should be a blank line after the last line of the docstring, before the closing ``"""`` symbol.

Abbreviations and naming
^^^^^^^^^^^^^^^^^^^^^^^^

For functions and method, we encourage the use of abbreviations to improve code conciseness. For instance, instead of naming a method ``minimal_reperesentation``, the abbreviation ``min_repr`` should be used. For up to 3 letter words, one should not use and acronym, for 5 letters or more acronyms are preferred, and for 7 letters or more, acronyms should definitely be used. The reason the function and methods names should not be self-explanatory, is that they should always be provided with a good, explanatory docstring, which removes the necessity of the functions name to be self-explanatory.

Secondly, one should separate different word in names with and underscore, e.g., ``min_repr`` and ``is_convex`` (instead of ``minrepr`` and ``isconvex``). The only **exception** to this rule, is when we use a single letter abbreviation for the first word in the abbreviation, such as ``bbox`` for the method ``bounding_box``.

Finally, for classes, the attributes ``self.is_<flag_name>`` are all flags, and should always be booleans. Note the underscore between the ``is`` and the flag name.

Strings
^^^^^^^

For machine-readable strings, one should use ``'single_quotes'``. For human-interpretable strings, one should use ``''double_quotes''``. Example, we have ``ax.plot(..., color='blue')``, as the keyword ``'blue'`` is to be interpreted by the program itself. On the contrary, we should use ``plt.title("This is a title")``, as the title is to be interpreted by the human user. This convention is used to make it clear which strings are meant for the program and which are meant for the user.
