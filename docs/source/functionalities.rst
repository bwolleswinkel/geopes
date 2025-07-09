Functionalities
===============

Here, we describe some functionalities. 

----------------
V-representation
----------------

For polytopes, we can have different representations. To go from H-representation to V-representation, we must solve the linear program given by

.. math:: \min_{\boldsymbol{x}} \boldsymbol{A} \boldsymbol{x} \leq \boldsymbol{b} \quad \text{subject to} \quad \boldsymbol{F} \boldsymbol{x} = \boldsymbol{g}.


----------------
Chebyshev center
----------------

Similar to the ``polytope`` package, we can compute the Chebyshev center by solving the linear program given by

.. math:: \max_{r, \boldsymbol{c}} r \quad \text{subject to} \quad \boldsymbol{a}_{i}^{\mathsf{T}} \boldsymbol{c} + \Vert \boldsymbol{a}_{i} \Vert \cdot r \leq b_{i}, r \geq 0, \forall i.

where 

.. math:: \boldsymbol{A} = \begin{bmatrix} \rule[.5ex]{2.5ex}{0.5pt} & \boldsymbol{a}_{1}^{\mathsf{T}} & \rule[.5ex]{2.5ex}{0.5pt} \\ & \vdots & \\ -\!\!\!-\!\!\!- & \boldsymbol{a}_{m}^{\mathsf{T}} & -\!\!\!-\!\!\!- \end{bmatrix} \quad \text{and} \quad \boldsymbol{b} = \begin{bmatrix} b_{1} \\ \vdots \\ b_{m} \end{bmatrix}.

----------------
Context managers
----------------

GeoPES also provides several context managers. Example are setting the default value of ``in_place`` to ``True`` or ``False``. Examples are:

.. code-block:: python

    X = geo.poly(F, g)  
    3 * X  # This will not modify the polytope in place.

    with geo.in_place(True):
        3 * X  # This will modify the polytope in place.
    
    with geo.default_precision(1e-6):
        print(X <= 0.99 * X)   # This will use the default precision of 1e-6.