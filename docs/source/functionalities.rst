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

.. math:: \boldsymbol{A} = \begin{bmatrix} --- & \boldsymbol{a}_{1}^{\mathsf{T}} & --- \\ & \vdots & \\ --- & \boldsymbol{a}_{m}^{\mathsf{T}} & --- \end{bmatrix} \quad \text{and} \quad \boldsymbol{b} = \begin{bmatrix} b_{1} \\ \vdots \\ b_{m} \end{bmatrix}.