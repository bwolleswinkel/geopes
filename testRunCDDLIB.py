"""Test script to run the pycddl library tests.

### NOTE: The intepreter must be set to `venvpycddlib` (such that pycddlib is installed).

"""

import numpy as np
import cdd
import matplotlib.pyplot as plt

# ------ SCRIPT ------

# Generate vertices for a polytope in R^2
V_poly_1 = np.array([[-0.2, 0], 
              [0, 1], 
              [1, 2],
              [1, 0]]).astype(float)

# Compute the H-representation
mat_poly_1 = cdd.matrix_from_array(np.column_stack((np.ones(V_poly_1.shape[0]), V_poly_1)), rep_type=cdd.RepType.GENERATOR)  # NOTE: Format for cddlib is [1, x_1, ..., x_n] where the first column is always 1 for vertices (denoting these are points, not rays)
vpoly_1 = cdd.polyhedron_from_matrix(mat_poly_1)
hmat_poly_1 = np.array(cdd.copy_inequalities(vpoly_1).array).astype(float)
A_poly_1, b_poly_1 = hmat_poly_1[:, 1:], -hmat_poly_1[:, 0]

# Generate half-space equations
A_poly_2 = np.array([[5, -1],
                     [0, 2],
                     [-1, -2],
                     [1, -1]]).astype(float)
b_poly_2 = np.array([-1, 0, -3, -1]).astype(float)

# Compute the V-representation
mat_poly_2 = cdd.matrix_from_array(np.column_stack((-b_poly_2, A_poly_2)), rep_type=cdd.RepType.INEQUALITY)  # NOTE: Format for cddlib is [-b, a_1, ..., a_n] where the first column is always -b for inequalities
hpoly_2 = cdd.polyhedron_from_matrix(mat_poly_2)
vmat_poly_2 = np.array(cdd.copy_generators(hpoly_2).array).astype(float)
V_poly_2 = vmat_poly_2[:, 1:]  # Discard the first column which is all 1s for vertices

# ------ PRINT ------

print("---- POLY 1 ----\n")
print("Points (V-representation):")
print(V_poly_1)
print("\nInequalities (H-representation):")
print("A:")
print(A_poly_1)
print("b:")
print(b_poly_1)

print("\n---- POLY 2 ----\n")
print("Inequalities (H-representation):")
print("A:")
print(A_poly_2)
print("b:")
print(b_poly_2)
print("\nPoints (V-representation):")
print(V_poly_2)

# ------ PLOTTING ------

# Set the range of points for plotting the H-representation
X = np.linspace(-1, 4, 100)

# Plot the points from the V-representation
fig_poly_1, ax_poly_1 = plt.subplots()
ax_poly_1.plot(V_poly_1[:, 0], V_poly_1[:, 1], 'bo', label="Given (V-representation)")
for idx in range(A_poly_1.shape[0]):
    if not np.isclose(A_poly_1[idx, 1], 0):
        ax_poly_1.plot(X, (b_poly_1[idx] - A_poly_1[idx, 0] * X) / A_poly_1[idx, 1], 'r--', label="Calculated (H-representation)" if idx == 0 else "")
    else:
        ax_poly_1.axvline(x=b_poly_1[idx] / A_poly_1[idx, 0], color='r', linestyle='--', label="Calculated (H-representation)" if idx == 0 else "")
ax_poly_1.set_title('Polyhedron from V-representation')
ax_poly_1.set_xlabel(r"$x_{1}$")
ax_poly_1.set_ylabel(r"$x_{2}$")
ax_poly_1.legend(loc='upper left')

# Plot the points from the H-representation
fig_poly_2, ax_poly_2 = plt.subplots()
for idx in range(A_poly_2.shape[0]):
    if not np.isclose(A_poly_2[idx, 1], 0):
        ax_poly_2.plot(X, (b_poly_2[idx] - A_poly_2[idx, 0] * X) / A_poly_2[idx, 1], 'b--', label="Given (H-representation)" if idx == 0 else "")
    else:
        ax_poly_2.axvline(x=b_poly_2[idx] / A_poly_2[idx, 0], color='b', linestyle='--', label="Given (H-representation)" if idx == 0 else "")
ax_poly_2.plot(V_poly_2[:, 0], V_poly_2[:, 1], 'ro', label="Calculated (V-representation)")
ax_poly_2.set_title('Polyhedron from H-representation')
ax_poly_2.set_xlabel(r"$x_{1}$")
ax_poly_2.set_ylabel(r"$x_{2}$")
ax_poly_2.legend(loc='upper left')

# Show the plots
plt.show()