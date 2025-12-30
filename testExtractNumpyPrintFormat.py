"""Script to test the extraction of NumPy print format settings.

# NOTE: This version has been edited by AI; there is a previous commit with the original version, which is more human-readable.
"""

import re
from typing import Literal
import warnings

import numpy as np

# ------ PARAMETERS ------

# Set the options for the integer array
int_rows, int_cols = 3, 10

# Set the options for the floating array
float_rows, float_cols = 2, 100

# Set the options for ellipsoid
n = 2

# Set the maximum number of vertices to print
max_nverts = 10

# ------ METHOD ------

def poly_hrepr(A: np.ndarray, b: np.ndarray, A_eq: np.ndarray | None = None, B_eq: np.ndarray | None = None, object_type: Literal['polyhedron', 'polytope', 'zonotope', 'cube'] = 'polytope', include_tabs: bool = True, trunc_mode: Literal['single_ellipsis', 'both_ellipsis'] = 'both_ellipsis', strip_double_b_brackets: bool = True) -> str:
    (n_ineq, n), edgeitems = A.shape, np.get_printoptions()['edgeitems']
    with np.printoptions(threshold=edgeitems ** 2, edgeitems=edgeitems, linewidth=1000):
        A_str, b_str = (A.flatten().__str__(), b.flatten().__str__()) if n_ineq == 1 else (A.__str__(), np.atleast_2d(b).T.__str__())
    A_lines, b_lines = A_str.splitlines(), b_str.splitlines()
    if strip_double_b_brackets:
        b_lines = [line.replace('[[', '[').replace(' ', '').replace(']]', ']') for line in b_lines]
    n_print_rows = len(A_lines)
    if n_print_rows > 1:
        A_lines = [line + ' ' for line in A_lines[:-1]] + [A_lines[-1]]
    row_trunc_idx = next((idx for idx, line in enumerate(A_lines) if '...' == line.strip(' ')), None)
    center_offset = 0 if n_print_rows <= 2 else 1
    tabs = ['  ' if include_tabs else '' for _ in range(n_print_rows)]
    x_vars = ([' |'] * (n_print_rows - center_offset - 1)) + [' x'] + ([' |'] * center_offset)
    inequalities = (['    '] * (n_print_rows - center_offset - 1)) + [' <= '] + (['    '] * center_offset)
    final_str = ''
    for row_idx in range(n_print_rows):
        if row_trunc_idx is not None and row_idx == row_trunc_idx:
            # TODO: Make a choice on one of these truncation modes, so that we can remove the other and streamline the code
            if trunc_mode not in ['single_ellipsis', 'both_ellipsis']:
                raise NotImplementedError(f"Truncation mode '{trunc_mode}' not recognized")
            final_str += tabs[row_idx] + A_lines[row_idx] + ('\n' if trunc_mode == 'single_ellipsis' else (' ' * (len(A_lines[-1]) - 5)) + x_vars[row_idx] + inequalities[row_idx] + b_lines[row_idx] + '\n')
        else:
            final_str += tabs[row_idx] + A_lines[row_idx] + x_vars[row_idx] + inequalities[row_idx] + b_lines[row_idx] + '\n'
    # TODO: Decide on the header format, and make it automatic for all print functions
    str_header = {'polytope': f"Polytope in R^{n}"}.get(object_type)
    if str_header is None:
        raise NotImplementedError(f"Printing for {object_type} is not yet implemented.")
    return str_header + '\n' + final_str

def poly_vrepr(verts: np.ndarray, rays: np.ndarray | None = None, object_type: Literal['polyhedron', 'polytope', 'zonotope', 'cube'] = 'polytope', include_tabs: bool = True, trunc_mode: Literal['single_ellipsis', 'all_ellipsis'] = 'both_ellipsis', strip_double_brackets: bool = True, max_nverts: int = 5) -> str:
    if max_nverts < 3:
        raise NotImplementedError("max_nverts must be at least 3 to allow for truncation.")
    n, edgeitems = verts.shape[0], np.get_printoptions()['edgeitems']
    with np.printoptions(edgeitems=edgeitems, threshold=(2 * edgeitems + 1), linewidth=1000):
        vert_strs = [np.atleast_2d(vert).T.__str__().splitlines() if n > 1 else vert.T.__str__().splitlines() for vert in verts.T]
    n_rows, n_cols = len(vert_strs[0]), min(len(vert_strs), max_nverts)
    row_trunc, col_trunc = n > n_rows, n_cols < verts.shape[1]
    start_col_idx, end_col_idx = (max_nverts // 2, max_nverts // 2 + 1) if col_trunc else (None, None)
    if not strip_double_brackets:
        vert_strs = [[elem.replace(']', '] ').replace('] ] ', ']]') for elem in row] for row in vert_strs] if n > 1 else vert_strs
    else:
        vert_strs = [['[' + elem.replace('[', '').replace(']', '') + ']' for elem in row] for row in vert_strs]
        warnings.warn("Stripping double brackets is NOT correctly implemented at the moment")
    idx_middle_row, center_offset = n_rows // 2, 0 if n_rows <= 2 else 1
    tabs = ['  ' if include_tabs else '' for _ in range(n_rows)]
    left_brackets = ([' /'] + ([' |'] * (idx_middle_row - 1)) + (['< '] if n_rows != 2 else []) + ([' |'] * (n_rows - idx_middle_row - 2)) + [' \\']) if n_rows != 1 else ['<']
    commas = ([' ' + (' ' if n_rows == 1 else '') for _ in range(n_rows - center_offset - 1)] + [',' + (' ' if n_rows == 1 else '')] + [' ' + (' ' if n_rows == 1 else '') for _ in range(center_offset)]) if n_cols > 1 else (['  '] if n_rows == 2 else [])
    right_brackets = (['\\ '] + (['| '] * (idx_middle_row - 1)) + ([' >'] if n_rows != 2 else []) + (['| '] * (n_rows - idx_middle_row - 2)) + ['/ ']) if n_rows != 1 else ['>']
    final_str = ''
    for row_idx in range(n_rows):
        final_str += tabs[row_idx] + left_brackets[row_idx]
        if row_trunc and row_idx == edgeitems and trunc_mode == 'single_ellipsis':
            n_spaces = sum(len(vert_strs[idx][0]) for idx in range(n_cols)) + (n_cols - 1) - 4
            final_str += ' ...' + (' ' * n_spaces)
        else:
            for col_idx in range(n_cols):
                if col_trunc and col_idx == start_col_idx:
                    final_str += (' ...' if row_idx == (n_rows - center_offset - 1) else '    ') + commas[row_idx]
                elif col_trunc and col_idx > start_col_idx and col_idx < end_col_idx:
                    continue
                elif col_trunc:
                    if row_trunc and row_idx == edgeitems and trunc_mode == 'all_ellipsis':
                        final_str += ' ...' + (' ' * (len(vert_strs[col_idx][0]) - 4))
                    else:
                        final_str += vert_strs[-(max_nverts - col_idx)][row_idx]
                    final_str += (commas[row_idx] if col_idx < n_cols - 1 else '')
                else:
                    if row_trunc and row_idx == edgeitems and trunc_mode == 'all_ellipsis':
                        final_str += ' ...' + (' ' * (len(vert_strs[col_idx][0]) - 4))
                    else:
                        final_str += vert_strs[col_idx][row_idx]
                    final_str += (commas[row_idx] if col_idx < n_cols - 1 else '')
        if row_trunc and row_idx == edgeitems and trunc_mode not in ['single_ellipsis', 'all_ellipsis']:
            raise NotImplementedError(f"Truncation mode '{trunc_mode}' not recognized")
        final_str += right_brackets[row_idx] + '\n'
    str_header = {'polytope': f"Polytope in R^{n}"}.get(object_type)
    if str_header is None:
        raise NotImplementedError(f"Printing for {object_type} is not yet implemented.")
    return str_header + '\n' + final_str

def ellipsoid_semidef_repr(c: np.ndarray, Q: np.ndarray, include_tabs: bool = True, object_type: Literal['polyhedron', 'polytope', 'zonotope', 'cube'] = 'polytope', filter_sym_part: bool = True) -> str:
    n, edgeitems = Q.shape[1], np.get_printoptions()['edgeitems']
    with np.printoptions(edgeitems=edgeitems, threshold=edgeitems ** 2, linewidth=1000):
        if n == 1:
            Q_lines, c_lines = [Q.__str__().replace('[', '').replace(']', '')], [c.__str__().replace('[', '').replace(']', '')]
        else:
            Q, c = Q, np.atleast_2d(c).T
            Q_str, c_str = Q.__str__().replace(']', '] ').replace('] ] ', ']]'), c.__str__().replace(']', '] ').replace('] ] ', ']]')
            Q_lines, c_lines = Q_str.splitlines(), c_str.splitlines()
            Q_lines, c_lines = [elem if elem != ' ...' else elem + (' ' * (len(Q_lines[-1]) - 4)) for elem in Q_lines], [elem if elem != ' ...' else elem + (' ' * (len(c_lines[-1]) - 4)) for elem in c_lines]
    n_print_rows = len(Q_lines)
    if filter_sym_part:
        for row_idx in range(n_print_rows):
            row_elems = Q_lines[row_idx].split()
            for col_idx in range(len(row_elems)):
                if col_idx > row_idx and '...' not in row_elems[col_idx]:
                    # FIXME: This `re` seems to raise the error (also due to random matrix generation?): "AttributeError: 'NoneType' object has no attribute 'start'"
                    n_digits, start_idx = len(row_elems[col_idx].replace('[', '').replace(']', '')), re.search(r'\d', row_elems[col_idx]).start()
                    Q_lines[row_idx] = Q_lines[row_idx].replace(row_elems[col_idx][start_idx:(start_idx + n_digits)], (' ' * (n_digits // 2)) + '*' + (' ' * ((n_digits // 2) if n_digits % 2 == 1 else (n_digits // 2 - 1))))
    center_offset = 0 if n_print_rows <= 2 else 1
    tabs = ['  ' if include_tabs else '' for _ in range(n_print_rows)]
    c_labels = (['   '] * (n_print_rows - center_offset - 1)) + ['c: '] + (['   '] * center_offset)
    commas = (['  '] * (n_print_rows - center_offset - 1)) + [', '] + (['  '] * center_offset)
    Q_labels = (['   '] * (n_print_rows - center_offset - 1)) + ['Q: '] + (['   '] * center_offset)
    final_str = ''.join(tabs[row_idx] + c_labels[row_idx] + c_lines[row_idx] + commas[row_idx] + Q_labels[row_idx] + Q_lines[row_idx] + '\n' for row_idx in range(n_print_rows))
    str_header = {'ellipsoid': f"Ellipsoid in R^{n}"}.get(object_type)
    if str_header is None:
        raise NotImplementedError(f"Printing for {object_type} is not yet implemented.")
    return str_header + '\n' + final_str

# ------ SCRIPT ------

# Create an integer array
# int_array_A, int_array_b = np.random.randint(0, 10, size=(int_rows, int_cols)), np.random.randint(0, 10, size=(int_rows))
int_array_A, int_array_b = np.random.rand(int_rows, int_cols), np.random.rand(int_rows) * 1E-6

# Create a floating-point array
float_array_verts = np.random.rand(float_rows, float_cols) * 1E-6

# Create a positive semidefinite matrix for the ellipsoid
# c, Q_root = np.random.randint(1000, 2000, size=n), np.random.randint(1000, 2000, size=(n, n))
c, Q_root = np.random.rand(n), np.random.rand(n, n) * 1E-0
Q = Q_root.T @ Q_root  # Make sure Q is positive semidefinite

# ------ PRINT ------

# Print the polytope H-representation
print("H-representation:")
np.set_printoptions(suppress=False, precision=1, edgeitems=2)
print(poly_hrepr(int_array_A, int_array_b, object_type='polytope', include_tabs=True, trunc_mode='both_ellipsis', strip_double_b_brackets=False))  # I think I prefer single_ellipsis here, and not stripping double brackets

# Print the polytope V-representation
print("V-representation:")
np.set_printoptions(suppress=False, precision=1, edgeitems=3)
print(poly_vrepr(float_array_verts, object_type='polytope', include_tabs=True, trunc_mode='all_ellipsis', strip_double_brackets=False, max_nverts=max_nverts))

# Print the ellipsoid semidefinite representation
print("Ellipsoid semidefinite representation:")
np.set_printoptions(suppress=False, precision=3, edgeitems=2)
print(ellipsoid_semidef_repr(c, Q, include_tabs=True, filter_sym_part=True, object_type='ellipsoid'))