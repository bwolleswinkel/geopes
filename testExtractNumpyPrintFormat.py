"""Script to test the extraction of NumPy print format settings."""

import re
from typing import Literal
import warnings

import numpy as np

# ------ PARAMETERS ------

# Set the options for the integer array
int_rows, int_cols = 3, 3

# Set the options for the floating array
float_rows, float_cols = 3, 100

# Set the options for ellipsoid
n = 6

# ------ METHOD ------

def poly_hrepr(A: np.ndarray, b: np.ndarray, A_eq: np.ndarray | None = None, B_eq: np.ndarray | None = None, object_type: Literal['polyhedron', 'polytope', 'zonotope', 'cube'] = 'polytope', include_tabs: bool = True, trunc_mode: Literal['single_ellipsis', 'both_ellipsis'] = 'both_ellipsis', strip_double_b_brackets: bool = True) -> str:
    #: Extract the dimensionality
    n = A.shape[1]
    #: Extract the Numpy print format with the right printing options
    # TODO: Move this to a config file?
    # FIXME: If the line width is more then the array fits on one line, the truncation does not work as expected: it will insert line breaks, which will mess up the formatting
    with np.printoptions(threshold=25, edgeitems=2, linewidth=1000):
        #: Flatten array is single row
        if A.shape[0] == 1:
            A, b = A.flatten(), b.flatten()
        else:
            A, b = A, np.atleast_2d(b).T
        A_np_str, b_np_str = A.__str__(), b.__str__()
    A_str_lines = A_np_str.splitlines()
    b_str_lines = b_np_str.splitlines()
    #: Check if we need to strip double brackets from b
    if strip_double_b_brackets:
        b_str_lines = [line.replace('[[', '[').replace(' ', '').replace(']]', ']') for line in b_str_lines]
    #: Retrieve the shape
    n_rows, n_cols = len(A_str_lines), len(A_str_lines[0].split())
    #: Join a space at the end of each entry on the A array except for the last line
    if n_rows > 1:
        A_str_lines = [line + ' ' for line in A_str_lines[:-1]] + [A_str_lines[-1]]
    #: Check if the there is a truncation index for the rows/columns
    row_idx_trunc = None if not any('...' == line.strip(' ') for line in A_str_lines) else next(idx for idx, line in enumerate(A_str_lines) if '...' == line.strip(' '))
    column_idx_trunc = None if not any('...' == item.strip(' ') for item in A_str_lines[0].split()) else next(idx for idx, item in enumerate(A_str_lines[0].split()) if '...' == item.strip(' '))
    #: Create the 'tab' string  # NOTE: A \t is way too wide
    tab_str_lines = ['  ' for _ in range(n_rows)] if include_tabs else ['' for _ in range(n_rows)]
    #: Create the intermediate 'x' string
    center_offset_bottom = 0 if n_rows <= 2 else 1
    x_str_lines = ([' |'] * (n_rows - center_offset_bottom - 1)) + [' x'] + ([' |'] * center_offset_bottom)
    #: Print the intermediate '<=' string
    ineq_str_lines = (['    '] * (n_rows - center_offset_bottom - 1)) + [' <= '] + (['    '] * center_offset_bottom)
    #: Print the ending return string
    return_str_lines = ['\n' for _ in range(n_rows)]
    #: Construct the final string line by line
    final_str = ''
    for row_idx in range(n_rows):
        if row_idx_trunc is not None and row_idx == row_idx_trunc:
            # I think I prefer single_ellipsis here
            match trunc_mode:
                case 'single_ellipsis':
                    final_str += tab_str_lines[row_idx] + A_str_lines[row_idx] + '\n'
                case 'both_ellipsis':
                    # FIXME: Why five lines of spaces?
                    final_str += tab_str_lines[row_idx] + A_str_lines[row_idx] + (' ' * (len(A_str_lines[-1]) - 5)) + x_str_lines[row_idx] + ineq_str_lines[row_idx] + b_str_lines[row_idx] + return_str_lines[row_idx]
                case _:
                    raise NotImplementedError(f"Truncation mode '{trunc_mode}' not recognized")
        else:
            final_str += tab_str_lines[row_idx] + A_str_lines[row_idx] + x_str_lines[row_idx] + ineq_str_lines[row_idx] + b_str_lines[row_idx] + return_str_lines[row_idx]
    #: Create the header string
    match object_type:
        case 'polyhedron':
            raise NotImplementedError("Printing for polyhedra is not yet implemented.")
        case 'polytope':
            str_header = f"Polytope in R^{n}"
        case 'zonotope':
            raise NotImplementedError("Printing for zonotopes is not yet implemented.")
        case 'cube':
            raise NotImplementedError("Printing for cubes is not yet implemented.")
    # TODO: Do we want to add equalities AFTER the inequalities? Or below?
    # Return the final string
    return str_header + '\n' + final_str

def poly_vrepr(verts: np.ndarray, rays: np.ndarray | None = None, object_type: Literal['polyhedron', 'polytope', 'zonotope', 'cube'] = 'polytope', include_tabs: bool = True, trunc_mode: Literal['single_ellipsis', 'all_ellipsis'] = 'both_ellipsis', strip_double_brackets: bool = True, max_nverts: int = 5) -> str:
    #: Check the parameters
    if max_nverts < 3:
        raise NotImplementedError("max_nverts must be at least 3 to allow for truncation.")
    #: Extract the dimensionality
    n = verts.shape[0]
    #: Extract the print options
    edgeitems = np.get_printoptions()['edgeitems']
    #: Extract the Numpy print format with the right printing options
    with np.printoptions(edgeitems=edgeitems, threshold=(2 * edgeitems + 1), linewidth=1000):
        verts_np_str = [np.atleast_2d(vert).T.__str__().splitlines() for vert in verts.T] if n > 1 else [vert.T.__str__().splitlines() for vert in verts.T]
    #: Extract the shape
    n_rows, n_cols = len(verts_np_str[0]), np.min([len(verts_np_str), max_nverts])
    #: Check if the columns need to be truncated
    row_trunc, column_trunc = n > n_rows, n_cols < verts.shape[1]
    #: Calculate the starting and ending row for column truncation
    if column_trunc:
        start_col_idx = max_nverts // 2
        end_col_idx = start_col_idx + 1
    else:
        start_col_idx, end_col_idx = None, None
    if not strip_double_brackets:
        verts_np_str = [[elem.replace(']', '] ').replace('] ] ', ']]') for elem in row] for row in verts_np_str] if n > 1 else verts_np_str
    else:
        verts_np_str = [['[' + elem.replace('[', '').replace(']', '') + ']' for elem in row] for row in verts_np_str]
        warnings.warn("Stripping double brackets is NOT correctly implemented at the moment")
    #: Create the 'tab' string  # NOTE: A \t is way too wide
    tab_str_lines = ['  ' for _ in range(n_rows)] if include_tabs else ['' for _ in range(n_rows)]
    #: Create the beginning set string
    middle_row = n_rows // 2
    begin_str_lines = [' /'] + ([' |'] * (middle_row - 1)) + (['< '] if n_rows != 2 else []) + ([' |'] * (n_rows - middle_row - 2)) + [' \\'] if n_rows != 1 else ['<']
    #: Create the intermediate commas string
    center_offset_bottom = 0 if n_rows <= 2 else 1
    comma_str_lines = [' ' + (' ' if n_rows == 1 else '') for _ in range(n_rows - center_offset_bottom - 1)] + [',' + (' ' if n_rows == 1 else '')] + [' ' + (' ' if n_rows == 1 else '') for _ in range(center_offset_bottom)] if n_cols > 1 else (['  '] if n_rows == 2 else [])
    #: Create the ending set string
    end_str_lines = ['\\ '] + (['| '] * (middle_row - 1)) + ([' >'] if n_rows != 2 else []) + (['| '] * (n_rows - middle_row - 2)) + ['/ '] if n_rows != 1 else ['>']
    #: Create the ending return string
    return_str_lines = ['\n' for _ in range(n_rows)]
    #: Construct the final string line by line
    final_str = ''
    for row_idx in range(n_rows):
        if row_trunc and row_idx == edgeitems:
            match trunc_mode:
                case 'single_ellipsis':
                    # FIXME: How do we determine the number of spaces here?
                    n_spaces = sum(len(verts_np_str[idx][0]) for idx in range(n_cols)) + (n_cols - 1) - 4
                    final_str += tab_str_lines[row_idx] + begin_str_lines[row_idx] + ' ...' + (' ' * n_spaces) + end_str_lines[row_idx] + return_str_lines[row_idx]
                case 'all_ellipsis':
                    # FIXME: Now, this is just a one-on-one copy
                    final_str += tab_str_lines[row_idx] + begin_str_lines[row_idx] 
                    for col_idx in range(n_cols):
                        if column_trunc and col_idx == start_col_idx:
                            final_str += (' ...' if row_idx == (n_rows - center_offset_bottom - 1) else '    ') + comma_str_lines[row_idx]
                        elif column_trunc and col_idx > start_col_idx and col_idx < end_col_idx:
                            continue
                        elif column_trunc:
                            final_str += ' ...' + (' ' * (len(verts_np_str[col_idx][0]) - 4)) + (comma_str_lines[row_idx] if col_idx < n_cols - 1 else '')
                        else:
                            final_str += ' ...' + (' ' * (len(verts_np_str[col_idx][0]) - 4)) + (comma_str_lines[row_idx] if col_idx < n_cols - 1 else '')
                    final_str += end_str_lines[row_idx] + return_str_lines[row_idx]
                case _:
                    raise NotImplementedError(f"Truncation mode '{trunc_mode}' not recognized")
        else:
            final_str += tab_str_lines[row_idx] + begin_str_lines[row_idx] 
            for col_idx in range(n_cols):
                if column_trunc and col_idx == start_col_idx:
                    final_str += (' ...' if row_idx == (n_rows - center_offset_bottom - 1) else '    ') + comma_str_lines[row_idx]
                elif column_trunc and col_idx > start_col_idx and col_idx < end_col_idx:
                    continue
                elif column_trunc:
                    final_str += verts_np_str[-(max_nverts - col_idx)][row_idx] + (comma_str_lines[row_idx] if col_idx < n_cols - 1 else '')
                else:
                    final_str += verts_np_str[col_idx][row_idx] + (comma_str_lines[row_idx] if col_idx < n_cols - 1 else '')
            final_str += end_str_lines[row_idx] + return_str_lines[row_idx]
    #: Create the header string
    match object_type:
        case 'polyhedron':
            raise NotImplementedError("Printing for polyhedra is not yet implemented.")
        case 'polytope':
            str_header = f"Polytope in R^{n}"
        case 'zonotope':
            raise NotImplementedError("Printing for zonotopes is not yet implemented.")
        case 'cube':
            raise NotImplementedError("Printing for cubes is not yet implemented.")
    # Return the final string
    return str_header + '\n' + final_str

def ellipsoid_semidef_repr(c: np.ndarray, Q: np.ndarray, include_tabs: bool = True, object_type: Literal['polyhedron', 'polytope', 'zonotope', 'cube'] = 'polytope', filter_sym_part: bool = True) -> str:
    #: Extract the dimensionality
    n = Q.shape[1]
    #: Extract the Numpy print format with the right printing options
    # TODO: Move this to a config file?
    # FIXME: If the line width is more then the array fits on one line, the truncation does not work as expected: it will insert line breaks, which will mess up the formatting
    #: Extract the print options
    edgeitems = np.get_printoptions()['edgeitems']
    #: Extract the Numpy print format with the right printing options
    with np.printoptions(edgeitems=edgeitems, threshold=edgeitems ** 2, linewidth=1000):
        #: Flatten array is single row
        if n == 1:
            Q_str_lines, c_str_lines = [Q.__str__().replace('[', '').replace(']', '')], [c.__str__().replace('[', '').replace(']', '')]
        else:
            Q, c = Q, np.atleast_2d(c).T
            Q_np_str, c_np_str = Q.__str__().replace(']', '] ').replace('] ] ', ']]'), c.__str__().replace(']', '] ').replace('] ] ', ']]')
            Q_str_lines, c_str_lines = Q_np_str.splitlines(), c_np_str.splitlines()
            Q_str_lines = [elem if elem != ' ...' else elem + (' ' * (len(Q_str_lines[-1]) - 4)) for elem in Q_str_lines]
            c_str_lines = [elem if elem != ' ...' else elem + (' ' * (len(c_str_lines[-1]) - 4)) for elem in c_str_lines]
    #: Retrieve the shape
    n_rows = len(Q_str_lines)
    #: Replace the symmetric part with `*` symbol and the appropriate number of spaces
    if filter_sym_part:
        for row_idx in range(n_rows):
            row_elems = Q_str_lines[row_idx].split()
            for col_idx in range(len(row_elems)):
                if col_idx > row_idx:
                    if '...' in row_elems[col_idx]:
                        continue
                    # FIXME: I very often get buggy behavior of `re` when running this script twice/multiple times in succession: "AttributeError: 'NoneType' object has no attribute 'start'"
                    n_digits, start_idx = len(row_elems[col_idx].replace('[', '').replace(']', '')), re.search(r'\d', row_elems[col_idx]).start()
                    # TODO: Make it such that this aligns nicely in the center of the decimal places
                    Q_str_lines[row_idx] = Q_str_lines[row_idx].replace(row_elems[col_idx][start_idx:(start_idx + n_digits)], (' ' * (n_digits // 2)) + '*' + (' ' * ((n_digits // 2) if n_digits % 2 == 1 else (n_digits // 2 - 1))))
    #: Create the 'tab' string  # NOTE: A \t is way too wide
    tab_str_lines = ['  ' for _ in range(n_rows)] if include_tabs else ['' for _ in range(n_rows)]
    #: Create the beginning c: string
    center_offset_bottom = 0 if n_rows <= 2 else 1
    begin_str_lines = (['   '] * (n_rows - center_offset_bottom - 1)) + ['c: '] + (['   '] * center_offset_bottom)
    #: Create the intermediate comma string
    comma_str_lines = (['  '] * (n_rows - center_offset_bottom - 1)) + [', '] + (['  '] * center_offset_bottom)
    #: Create the beginning Q: string
    begin_Q_str_lines = (['   '] * (n_rows - center_offset_bottom - 1)) + ['Q: '] + (['   '] * center_offset_bottom)
    #: Create the ending return string
    return_str_lines = ['\n' for _ in range(n_rows)]
    #: Construct the final string line by line
    final_str = ''
    for row_idx in range(n_rows):
        final_str += tab_str_lines[row_idx] + begin_str_lines[row_idx] + c_str_lines[row_idx] + comma_str_lines[row_idx] + begin_Q_str_lines[row_idx] + Q_str_lines[row_idx] + return_str_lines[row_idx]
    #: Create the header string
    match object_type:
        case 'ellipsoid':
            str_header = f"Ellipsoid in R^{n}"
        case 'sphere':
            raise NotImplementedError("Printing for sphere is not yet implemented.")
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
np.set_printoptions(suppress=False, precision=2, edgeitems=1)
print(poly_hrepr(int_array_A, int_array_b, object_type='polytope', include_tabs=True, trunc_mode='single_ellipsis', strip_double_b_brackets=False))  # I think I prefer single_ellipsis here, and not stripping double brackets

# Print the polytope V-representation
print("V-representation:")
np.set_printoptions(suppress=False, precision=1, edgeitems=3)
print(poly_vrepr(float_array_verts, object_type='polytope', include_tabs=True, trunc_mode='all_ellipsis', strip_double_brackets=False, max_nverts=5))

# Print the ellipsoid semidefinite representation
print("Ellipsoid semidefinite representation:")
np.set_printoptions(suppress=False, precision=3, edgeitems=2)
print(ellipsoid_semidef_repr(c, Q, include_tabs=True, filter_sym_part=True, object_type='ellipsoid'))