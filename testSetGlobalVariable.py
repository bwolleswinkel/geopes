"""Example script to demonstrate how `mp.set_tol` acts as a global setter and context manager. Note that, for implementation, we actually would need to split the following into three files: `config.py`, `mypackage.py`, and `main.py`. However, for demonstration purposes, we keep everything in one file here.

"""

# ====== config.py ======

"""Config file for global variables."""

from __future__ import annotations

# FROM: https://numpy.org/doc/stable/reference/generated/numpy.isclose.html
RTOL, ATOL = 1E-5, 1E-8

class Tolerance:
    def __call__(self, rtol: float, atol: float = None) -> Tolerance:
        global RTOL, ATOL
        self._prev_rtol, self._prev_atol = RTOL, ATOL  # NOTE: This is necessary for context manager use, as the `with` statement calls __call__ first (which is unwanted, and thus needs to be mitigated by storing the 'true' globals before this call is made).
        RTOL = rtol
        if atol is not None:
            ATOL = atol
        self._rtol, self._atol = rtol, atol  # Store for context manager use
        return self

    def __enter__(self) -> Tolerance:
        global RTOL, ATOL
        RTOL = self._rtol
        if self._atol is not None:
            ATOL = self._atol
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        global RTOL, ATOL
        RTOL, ATOL = self._prev_rtol, self._prev_atol

set_tol = Tolerance()

# ====== mypackage.py ======

import config
from config import set_tol

import numpy as np

def is_close(a, b):
    return np.isclose(a, b, rtol=config.RTOL, atol=config.ATOL)

# ====== main.py ======

import mypackage as mp

delta = 1E-5
a, b = 0.1, 0.1 + delta

print(f"rtol: {mp.config.RTOL}, atol: {mp.config.ATOL}")  # 1e-05 1e-08

print(mp.is_close(a, b))  # False

mp.set_tol(1E-3)  # acts as global setter
print(mp.is_close(a, b))  # True

with mp.set_tol(1E-9):  # acts as local context manager
    print(mp.is_close(a, b))  # False

print(mp.is_close(a, b))  # True