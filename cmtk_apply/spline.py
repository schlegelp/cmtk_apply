import os
import numba

import numpy as np

from dataclasses import dataclass
from typing import Iterable, Optional
from scipy.optimize import least_squares

from .affine import AffineXform

# Allow users to enable fastmath via environment variable for potentially
# faster interpolation at the cost of some precision and stability in edge cases.
FASTMATH = os.getenv("CMTK_APPLY_FASTMATH", "").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)


@dataclass
class SplineWarp:
    dims: np.ndarray
    domain: np.ndarray
    origin: np.ndarray
    coefficients: np.ndarray
    absolute: bool
    affine: Optional[AffineXform] = None

    @classmethod
    def from_dict(cls, data: dict, legacy: bool = False) -> "SplineWarp":
        affine = None
        if "affine_xform" in data:
            affine = AffineXform.from_dict(data["affine_xform"], legacy=legacy)

        dims = np.asarray(data["dims"], dtype=int)
        domain = np.asarray(data["domain"], dtype=float)
        origin = np.asarray(data["origin"], dtype=float)
        coeffs = np.asarray(data["coefficients"], dtype=float)
        absolute = bool(data.get("absolute", False))

        return cls(
            dims=dims,
            domain=domain,
            origin=origin,
            coefficients=coeffs,
            absolute=absolute,
            affine=affine,
        )

    @property
    def spacing(self) -> np.ndarray:
        return self.domain / (self.dims - 3)

    def apply(
        self,
        points: Iterable[Iterable[float]],
        absolute: Optional[bool] = None,
        allow_extrapolation: bool = True,
    ) -> np.ndarray:
        pts = np.asarray(points, dtype=float)
        if pts.ndim == 1:
            pts = pts[None, :]

        use_absolute = self.absolute if absolute is None else absolute
        spacing = self.spacing

        out = np.empty((len(pts), 3), dtype=float)

        for i, pt in enumerate(pts):
            result = _interpolate_point_numba_select(
                pt,
                self.origin,
                spacing,
                self.dims,
                self.coefficients,
                allow_extrapolation,
            )

            if result[3] < 0.5:  # Not valid (validity flag = 0)
                out[i] = np.array([np.nan, np.nan, np.nan], dtype=float)
            else:
                warped = result[:3]
                out[i] = warped if use_absolute else pt + warped

        return out

    def _interpolate_point(
        self, point: np.ndarray, allow_extrapolation: bool
    ) -> Optional[np.ndarray]:
        """Pure Python B-spline interpolation (for compatibility with iterative solvers).

        The Numba-compiled version is used in apply() for better performance.
        This method is kept for compatibility with code that needs the Optional[np.ndarray] return type.
        """
        result = _interpolate_point_numba_select(
            point,
            self.origin,
            self.spacing,
            self.dims,
            self.coefficients,
            allow_extrapolation,
        )

        if result[3] < 0.5:  # Not valid
            return None
        else:
            return result[:3]

    def solve_inverse(
        self,
        target_points: np.ndarray,
        initial_guess: Optional[np.ndarray] = None,
        max_iter: int = 15,
        tolerance: float = 1e-6,
        solver: str = "auto",
    ) -> np.ndarray:
        """Solve for input points given output (target) points using least-squares.

        Given output coordinates y, find input coordinates x such that apply(x) ≈ y.
        Uses iterative least-squares solver with either numerical or analytical Jacobian.

        Args:
            target_points: Output coordinates to invert (N, 3)
            initial_guess: Initial guess for input points. If None, uses target_points.
            max_iter: Maximum iterations for solver (default: 15, reduced from 50 for speed)
            tolerance: Convergence tolerance
            solver: 'auto' (default), 'numerical', or 'analytical'
                - 'auto': Uses a heuristic based on batch size
                - 'numerical': Uses finite differences (3 forward evals per iteration)
                - 'analytical': Uses analytical Jacobian (1 forward + 3 derivative evals per iteration)

        Returns:
            Input coordinates that approximately produce the target_points
        """
        target_pts = np.asarray(target_points, dtype=float)
        if target_pts.ndim == 1:
            target_pts = target_pts[None, :]

        if initial_guess is None:
            x0 = target_pts.copy()
        else:
            x0 = np.asarray(initial_guess, dtype=float)
            if x0.ndim == 1:
                x0 = x0[None, :]

        if solver not in ("auto", "numerical", "analytical"):
            raise ValueError("solver must be 'auto', 'numerical', or 'analytical'")

        solver_choice = solver
        if solver_choice == "auto":
            solver_choice = _choose_inverse_solver(len(target_pts))

        results = []
        for i, target_pt in enumerate(target_pts):
            # Use initial guess if available and not NaN
            if i < len(x0) and not np.any(np.isnan(x0[i])):
                guess_pt = x0[i]
            else:
                guess_pt = target_pt.copy()

            if solver_choice == "analytical":
                # Use analytical Jacobian
                def residual(x):
                    forward = self.apply([x], allow_extrapolation=True)[0]
                    return forward - target_pt

                def jacobian(x):
                    jac_result = _interpolate_derivative_numba_select(
                        x,
                        self.origin,
                        self.spacing,
                        self.dims,
                        self.coefficients,
                        allow_extrapolation=True,
                    )
                    if jac_result[9] < 0.5:  # Not valid
                        return None
                    # Return flattened 3x3 Jacobian
                    return jac_result[:9].reshape(3, 3)

                sol = least_squares(
                    residual,
                    guess_pt,
                    jac=jacobian,
                    max_nfev=max_iter
                    * 4,  # Slightly more evaluations since Jacobian is faster
                    xtol=tolerance,
                    ftol=tolerance,
                    gtol=tolerance,
                )
            else:
                # Use numerical Jacobian (default)
                def residual(x):
                    forward = self.apply([x], allow_extrapolation=True)[0]
                    return forward - target_pt

                sol = least_squares(
                    residual,
                    guess_pt,
                    max_nfev=max_iter * 3,
                    xtol=tolerance,
                    ftol=tolerance,
                    gtol=tolerance,
                )
            results.append(sol.x)

        return np.array(results, dtype=float)


def _choose_inverse_solver(n_points: int) -> str:
    if n_points >= 500:
        return "analytical"
    return "numerical"


def _bspline_weights(t: float) -> np.ndarray:
    t2 = t * t
    t3 = t2 * t
    w0 = (1 - 3 * t + 3 * t2 - t3) / 6.0
    w1 = (4 - 6 * t2 + 3 * t3) / 6.0
    w2 = (1 + 3 * t + 3 * t2 - 3 * t3) / 6.0
    w3 = t3 / 6.0
    return np.array([w0, w1, w2, w3], dtype=float)


@numba.njit
def _bspline_weights_numba(t: float) -> np.ndarray:
    """Numba-compiled B-spline weight calculation."""
    t2 = t * t
    t3 = t2 * t
    w0 = (1 - 3 * t + 3 * t2 - t3) / 6.0
    w1 = (4 - 6 * t2 + 3 * t3) / 6.0
    w2 = (1 + 3 * t + 3 * t2 - 3 * t3) / 6.0
    w3 = t3 / 6.0
    return np.array([w0, w1, w2, w3], dtype=np.float64)


@numba.njit
def _bspline_weights_derivative_numba(t: float) -> np.ndarray:
    """Numba-compiled B-spline weight derivative calculation.

    Computes d(w)/dt for cubic B-spline basis functions.
    """
    t2 = t * t
    dw0 = (-3 + 6 * t - 3 * t2) / 6.0
    dw1 = (-12 * t + 9 * t2) / 6.0
    dw2 = (3 + 6 * t - 9 * t2) / 6.0
    dw3 = (3 * t2) / 6.0
    return np.array([dw0, dw1, dw2, dw3], dtype=np.float64)


@numba.njit(fastmath=True)
def _bspline_weights_numba_fastmath(t: float) -> np.ndarray:
    t2 = t * t
    t3 = t2 * t
    w0 = (1 - 3 * t + 3 * t2 - t3) / 6.0
    w1 = (4 - 6 * t2 + 3 * t3) / 6.0
    w2 = (1 + 3 * t + 3 * t2 - 3 * t3) / 6.0
    w3 = t3 / 6.0
    return np.array([w0, w1, w2, w3], dtype=np.float64)


@numba.njit(fastmath=True)
def _bspline_weights_derivative_numba_fastmath(t: float) -> np.ndarray:
    t2 = t * t
    dw0 = (-3 + 6 * t - 3 * t2) / 6.0
    dw1 = (-12 * t + 9 * t2) / 6.0
    dw2 = (3 + 6 * t - 9 * t2) / 6.0
    dw3 = (3 * t2) / 6.0
    return np.array([dw0, dw1, dw2, dw3], dtype=np.float64)


@numba.njit
def _interpolate_point_numba(
    point: np.ndarray,
    origin: np.ndarray,
    spacing: np.ndarray,
    dims: np.ndarray,
    coeffs: np.ndarray,
    allow_extrapolation: bool,
) -> np.ndarray:
    """Numba-compiled B-spline interpolation for a single point.

    Returns a 4-element array: [x, y, z, valid] where valid=1 if the point
    was successfully interpolated, 0 if out of bounds.
    """
    # Compute parameter coordinates
    u = (point - origin) / spacing

    # Check bounds
    if not allow_extrapolation:
        if u[0] < 0 or u[0] >= dims[0] - 3:
            return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        if u[1] < 0 or u[1] >= dims[1] - 3:
            return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        if u[2] < 0 or u[2] >= dims[2] - 3:
            return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)

    # Base indices and fractional parts
    base_x = int(np.floor(u[0])) - 1
    base_y = int(np.floor(u[1])) - 1
    base_z = int(np.floor(u[2])) - 1

    t_x = u[0] - np.floor(u[0])
    t_y = u[1] - np.floor(u[1])
    t_z = u[2] - np.floor(u[2])

    # Compute weights
    wx = _bspline_weights_numba(t_x)
    wy = _bspline_weights_numba(t_y)
    wz = _bspline_weights_numba(t_z)

    # Accumulate interpolation
    nx, ny, nz = dims[0], dims[1], dims[2]
    result = np.zeros(3, dtype=np.float64)

    for dx in range(4):
        for dy in range(4):
            for dz in range(4):
                ix = base_x + dx
                iy = base_y + dy
                iz = base_z + dz

                # Clamp to valid range
                ix_clamped = max(0, min(ix, int(nx) - 1))
                iy_clamped = max(0, min(iy, int(ny) - 1))
                iz_clamped = max(0, min(iz, int(nz) - 1))

                # Check if clamped (out of bounds case)
                if not allow_extrapolation:
                    if ix != ix_clamped or iy != iy_clamped or iz != iz_clamped:
                        return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)

                w = wx[dx] * wy[dy] * wz[dz]
                idx = ix_clamped + int(nx) * (iy_clamped + int(ny) * iz_clamped)
                result += w * coeffs[idx]

    # Return result with validity flag (1.0 = valid)
    return np.array([result[0], result[1], result[2], 1.0], dtype=np.float64)


@numba.njit(fastmath=True)
def _interpolate_point_numba_fastmath(
    point: np.ndarray,
    origin: np.ndarray,
    spacing: np.ndarray,
    dims: np.ndarray,
    coeffs: np.ndarray,
    allow_extrapolation: bool,
) -> np.ndarray:
    u = (point - origin) / spacing

    if not allow_extrapolation:
        if u[0] < 0 or u[0] >= dims[0] - 3:
            return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        if u[1] < 0 or u[1] >= dims[1] - 3:
            return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        if u[2] < 0 or u[2] >= dims[2] - 3:
            return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)

    base_x = int(np.floor(u[0])) - 1
    base_y = int(np.floor(u[1])) - 1
    base_z = int(np.floor(u[2])) - 1

    t_x = u[0] - np.floor(u[0])
    t_y = u[1] - np.floor(u[1])
    t_z = u[2] - np.floor(u[2])

    wx = _bspline_weights_numba_fastmath(t_x)
    wy = _bspline_weights_numba_fastmath(t_y)
    wz = _bspline_weights_numba_fastmath(t_z)

    nx, ny, nz = dims[0], dims[1], dims[2]
    result = np.zeros(3, dtype=np.float64)

    for dx in range(4):
        for dy in range(4):
            for dz in range(4):
                ix = base_x + dx
                iy = base_y + dy
                iz = base_z + dz

                ix_clamped = max(0, min(ix, int(nx) - 1))
                iy_clamped = max(0, min(iy, int(ny) - 1))
                iz_clamped = max(0, min(iz, int(nz) - 1))

                if not allow_extrapolation:
                    if ix != ix_clamped or iy != iy_clamped or iz != iz_clamped:
                        return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)

                w = wx[dx] * wy[dy] * wz[dz]
                idx = ix_clamped + int(nx) * (iy_clamped + int(ny) * iz_clamped)
                result += w * coeffs[idx]

    return np.array([result[0], result[1], result[2], 1.0], dtype=np.float64)


@numba.njit
def _interpolate_derivative_numba(
    point: np.ndarray,
    origin: np.ndarray,
    spacing: np.ndarray,
    dims: np.ndarray,
    coeffs: np.ndarray,
    allow_extrapolation: bool,
) -> np.ndarray:
    """Numba-compiled B-spline gradient computation for a single point.

    Computes the Jacobian (3x3 gradient matrix) of the transformation.
    Returns a 10-element array: [J[0,0], J[0,1], J[0,2], J[1,0], J[1,1], J[1,2], J[2,0], J[2,1], J[2,2], valid]
    where J[i,j] = d(output[i])/d(input[j])
    """
    # Compute parameter coordinates
    u = (point - origin) / spacing

    # Check bounds
    if not allow_extrapolation:
        if u[0] < 0 or u[0] >= dims[0] - 3:
            return np.array(
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64
            )
        if u[1] < 0 or u[1] >= dims[1] - 3:
            return np.array(
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64
            )
        if u[2] < 0 or u[2] >= dims[2] - 3:
            return np.array(
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64
            )

    # Base indices and fractional parts
    base_x = int(np.floor(u[0])) - 1
    base_y = int(np.floor(u[1])) - 1
    base_z = int(np.floor(u[2])) - 1

    t_x = u[0] - np.floor(u[0])
    t_y = u[1] - np.floor(u[1])
    t_z = u[2] - np.floor(u[2])

    # Compute weights and derivatives
    wx = _bspline_weights_numba(t_x)
    wy = _bspline_weights_numba(t_y)
    wz = _bspline_weights_numba(t_z)

    dwx = _bspline_weights_derivative_numba(t_x)
    dwy = _bspline_weights_derivative_numba(t_y)
    dwz = _bspline_weights_derivative_numba(t_z)

    # Accumulate Jacobian components
    # J[i,j] = d(output[i])/d(input[j])
    # output[i] = sum over basis functions of: weight(u) * coeff[grid_index][i]
    # d(output[i])/d(input[j]) = sum of: (d(weight)/d(u[j]) / spacing[j]) * coeff[grid_index][i]

    nx, ny, nz = dims[0], dims[1], dims[2]
    jacobian = np.zeros((3, 3), dtype=np.float64)

    for dx in range(4):
        for dy in range(4):
            for dz in range(4):
                ix = base_x + dx
                iy = base_y + dy
                iz = base_z + dz

                # Clamp to valid range
                ix_clamped = max(0, min(ix, int(nx) - 1))
                iy_clamped = max(0, min(iy, int(ny) - 1))
                iz_clamped = max(0, min(iz, int(nz) - 1))

                # Check if clamped (out of bounds case)
                if not allow_extrapolation:
                    if ix != ix_clamped or iy != iy_clamped or iz != iz_clamped:
                        return np.array(
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            dtype=np.float64,
                        )

                idx = ix_clamped + int(nx) * (iy_clamped + int(ny) * iz_clamped)
                coeff_vec = coeffs[idx]  # 3D control point vector

                # Derivatives w.r.t. input dimension 0
                w_dy_dz = (wy[dy] * wz[dz]) / spacing[0]
                dw0 = dwx[dx] * w_dy_dz
                jacobian[0, 0] += dw0 * coeff_vec[0]
                jacobian[1, 0] += dw0 * coeff_vec[1]
                jacobian[2, 0] += dw0 * coeff_vec[2]

                # Derivatives w.r.t. input dimension 1
                w_dx_dz = (wx[dx] * wz[dz]) / spacing[1]
                dw1 = dwy[dy] * w_dx_dz
                jacobian[0, 1] += dw1 * coeff_vec[0]
                jacobian[1, 1] += dw1 * coeff_vec[1]
                jacobian[2, 1] += dw1 * coeff_vec[2]

                # Derivatives w.r.t. input dimension 2
                w_dx_dy = (wx[dx] * wy[dy]) / spacing[2]
                dw2 = dwz[dz] * w_dx_dy
                jacobian[0, 2] += dw2 * coeff_vec[0]
                jacobian[1, 2] += dw2 * coeff_vec[1]
                jacobian[2, 2] += dw2 * coeff_vec[2]

    # Return flattened Jacobian with validity flag
    return np.array(
        [
            jacobian[0, 0],
            jacobian[0, 1],
            jacobian[0, 2],
            jacobian[1, 0],
            jacobian[1, 1],
            jacobian[1, 2],
            jacobian[2, 0],
            jacobian[2, 1],
            jacobian[2, 2],
            1.0,
        ],
        dtype=np.float64,
    )


@numba.njit(fastmath=True)
def _interpolate_derivative_numba_fastmath(
    point: np.ndarray,
    origin: np.ndarray,
    spacing: np.ndarray,
    dims: np.ndarray,
    coeffs: np.ndarray,
    allow_extrapolation: bool,
) -> np.ndarray:
    u = (point - origin) / spacing

    if not allow_extrapolation:
        if u[0] < 0 or u[0] >= dims[0] - 3:
            return np.array(
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64
            )
        if u[1] < 0 or u[1] >= dims[1] - 3:
            return np.array(
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64
            )
        if u[2] < 0 or u[2] >= dims[2] - 3:
            return np.array(
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64
            )

    base_x = int(np.floor(u[0])) - 1
    base_y = int(np.floor(u[1])) - 1
    base_z = int(np.floor(u[2])) - 1

    t_x = u[0] - np.floor(u[0])
    t_y = u[1] - np.floor(u[1])
    t_z = u[2] - np.floor(u[2])

    wx = _bspline_weights_numba_fastmath(t_x)
    wy = _bspline_weights_numba_fastmath(t_y)
    wz = _bspline_weights_numba_fastmath(t_z)

    dwx = _bspline_weights_derivative_numba_fastmath(t_x)
    dwy = _bspline_weights_derivative_numba_fastmath(t_y)
    dwz = _bspline_weights_derivative_numba_fastmath(t_z)

    nx, ny, nz = dims[0], dims[1], dims[2]
    jacobian = np.zeros((3, 3), dtype=np.float64)

    for dx in range(4):
        for dy in range(4):
            for dz in range(4):
                ix = base_x + dx
                iy = base_y + dy
                iz = base_z + dz

                ix_clamped = max(0, min(ix, int(nx) - 1))
                iy_clamped = max(0, min(iy, int(ny) - 1))
                iz_clamped = max(0, min(iz, int(nz) - 1))

                if not allow_extrapolation:
                    if ix != ix_clamped or iy != iy_clamped or iz != iz_clamped:
                        return np.array(
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            dtype=np.float64,
                        )

                idx = ix_clamped + int(nx) * (iy_clamped + int(ny) * iz_clamped)
                coeff_vec = coeffs[idx]

                w_dy_dz = (wy[dy] * wz[dz]) / spacing[0]
                dw0 = dwx[dx] * w_dy_dz
                jacobian[0, 0] += dw0 * coeff_vec[0]
                jacobian[1, 0] += dw0 * coeff_vec[1]
                jacobian[2, 0] += dw0 * coeff_vec[2]

                w_dx_dz = (wx[dx] * wz[dz]) / spacing[1]
                dw1 = dwy[dy] * w_dx_dz
                jacobian[0, 1] += dw1 * coeff_vec[0]
                jacobian[1, 1] += dw1 * coeff_vec[1]
                jacobian[2, 1] += dw1 * coeff_vec[2]

                w_dx_dy = (wx[dx] * wy[dy]) / spacing[2]
                dw2 = dwz[dz] * w_dx_dy
                jacobian[0, 2] += dw2 * coeff_vec[0]
                jacobian[1, 2] += dw2 * coeff_vec[1]
                jacobian[2, 2] += dw2 * coeff_vec[2]

    return np.array(
        [
            jacobian[0, 0],
            jacobian[0, 1],
            jacobian[0, 2],
            jacobian[1, 0],
            jacobian[1, 1],
            jacobian[1, 2],
            jacobian[2, 0],
            jacobian[2, 1],
            jacobian[2, 2],
            1.0,
        ],
        dtype=np.float64,
    )


def _interpolate_point_numba_select(
    point: np.ndarray,
    origin: np.ndarray,
    spacing: np.ndarray,
    dims: np.ndarray,
    coeffs: np.ndarray,
    allow_extrapolation: bool,
) -> np.ndarray:
    if FASTMATH:
        return _interpolate_point_numba_fastmath(
            point, origin, spacing, dims, coeffs, allow_extrapolation
        )
    return _interpolate_point_numba(
        point, origin, spacing, dims, coeffs, allow_extrapolation
    )


def _interpolate_derivative_numba_select(
    point: np.ndarray,
    origin: np.ndarray,
    spacing: np.ndarray,
    dims: np.ndarray,
    coeffs: np.ndarray,
    allow_extrapolation: bool,
) -> np.ndarray:
    if FASTMATH:
        return _interpolate_derivative_numba_fastmath(
            point, origin, spacing, dims, coeffs, allow_extrapolation
        )
    return _interpolate_derivative_numba(
        point, origin, spacing, dims, coeffs, allow_extrapolation
    )
