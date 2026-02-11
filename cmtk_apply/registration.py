import os

import numpy as np

from dataclasses import dataclass
from typing import Iterable, List, Optional, Union

from .affine import AffineXform
from .spline import SplineWarp
from .typedstream import read_typedstream


@dataclass
class Registration:
    affine: Optional[AffineXform] = None
    spline: Optional[SplineWarp] = None
    version: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict, version: Optional[str] = None) -> "Registration":
        """Build a Registration from parsed TypedStream data."""
        legacy = _is_legacy_affine(version)

        affine = None
        if "affine_xform" in data:
            affine = AffineXform.from_dict(data["affine_xform"], legacy=legacy)

        spline = None
        if "spline_warp" in data:
            spline = SplineWarp.from_dict(data["spline_warp"], legacy=legacy)

        return cls(affine=affine, spline=spline, version=version)

    def transform_points(
        self,
        points: Iterable[Iterable[float]],
        transform: str = "warp",
        allow_extrapolation: bool = False,
        fallback_to_affine: bool = True,
    ) -> np.ndarray:
        """Apply the registration transform to input points.

        Args:
            points: Input coordinates (N, 3)
            transform: 'affine' or 'warp' - which transformation to apply
            allow_extrapolation: Whether to allow extrapolation beyond bounds (for warp)
            fallback_to_affine: Whether to fallback to affine if warp is invalid. Doesn't apply if extrapolation is allowed.

        Returns:
            Transformed coordinates (N, 3)
        """
        if transform not in ("warp", "affine"):
            raise ValueError("transform must be 'warp' or 'affine'")

        pts = np.asarray(points, dtype=float)
        if pts.ndim == 1:
            pts = pts[None, :]

        if transform == "affine" or self.spline is None:
            if self.affine is None:
                raise ValueError("No affine transform available")
            return self.affine.apply(pts)

        # For warp transform: evaluate spline at original points to get absolute coordinates
        warped = self.spline.apply(pts, allow_extrapolation=allow_extrapolation)

        if fallback_to_affine and np.any(np.isnan(warped)):
            if self.affine is None:
                raise ValueError("No affine transform available for fallback")
            mask = np.isnan(warped[:, 0])
            warped[mask] = self.affine.apply(pts)[mask]
        return warped

    def inverse_transform_points(
        self,
        points: Iterable[Iterable[float]],
        transform: str = "warp",
        initial_guess: Optional[np.ndarray] = None,
        max_iter: int = 15,
        tolerance: float = 1e-6,
        allow_extrapolation: bool = False,
        solver: str = "auto",
    ) -> np.ndarray:
        """Solve for input points given output (transformed) points.

        This inverts the transformation. For affine-only transformations, this is
        exact. For warp transformations, uses iterative solver (least-squares).

        Args:
            points: Output/transformed coordinates
            transform: 'affine' or 'warp' - which forward transform to invert
            initial_guess: Initial guess for solver (for warp). If None, uses input points.
            max_iter: Maximum iterations for iterative solver (default: 15, reduced from 50)
            tolerance: Convergence tolerance for iterative solver
            allow_extrapolation: Whether to allow extrapolation in warp inverse
            solver: 'auto' (default), 'numerical', or 'analytical' - Jacobian computation method

        Returns:
            Input/original coordinates
        """
        if transform not in ("warp", "affine"):
            raise ValueError("transform must be 'warp' or 'affine'")

        pts = np.asarray(points, dtype=float)
        if pts.ndim == 1:
            pts = pts[None, :]

        if transform == "affine" or self.spline is None:
            if self.affine is None:
                raise ValueError("No affine transform available")
            return self.affine.solve_inverse(pts)

        # For warp transform: use iterative solver
        if self.spline is None:
            raise ValueError("No spline warp available")

        return self.spline.solve_inverse(
            pts,
            initial_guess=initial_guess,
            max_iter=max_iter,
            tolerance=tolerance,
            solver=solver,
        )


@dataclass
class RegistrationChain:
    """Chain multiple registrations for sequential composition.

    Applies transformations in order: input → reg1 → reg2 → ... → output
    Inverse applies in reverse order: output → reg_n_inv → ... → reg1_inv → input
    """

    registrations: List[Registration]

    def transform_points(
        self,
        points: Iterable[Iterable[float]],
        transform: str = "warp",
        allow_extrapolation: bool = False,
        fallback_to_affine: bool = True,
    ) -> np.ndarray:
        """Apply all registrations in sequence.

        Args:
            points: Input coordinates
            transform: 'affine' or 'warp' - transformation type to apply
            allow_extrapolation: Whether to allow extrapolation beyond bounds
            fallback_to_affine: Whether to fallback to affine if warp is invalid

        Returns:
            Transformed coordinates after all registrations applied
        """
        result = np.asarray(points, dtype=float)
        if result.ndim == 1:
            result = result[None, :]

        for reg in self.registrations:
            result = reg.transform_points(
                result,
                transform=transform,
                allow_extrapolation=allow_extrapolation,
                fallback_to_affine=fallback_to_affine,
            )

        return result

    def inverse_transform_points(
        self,
        points: Iterable[Iterable[float]],
        transform: str = "warp",
        initial_guess: Optional[np.ndarray] = None,
        max_iter: int = 15,
        tolerance: float = 1e-6,
        allow_extrapolation: bool = False,
        solver: str = "auto",
    ) -> np.ndarray:
        """Invert all registrations in reverse sequence.

        Applies inverse transformations in reverse order of composition.
        For a chain reg1 → reg2, this computes reg2^{-1} ∘ reg1^{-1}.

        Args:
            points: Output/transformed coordinates
            transform: 'affine' or 'warp' - which forward transform to invert
            initial_guess: Initial guess for solver (for warp)
            max_iter: Maximum iterations for iterative solver (default: 15, reduced from 50)
            tolerance: Convergence tolerance for iterative solver
            allow_extrapolation: Whether to allow extrapolation in warp inverse
            solver: 'auto' (default), 'numerical', or 'analytical' - Jacobian computation method

        Returns:
            Original/input coordinates
        """
        result = np.asarray(points, dtype=float)
        if result.ndim == 1:
            result = result[None, :]

        # Apply in reverse order
        for reg in reversed(self.registrations):
            # For chain inverse, we don't provide initial_guess after first iteration
            # since we're backtracking through unknown intermediate spaces
            result = reg.inverse_transform_points(
                result,
                transform=transform,
                initial_guess=initial_guess if reg is self.registrations[0] else None,
                max_iter=max_iter,
                tolerance=tolerance,
                allow_extrapolation=allow_extrapolation,
                solver=solver,
            )

        return result


def load_registration(
    path: Union[str, List[str]],
) -> Union[Registration, RegistrationChain]:
    """Load one or more registrations.

    Args:
        path: Single registration path (str) or multiple paths (list of str)
              If a directory is provided, looks for 'registration' file inside

    Returns:
        Registration object if single path provided
        RegistrationChain object if list of paths provided
    """
    if isinstance(path, str):
        # Single registration
        reg_path = _resolve_registration_path(path)
        data, version = read_typedstream(reg_path)
        if "registration" not in data:
            raise ValueError("No registration block found")
        return Registration.from_dict(data["registration"], version=version)

    elif isinstance(path, (list, tuple)):
        # Multiple registrations - return a chain
        registrations = [load_registration(p) for p in path]
        return RegistrationChain(registrations)

    else:
        raise TypeError("path must be str or list of str")


def _resolve_registration_path(path: str) -> str:
    if os.path.isdir(path):
        reg_path = os.path.join(path, "registration")
        if os.path.exists(reg_path):
            return reg_path
        gz_path = reg_path + ".gz"
        if os.path.exists(gz_path):
            return gz_path
        raise FileNotFoundError(f"No registration file in {path}")
    return path


def _is_legacy_affine(version: Optional[str]) -> bool:
    if version is None:
        return False
    try:
        return tuple(int(v) for v in version.split(".")) < (2, 4)
    except ValueError:
        return False
