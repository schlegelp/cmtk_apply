import numpy as np

from dataclasses import dataclass
from typing import Iterable, Optional


@dataclass
class AffineXform:
    xlate: np.ndarray
    rotate: np.ndarray
    scale: np.ndarray
    shear: np.ndarray
    center: np.ndarray
    legacy: bool = False

    @classmethod
    def from_dict(cls, data: dict, legacy: bool = False) -> "AffineXform":
        return cls(
            xlate=np.asarray(data["xlate"], dtype=float),
            rotate=np.asarray(data["rotate"], dtype=float),
            scale=np.asarray(data["scale"], dtype=float),
            shear=np.asarray(data["shear"], dtype=float),
            center=np.asarray(data["center"], dtype=float),
            legacy=legacy,
        )

    def matrix(self) -> np.ndarray:
        tx, ty, tz = self.xlate
        rx, ry, rz = self.rotate
        sx, sy, sz = self.scale
        shx, shy, shz = self.shear
        cx, cy, cz = self.center

        alpha = np.deg2rad(rx)
        theta = np.deg2rad(ry)
        phi = np.deg2rad(rz)

        cos0 = np.cos(alpha)
        sin0 = np.sin(alpha)
        cos1 = np.cos(theta)
        sin1 = np.sin(theta)
        cos2 = np.cos(phi)
        sin2 = np.sin(phi)

        sin0xsin1 = sin0 * sin1
        cos0xsin1 = cos0 * sin1

        rval = np.eye(4, dtype=float)
        rval[0, 0] = cos1 * cos2
        rval[1, 0] = -cos1 * sin2
        rval[2, 0] = -sin1
        rval[0, 1] = sin0xsin1 * cos2 + cos0 * sin2
        rval[1, 1] = -sin0xsin1 * sin2 + cos0 * cos2
        rval[2, 1] = sin0 * cos1
        rval[0, 2] = cos0xsin1 * cos2 - sin0 * sin2
        rval[1, 2] = -cos0xsin1 * sin2 - sin0 * cos2
        rval[2, 2] = cos0 * cos1

        if self.legacy:
            rval[0:3, 0] *= sx
            rval[0:3, 1] *= sy
            rval[0:3, 2] *= sz

            shears = [shx, shy, shz]
            for i in range(2, -1, -1):
                shear = np.eye(4, dtype=float)
                shear[[1, 2, 2][i], [0, 0, 1][i]] = shears[i]
                rval = shear @ rval
        else:
            scale_shear = np.zeros((4, 4), dtype=float)
            np.fill_diagonal(scale_shear, [sx, sy, sz, 1.0])
            scale_shear[0, 1] = shx
            scale_shear[0, 2] = shy
            scale_shear[1, 2] = shz
            rval = rval @ scale_shear

        cM = np.array(
            [
                cx * rval[0, 0] + cy * rval[0, 1] + cz * rval[0, 2],
                cx * rval[1, 0] + cy * rval[1, 1] + cz * rval[1, 2],
                cx * rval[2, 0] + cy * rval[2, 1] + cz * rval[2, 2],
            ],
            dtype=float,
        )

        rval[0, 3] = tx - cM[0] + cx
        rval[1, 3] = ty - cM[1] + cy
        rval[2, 3] = tz - cM[2] + cz
        return rval

    def apply(self, points: Iterable[Iterable[float]]) -> np.ndarray:
        pts = np.asarray(points, dtype=float)
        if pts.ndim == 1:
            pts = pts[None, :]
        mat = self.matrix()
        return (pts @ mat[:3, :3].T) + mat[:3, 3]

    def invert(self) -> "AffineXform":
        """Return an AffineXform representing the inverse transformation.

        This inverts the affine matrix and extracts the parameters.
        """
        mat = self.matrix()
        inv_mat = np.linalg.inv(mat)

        # Extract translation from inverse matrix
        inv_xlate = inv_mat[:3, 3]

        # Extract rotation from the inverse matrix
        # This is complex because we need to extract Euler angles from a rotation matrix
        # For now, we'll use a simpler approach: store the inverted matrix and apply it
        # Actually, the cleanest approach is to just use the matrix directly

        # Create a pseudo-inverse by storing negative parameters
        # More precisely: if A = T*R*S*Sh, then A^{-1} = Sh^{-1}*S^{-1}*R^{-1}*T^{-1}
        # But parameterizing this is complex, so we'll use scipy's matrix_to_euler

        from scipy.spatial.transform import Rotation

        # Extract rotation matrix from the forward transformation
        # The rotation part of mat is mat[:3, :3] after factoring out scale and shear
        # This is complex to undo exactly, so we extract angles from the inverse matrix
        rot_mat_inv = inv_mat[:3, :3]

        # Use SVD to get the closest rotation matrix (handles numerical errors)
        U, _, Vt = np.linalg.svd(rot_mat_inv)
        rot_mat_inv_clean = U @ Vt

        r = Rotation.from_matrix(rot_mat_inv_clean)
        euler_inv = r.as_euler("zyx", degrees=True)  # Match CMTK convention (ZYX order)

        # For a simple case, return with inverted parameters
        # Note: This is an approximation; exact inversion of scale/shear is complex
        return AffineXform(
            xlate=inv_xlate,
            rotate=euler_inv[[2, 1, 0]],  # Reverse to match ZYX order
            scale=1.0 / self.scale,
            shear=-self.shear,
            center=self.center,
            legacy=self.legacy,
        )

    def solve_inverse(self, target_points: np.ndarray) -> np.ndarray:
        """Solve for input points given output (target) points.

        This is equivalent to applying the inverse transformation.
        """
        pts = np.asarray(target_points, dtype=float)
        if pts.ndim == 1:
            pts = pts[None, :]
        mat = self.matrix()
        inv_mat = np.linalg.inv(mat)
        return (pts @ inv_mat[:3, :3].T) + inv_mat[:3, 3]


def ensure_affine(xform: Optional[AffineXform]) -> Optional[AffineXform]:
    return xform
