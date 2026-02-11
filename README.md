# cmtk-apply

[![Tests](https://github.com/schlegelp/cmtk_apply/actions/workflows/tests.yml/badge.svg)](https://github.com/schlegelp/cmtk_apply/actions/workflows/tests.yml)

Fast, accurate and user-friendly Python interface for applying [CMTK (Computational Morphometry Toolkit)](https://www.nitrc.org/projects/cmtk) transformations, without requiring subprocess calls to external command-line tools.

Thanks to the use of `numba`, we are on par for inverse transforms and ~2x faster for forward transforms compared to CMTK's `streamxform` tool. For smaller point sets the speed-up is even more dramatic due to eliminating overhead from process startup and file I/O. See [BENCHMARK_RESULTS.md](.benchmarks/BENCHMARK_RESULTS.md) for details.

## Installation

### From PyPI

```bash
pip install cmtk-apply
```

### From Github

```bash
pip install git+https://github.com/schlegelp/cmtk_apply
```

### From source with development dependencies

Clone the repository and run:

```bash
pip install -e ".[dev]"
```

## Quick start

```python
from cmtk_apply import load_registration

# Load a registration (directory or file path)
reg = load_registration("JFRC2_FCWB.list")

# Forward transform points
points = [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]]
forward = reg.transform_points(points)
print(forward)

# Inverse transform
inverse = reg.inverse_transform_points(forward)
print(inverse)
```

## Optional Fastmath

Enable fastmath for Numba kernels by setting the environment variable:

```bash
export CMTK_APPLY_FASTMATH=1
```

Expected impact (warp forward): ~1.5x speedup with max absolute drift around 1e-13 on typical inputs.
Restart the Python process after changing the variable.

## API Reference

### `load_registration(path: str) -> Registration`

Load a CMTK registration from a `.list` folder or `registration` file.

**Parameters:**
- `path`: Path to a CMTK registration directory (`.list` folder) or a `registration` file (optionally gzipped).

**Returns:** A `Registration` object.

### `Registration.transform_points(...) -> np.ndarray`

Apply the registration transformation to points.

```python
output = reg.transform_points(
    points,                      # N×3 array-like of 3D points
    transform="warp",            # "warp" (default) or "affine"
    allow_extrapolation=True,   # Allow evaluation outside spline grid
    fallback_to_affine=True,     # Use affine if spline fails
)
```

**Parameters:**
- `points`: N×3 array or list of 3D coordinates
- `transform`: `"warp"` evaluates both affine and spline; `"affine"` evaluates affine only
- `allow_extrapolation`: If `False`, points outside the spline domain return NaN
- `fallback_to_affine`: If `True`, points that fail spline evaluation fall back to affine

**Returns:** N×3 NumPy array of transformed points

### `Registration.inverse_transform_points(...) -> np.ndarray`

Apply the inverse transformation to points.

```python
output = reg.inverse_transform_points(
    points,                      # N×3 array-like of 3D points
    transform="warp",            # "warp" (default) or "affine"
    allow_extrapolation=True,   # Allow evaluation outside spline grid
    fallback_to_affine=True,     # Use affine if spline fails
    solver="auto",               # "auto" (default), "analytical" or "numerical"
)
```

**Parameters:**
- `points`: N×3 array or list of 3D coordinates
- `transform`: `"warp"` evaluates both affine and spline; `"affine"` evaluates affine only
- `allow_extrapolation`: If `False`, points outside the spline domain return NaN
- `fallback_to_affine`: If `True`, points that fail spline evaluation fall back to affine
- `solver`: Method for solving inverse warp; `"auto"` uses analytical for >500 points and "numerical" otherwise

**Returns:** N×3 NumPy array of inverse transformed points

## Implementation Details

### Spline Warp

The library interprets the `absolute` flag in the spline warp configuration:
- `absolute yes`: Coefficients represent absolute positions → final = spline(affine_transformed_point)
- `absolute no`: Coefficients represent displacements → final = affine_transformed_point + spline_displacement

### Affine Composition

CMTK affine parameters are decomposed into:
- **xlate**: Translation (tx, ty, tz)
- **rotate**: Rotation angles in degrees (rx, ry, rz)
- **scale**: Anisotropic scaling (sx, sy, sz)
- **shear**: Shear components (shx, shy, shz)
- **center**: Center of rotation (cx, cy, cz)

The matrix composition follows CMTK's post-2.4.0 convention. Legacy (<2.4.0) behavior is auto-detected from the TypedStream version field.

### B-spline Basis

The cubic B-spline basis functions for parameter *t* ∈ [0,1] are:

$$w_0(t) = \frac{(1 - 3t + 3t^2 - t^3)}{6}$$

$$w_1(t) = \frac{(4 - 6t^2 + 3t^3)}{6}$$

$$w_2(t) = \frac{(1 + 3t + 3t^2 - 3t^3)}{6}$$

$$w_3(t) = \frac{t^3}{6}$$

A 4×4×4 neighborhood of control points is evaluated for each query point using tensor products of these weights.

## Testing

Run tests with pytest:

```bash
pytest tests/
```

To compare against CMTK's `streamxform` tool (if available):

```bash
pytest -v tests/test_cmtk_xf.py::test_affine_matches_streamxform
pytest -v tests/test_cmtk_xf.py::test_warp_matches_streamxform
```

Tests skip automatically if `streamxform` is not in the expected location.

## Example: JFRC2 to FCWB Registration

The `JFRC2_FCWB.list/` folder contains a pre-computed registration from the Drosophila reference brain JFRC2 to the FCWB template. This registration includes:
- An affine component (12 DOF)
- A 59×27×11 B-spline warp grid with deformation coefficients

## File Format

CMTK's TypedStream format is a text-based nested structure:

```
! TYPEDSTREAM 2.4

registration {
  reference_study "..."
  floating_study "..."
  affine_xform {
    xlate 0 0 0
    rotate 0 0 0
    ...
  }
  spline_warp {
    affine_xform { ... }
    absolute yes
    dims 59 27 11
    origin -11.36 -13.24 -16.87
    domain 636.39 317.88 134.99
    coefficients ...
  }
}
```

## Limitations

- This library focuses on the core transformation logic; it does not reformat images (use CMTK's `reformatx` for that).
- The `active` field in spline warps is parsed but not used; all control points are weighted equally (matches behavior of CMTK's `streamxform`).

## References

- [CMTK on NITRC](https://www.nitrc.org/projects/cmtk/)
- [nat (R package) CMTK I/O](https://github.com/natverse/nat/blob/HEAD/R/cmtk_io.R)
- [Drosophila brain templates (JFRC2, FCWB)](https://github.com/jefferislab/BridgingRegistrations)

## License

MIT
