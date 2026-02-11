#!/usr/bin/env python3
"""
Performance benchmark: cmtk_apply vs streamxform

Compares execution time for forward and inverse transforms.
"""

import os
import time
import pathlib
import platform
import subprocess

import numpy as np

from cmtk_apply import load_registration

# Configuration
REG_PATH = os.path.abspath("../JFRC2_FCWB.list")

_search_path = os.environ["PATH"]
_search_path = [i for i in _search_path.split(os.pathsep) if len(i) > 0]
_search_path += [
    "~/bin",
    "/usr/lib/cmtk/bin/",
    "/usr/local/lib/cmtk/bin",
    "/usr/local/bin",
    "/opt/local/bin",
    "/opt/local/lib/cmtk/bin/",
    "/Applications/IGSRegistrationTools/bin",
]

if platform.system() == "Windows":
    _search_path += [
        r"C:\cygwin64\usr\local\lib\cmtk\bin",
        r"C:\Program Files\CMTK-3.3\CMTK\lib\cmtk\bin",
    ]


def find_binary(tool: str = "streamxform") -> str:
    """Find directory with binaries."""
    for path in _search_path:
        path = pathlib.Path(path).absolute()
        if not path.is_dir():
            continue

        try:
            return next(path.glob(tool)).resolve()
        except StopIteration:
            continue
        except BaseException:
            raise
    return tool  # Fallback to relying on system PATH


STREAMXFORM = find_binary()


def generate_test_points(n_points: int) -> np.ndarray:
    """Generate test points within a reasonable domain."""
    np.random.seed(42)
    points = np.random.uniform(-20, 250, size=(n_points, 3))
    return points


def benchmark_streamxform(
    points: np.ndarray,
    affine_only: bool = False,
    inverse: bool = False,
    n_runs: int = 1,
) -> float:
    """Run streamxform and measure execution time."""
    if not os.path.exists(STREAMXFORM):
        return None

    times = []
    for _ in range(n_runs):
        args = [STREAMXFORM]
        if affine_only:
            args.append("--affine-only")
        args.append("--")
        if inverse:
            args.append("--inverse")
        args.append(REG_PATH)

        inp = "\n".join(f"{x:.6f} {y:.6f} {z:.6f}" for x, y, z in points) + "\n"

        start = time.perf_counter()
        subprocess.run(args, input=inp, text=True, capture_output=True, check=False)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return np.mean(times) * 1000  # Convert to ms


def benchmark_cmtk_apply(
    reg,
    points: np.ndarray,
    transform: str = "affine",
    inverse: bool = False,
    n_runs: int = 1,
) -> float:
    """Run cmtk_apply and measure execution time."""
    times = []

    # For inverse, pre-compute the forward-transformed points once
    if inverse:
        forward_points = reg.transform_points(points, transform=transform)

    for _ in range(n_runs):
        start = time.perf_counter()

        if inverse:
            if transform == "affine":
                reg.inverse_transform_points(forward_points, transform="affine")
            else:
                # For warp inverse, use original points as initial guess for faster convergence
                reg.inverse_transform_points(
                    forward_points,
                    transform="warp",
                    initial_guess=points,
                    max_iter=20,
                    tolerance=1e-6,
                    solver="auto",
                )
        else:
            reg.transform_points(points, transform=transform)

        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return np.mean(times) * 1000  # Convert to ms


def print_header(title):
    """Print a section header."""
    print(f"\n{title}")
    print("-" * 95)
    print(
        "  Operation                  N      streamxform   cmtk_apply   Speedup   sx/pt    ca/pt"
    )
    print("-" * 95)


def print_row(label, n_points, sx_time, ca_time, speedup):
    """Print a benchmark row."""
    if sx_time is None:
        print(
            f"  {label:<22s} {n_points:>5d}         N/A          N/A        N/A      N/A      N/A"
        )
    else:
        sx_per = sx_time * 1000 / n_points
        ca_per = ca_time * 1000 / n_points
        print(
            f"  {label:<22s} {n_points:>5d}  {sx_time:>10.4f}ms  {ca_time:>10.4f}ms  {speedup:>6.2f}x  {sx_per:>6.2f}µs  {ca_per:>6.2f}µs"
        )


# Main benchmark
print("=" * 95)
print("PERFORMANCE BENCHMARK: cmtk_apply vs streamxform".center(95))
print("=" * 95)
print(
    "NOTE: cmtk_apply warp inverse uses solver='auto' (>=500 points -> analytical, else numerical)"
)

test_sizes = [10, 50, 100, 500, 1000, 10000, 100000]

# Load registration once
reg = load_registration(REG_PATH)


def warmup_numba(reg):
    """Warm up Numba JIT to avoid including compilation time in benchmarks."""
    warmup_points = generate_test_points(10)
    reg.transform_points(warmup_points, transform="warp")
    reg.inverse_transform_points(
        reg.transform_points(warmup_points, transform="warp"),
        transform="warp",
        initial_guess=warmup_points,
        max_iter=2,
        tolerance=1e-6,
        solver="auto",
    )


warmup_numba(reg)


# Determine number of runs based on point count for efficiency
def get_n_runs(n_points: int) -> int:
    """Adaptive number of runs based on point count."""
    if n_points <= 1000:
        return 20
    elif n_points <= 10000:
        return 10
    else:
        return 1  # Very large batches only need 1 run


# Forward transforms
print_header("FORWARD TRANSFORMS")

for n_points in test_sizes:
    points = generate_test_points(n_points)
    n_runs = get_n_runs(n_points)

    # Affine forward
    sx_time = benchmark_streamxform(points, affine_only=True, n_runs=n_runs)
    ca_time = benchmark_cmtk_apply(reg, points, transform="affine", n_runs=n_runs)
    speedup = sx_time / ca_time if sx_time else 0
    print_row("Affine (forward)", n_points, sx_time, ca_time, speedup)

    # Warp forward
    sx_time = benchmark_streamxform(points, affine_only=False, n_runs=n_runs)
    ca_time = benchmark_cmtk_apply(reg, points, transform="warp", n_runs=n_runs)
    speedup = sx_time / ca_time if sx_time else 0
    print_row("Warp (forward)", n_points, sx_time, ca_time, speedup)

# Inverse transforms
print_header("INVERSE TRANSFORMS")

for n_points in test_sizes:
    points = generate_test_points(n_points)
    n_runs = get_n_runs(n_points)

    # Affine inverse
    sx_time = benchmark_streamxform(
        points, affine_only=True, inverse=True, n_runs=n_runs
    )
    ca_time = benchmark_cmtk_apply(
        reg, points, transform="affine", inverse=True, n_runs=n_runs
    )
    speedup = sx_time / ca_time if sx_time else 0
    print_row("Affine (inverse)", n_points, sx_time, ca_time, speedup)

    # Warp inverse
    sx_time = benchmark_streamxform(
        points, affine_only=False, inverse=True, n_runs=n_runs
    )
    ca_time = benchmark_cmtk_apply(
        reg, points, transform="warp", inverse=True, n_runs=n_runs
    )
    speedup = sx_time / ca_time if sx_time else 0
    print_row("Warp (inverse)", n_points, sx_time, ca_time, speedup)
