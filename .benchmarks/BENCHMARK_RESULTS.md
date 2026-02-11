# Benchmark Summary: cmtk_apply vs streamxform

## Overview
This document summarizes the performance characteristics of the cmtk_apply library compared to CMTK's reference implementation in the `streamxform` binary.

## Test Setup
- **N points tested**: 10, 50, 100, 500, 1000, 10,000, 100,000
- **Runs per test**: 20 (small), 10 (medium), 1 (large)
- **Registration**: JFRC2_FCWB.list (standard *Drosophila* brain template)
- **Point domain**: Uniformly random points in range [-20, 250]
- **JIT warmup**: Timings are measured after a warmup pass (Numba compilation excluded)
- **Fastmath (optional)**: `CMTK_APPLY_FASTMATH=1` yields ~1.5x warp forward speedup with max drift ~1e-13 (not used in below benchmarks)

## Results

Note: we're testing affine and warp transforms separately here. Affine transforms are blazingly fast anyway and the more significant test are warp transforms. For smaller point sets the speed-up from saving overhead from process startup and file I/O is very pronounced.

### Forward Transform Summary

#### Affine Forward Transform
| Points | streamxform | cmtk_apply | Speedup   |
|--------|-------------|------------|-----------|
| 10     | 13.0ms      | 0.021ms    | 611x      |
| 50     | 13.8ms      | 0.021ms    | 670x      |
| 100    | 14.0ms      | 0.020ms    | 709x      |
| 500    | 16.6ms      | 0.023ms    | 714x      |
| 1,000  | 20.3ms      | 0.038ms    | 541x      |
| 10,000 | 78.9ms      | 0.122ms    | 648x      |
| 100,000| 669.9ms     | 1.353ms    | **495x**  |

**Per-point**: ~6.70µs (streamxform) vs ~0.01µs (cmtk_apply) for 100k points

#### Warp Forward Transform
| Points | streamxform | cmtk_apply | Speedup |
|--------|------------|-----------|--------|
| 10     | 13.3ms     | 0.036ms   | 372x |
| 50     | 13.6ms     | 0.142ms   | 95x |
| 100    | 14.0ms     | 0.294ms   | 47.5x |
| 500    | 16.6ms     | 1.372ms   | 12.1x |
| 1,000  | 20.3ms     | 2.735ms   | 7.4x |
| 10,000 | 80.2ms     | 27.5ms    | 2.9x |
| 100,000| 664.2ms    | 277.2ms   | **2.4x** |

**Per-point**: ~6.64µs (streamxform) vs ~2.77µs (cmtk_apply) for 100k points

### Inverse Transform Summary

#### Affine Inverse Transform
| Points | streamxform | cmtk_apply | Speedup |
|--------|------------|-----------|---------|
| 10     | 14.0ms     | 0.025ms   | 567x    |
| 50     | 14.4ms     | 0.022ms   | 661x    |
| 100    | 14.8ms     | 0.022ms   | 677x    |
| 500    | 17.4ms     | 0.027ms   | 658x    |
| 1,000  | 21.9ms     | 0.040ms   | 552x    |
| 10,000 | 80.3ms     | 0.129ms   | 621x    |
| 100,000| 659.9ms    | 1.830ms   | **361x** |

**Per-point**: ~6.61µs (streamxform) vs ~0.02µs (cmtk_apply) for 100k points

#### Warp Inverse Transform
| Points | streamxform | cmtk_apply | Speedup   |
|--------|------------|-----------|--------|
| 10     | 15.0ms     | 0.944ms   | 15.9x |
| 50     | 16.7ms     | 4.576ms   | 3.7x |
| 100    | 19.8ms     | 9.135ms   | 2.2x |
| 500    | 39.7ms     | 44.175ms  | 0.90x |
| 1,000  | 64.6ms     | 52.666ms  | 1.23x |
| 10,000 | 509.9ms    | 522.651ms | 0.98x |
| 100,000| 5277.3ms   | 5246.0ms  | **1.01x** |

**Per-point**: ~52.77µs (streamxform) vs ~52.46µs (cmtk_apply) for 100k points