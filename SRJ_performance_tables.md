# Performance Tables for SRJ

## SRJ, N = 256, N×N = 65536

| Method | Machine | Time per timestep (s) | Relative Speedup vs Serial | Relative Speedup vs OpenMP |
|--------|---------|----------------------|----------------------------|----------------------------|
| **Serial** | Intel Xeon Gold Cascade Lake 6248R (1 core) | 0.0003 | 1.00 | 0.22 |
| **OpenMP** | Intel Xeon Gold Cascade Lake 6248R (48 cores) | 0.0001 | 4.55 | 1.00 |
| **OpenACC** | Nvidia a100 | 0.0001 | 3.09 | 0.68 |
| **CUDA** | Nvidia a100 | 0.0000 | 5.45 | 1.20 |

## SRJ, N = 512, N×N = 262144

| Method | Machine | Time per timestep (s) | Relative Speedup vs Serial | Relative Speedup vs OpenMP |
|--------|---------|----------------------|----------------------------|----------------------------|
| **Serial** | Intel Xeon Gold Cascade Lake 6248R (1 core) | 0.0017 | 1.00 | 0.05 |
| **OpenMP** | Intel Xeon Gold Cascade Lake 6248R (48 cores) | 0.0001 | 20.82 | 1.00 |
| **OpenACC** | Nvidia a100 | 0.0001 | 23.18 | 1.11 |
| **CUDA** | Nvidia a100 | 0.0001 | 33.64 | 1.62 |

## SRJ, N = 1024, N×N = 1048576

| Method | Machine | Time per timestep (s) | Relative Speedup vs Serial | Relative Speedup vs OpenMP |
|--------|---------|----------------------|----------------------------|----------------------------|
| **Serial** | Intel Xeon Gold Cascade Lake 6248R (1 core) | 0.0143 | 1.00 | 0.03 |
| **OpenMP** | Intel Xeon Gold Cascade Lake 6248R (48 cores) | 0.0004 | 36.28 | 1.00 |
| **OpenACC** | Nvidia a100 | 0.0001 | 99.70 | 2.75 |
| **CUDA** | Nvidia a100 | 0.0001 | 116.44 | 3.21 |

## SRJ, N = 2048, N×N = 4194304

| Method | Machine | Time per timestep (s) | Relative Speedup vs Serial | Relative Speedup vs OpenMP |
|--------|---------|----------------------|----------------------------|----------------------------|
| **OpenMP** | Intel Xeon Gold Cascade Lake 6248R (48 cores) | 0.0029 | N/A | 1.00 |
| **OpenACC** | Nvidia a100 | 0.0004 | N/A | 7.09 |
| **CUDA** | Nvidia a100 | 0.0004 | N/A | 6.83 |

## SRJ, N = 4096, N×N = 16777216

| Method | Machine | Time per timestep (s) | Relative Speedup vs Serial | Relative Speedup vs OpenMP |
|--------|---------|----------------------|----------------------------|----------------------------|
| **OpenMP** | Intel Xeon Gold Cascade Lake 6248R (48 cores) | 0.0141 | N/A | 1.00 |
| **OpenACC** | Nvidia a100 | 0.0014 | N/A | 9.89 |
| **CUDA** | Nvidia a100 | 0.0015 | N/A | 9.60 |

