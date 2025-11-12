# Performance Tables for BICGSTAB

## BICGSTAB, N = 256, N×N = 65536

| Method | Machine | Time per timestep (s) | Relative Speedup vs Serial | Relative Speedup vs OpenMP |
|--------|---------|----------------------|----------------------------|----------------------------|
| **Serial** | Intel Xeon Gold Cascade Lake 6248R (1 core) | 0.0010 | 1.00 | 0.21 |
| **OpenMP** | Intel Xeon Gold Cascade Lake 6248R (48 cores) | 0.0002 | 4.78 | 1.00 |
| **OpenACC** | Nvidia a100 | 0.0004 | 2.53 | 0.53 |
| **CUDA** | Nvidia a100 | 0.0003 | 3.23 | 0.68 |

## BICGSTAB, N = 512, N×N = 262144

| Method | Machine | Time per timestep (s) | Relative Speedup vs Serial | Relative Speedup vs OpenMP |
|--------|---------|----------------------|----------------------------|----------------------------|
| **Serial** | Intel Xeon Gold Cascade Lake 6248R (1 core) | 0.0072 | 1.00 | 0.05 |
| **OpenMP** | Intel Xeon Gold Cascade Lake 6248R (48 cores) | 0.0004 | 20.14 | 1.00 |
| **OpenACC** | Nvidia a100 | 0.0004 | 18.93 | 0.94 |
| **CUDA** | Nvidia a100 | 0.0004 | 19.88 | 0.99 |

## BICGSTAB, N = 1024, N×N = 1048576

| Method | Machine | Time per timestep (s) | Relative Speedup vs Serial | Relative Speedup vs OpenMP |
|--------|---------|----------------------|----------------------------|----------------------------|
| **Serial** | Intel Xeon Gold Cascade Lake 6248R (1 core) | 0.0655 | 1.00 | 0.03 |
| **OpenMP** | Intel Xeon Gold Cascade Lake 6248R (48 cores) | 0.0017 | 38.03 | 1.00 |
| **OpenACC** | Nvidia a100 | 0.0006 | 101.34 | 2.66 |
| **CUDA** | Nvidia a100 | 0.0006 | 117.53 | 3.09 |

## BICGSTAB, N = 2048, N×N = 4194304

| Method | Machine | Time per timestep (s) | Relative Speedup vs Serial | Relative Speedup vs OpenMP |
|--------|---------|----------------------|----------------------------|----------------------------|
| **Serial** | Intel Xeon Gold Cascade Lake 6248R (1 core) | 0.2387 | 1.00 | 0.10 |
| **OpenMP** | Intel Xeon Gold Cascade Lake 6248R (48 cores) | 0.0231 | 10.32 | 1.00 |
| **OpenACC** | Nvidia a100 | 0.0017 | 142.82 | 13.83 |
| **CUDA** | Nvidia a100 | 0.0023 | 102.99 | 9.97 |

## BICGSTAB, N = 4096, N×N = 16777216

| Method | Machine | Time per timestep (s) | Relative Speedup vs Serial | Relative Speedup vs OpenMP |
|--------|---------|----------------------|----------------------------|----------------------------|
| **Serial** | Intel Xeon Gold Cascade Lake 6248R (1 core) | 0.9342 | 1.00 | 0.06 |
| **OpenMP** | Intel Xeon Gold Cascade Lake 6248R (48 cores) | 0.0535 | 17.46 | 1.00 |
| **OpenACC** | Nvidia a100 | 0.0054 | 174.37 | 9.99 |
| **CUDA** | Nvidia a100 | 0.0067 | 138.57 | 7.94 |

