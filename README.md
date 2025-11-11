# MyPoisonCu

A high-performance 2D Poisson equation solver with support for multiple backends (CUDA, OpenACC, OpenMP, Serial) and various iterative solution methods.

## Overview

MyPoisonCu solves the 2D Poisson equation:
```
∇²φ = f
```
on a rectangular domain with configurable boundary conditions using finite difference methods.

## Features

- **Multiple Backends**: CUDA, OpenACC, OpenMP, and Serial execution
- **Multiple Solvers**: 
  - Jacobi iteration with relaxation
  - SRJ (Scheduled Relaxation Jacobi) with optimized relaxation parameters
  - BiCGSTAB (Biconjugate Gradient Stabilized) with preconditioning
- **Boundary Conditions**: Dirichlet, Neumann, and Periodic
- **Flexible Precision**: Single (float) or double precision support
- **RAII Design**: Automatic memory management with C++ classes
- **Unified Interface**: Single codebase for all backends

## Building

The project supports multiple build configurations:

```bash
# For CUDA
make -f Makefile.cuda

# For OpenACC
make -f Makefile.acc

# For OpenMP
make -f Makefile.omp

# For Serial
make -f Makefile.serial
```

### Build Options

- `USE_DOUBLE=1`: Enable double precision (default: single precision)
- `DEBUG=1`: Enable debug symbols and disable optimization

Example:
```bash
make -f Makefile.cuda USE_DOUBLE=1
```

## Usage

```bash
./mypoisoncu
```

The program reads configuration from `input.dat` and outputs:
- Solution files in Tecplot format to `TECOUT/` directory
- Convergence history to `HISTORY/` directory

## Performance

The solver automatically selects optimal thread/block configurations for GPU execution and efficiently handles data transfers between host and device memory. The unified calculator interface ensures consistent performance across all backends.

## Architecture

The code uses a modular design with:
- **Calculators**: Core computational kernels with backend-specific optimizations
- **Solvers**: Implementation of iterative methods
- **Data Structures**: RAII-based Field, Operator, and Solver classes
- **I/O**: Tecplot output and convergence history tracking
