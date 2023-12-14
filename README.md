# Matrix Multiplication Library (for now)

This matrix multiplication library is a high-performance implementation written in Zig. It utilizes block matrix multiplication, SIMD optimizations, and multithreading to achieve efficient matrix operations.

## Features

- **Block Matrix Multiplication**: Splits matrices into smaller blocks for improved cache utilization.
- **SIMD Optimizations**: Utilizes Single Instruction, Multiple Data (SIMD) operations for faster computation.
- **Multithreading Support**: Leverages threading for parallel computation, enhancing performance on multi-core processors.
- **Alignment Optimized**: Ensures data structures are cache-line aligned for efficient memory access.

## Requirements

- Zig compiler (version 11.0 or newer recommended)

## Installation

Clone the repository to your local machine:

```bash
git clone https://github.com/ethanthoma/ml-zig-lib.git matrix_multiplication-library
cd matrix-multiplication-library
```

Build the project using Zig:

```bash
zig build -Doptimize=ReleaseFast
```

## Usage

```zig
const Matrix = @import("matrix.zig").Matrix;

var matrixA = Matrix(f32).init(...); // Initialize matrix A
var matrixB = Matrix(f32).init(...); // Initialize matrix B

// Perform matrix multiplication
var result = matrixA.matmul(matrixB) catch unreachable;
```

## Benchmarks

There are no benchmarks ran yet. On a AMD Ryzen 7 5700G with Radeon Graphics CPU, I get 33-39 ms for multiplying two 1024x1024 matricies.

