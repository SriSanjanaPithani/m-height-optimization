# M-Height Optimization

An algorithmic optimization project focused on finding systematic generator
matrices with small m-heights for binary linear error-correcting codes.

The project combines exact m-height computation using linear programming
with heuristic search techniques to efficiently explore candidate generator
matrices.

## Overview

For a generator matrix G = [I_k | P], the project searches for matrices
that minimize the maximum m-height across the required parameter combinations.

The m-height of a codeword is determined by the ratio between its m-th
largest and smallest nonzero absolute coordinate values.

The optimization process uses the exact m-height calculation as an
evaluation function while applying heuristic search to explore the space
of possible generator matrices.

## Algorithms and Techniques

- Linear programming for exact m-height computation
- Systematic generator matrix construction
- Beam search
- Local search
- Random restarts
- Matrix mutation strategies
- Simulated-annealing-style candidate acceptance
- Parallel processing for m-height calculations

## Optimization Approach

The search process generates candidate generator matrices and evaluates
their m-heights.

Promising candidates are retained while the matrix is modified through
different mutation strategies. Multiple candidate solutions are explored
simultaneously to reduce the likelihood of becoming trapped in a local
minimum.

Random restarts provide additional exploration of the search space, while
the exact linear-programming formulation is used to evaluate candidate
solutions.

## Technologies

- Python
- NumPy
- SciPy
- Linear Programming
- Multiprocessing

## Results

The project evaluates generator matrices across the required combinations
of code length, dimension, and m-height parameter.

The optimization process is designed to reduce the maximum m-height while
maintaining the required systematic generator-matrix structure.
```bash
git clone <your-repository-url>
cd m-height-optimization
