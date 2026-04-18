# Constraint-Aware Metaheuristics for a Vehicle Routing Problem with Time Windows

This repository contains the final implementation of a large-scale optimization project on a Vehicle Routing Problem with Time Windows (VRPTW) with heterogeneous vehicle families and time-dependent travel times.

The objective is not limited to route distance: each solution combines fixed vehicle rental costs, fuel costs, and a route-dispersion penalty. The final solver relies on a hybrid metaheuristic combining greedy construction, Large Neighborhood Search, local search operators, and Simulated Annealing.

## Project overview

The problem involves:
- Customer-specific demands and service times.
- Time windows that must be respected.
- Multiple vehicle families with different capacities and cost structures.
- Time-dependent travel times.
- A composite cost function including rental, fuel, and radius-related costs.

Because the exact problem is computationally difficult, the project focuses on high-quality heuristic design rather than exact optimization on the original formulation.

## Method

The final approach combines several components:

- **Greedy initialization** based on time-window ordering.
- **Vehicle-family selection** during construction and repair.
- **Feasibility repair** by removing the most critical customers from infeasible routes.
- **Large Neighborhood Search (LNS)** with ruin-and-recreate steps.
- **Slack-aware insertion** to avoid building fragile routes.
- **Local search operators** (2-opt, Relocate, Cross-exchange).
- **Route elimination** to reduce fleet size and fixed rental costs.
- **Simulated Annealing acceptance** to escape poor local minima.

## Lower bound

To evaluate solution quality, the project also computes a lower bound on a relaxed version of the problem:
- A **rental lower bound**, formulated as a fleet-cost covering problem.
- A **fuel lower bound**, obtained from a degree-2 relaxation.

These relaxed problems are solved with **HiGHS**, allowing us to report a conservative optimality gap.

## Final results

The final heuristic obtained:
- **Best total cost:** `27675.84`
- **Weighted gap vs relaxed lower bounds:** `18.20%`

## Repository structure

```text
.
├── README.md
├── src/
│   └── solver.cpp
├── docs/
│   ├── project_report.pdf
│   └── problem_statement.pdf
└── results/
    └── final_results.csv
```

## Build

A simple compilation command is:

```bash
g++ -O3 -std=c++17 -march=native src/solver.cpp -o solver
```

## Usage

Example:

```bash
./solver --instances_dir data/instances --time_limit 600 --seed 11
```

## Parameters

| Parameter | Description |
|---|---|
| `--instances_dir` | Directory containing the instance files |
| `--time_limit` | Time limit per instance in seconds |
| `--seed` | Random seed |

## Author

Noé Hagelauer
