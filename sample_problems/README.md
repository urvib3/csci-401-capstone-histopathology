# Histopathology WCSP Capstone

AI methods for histopathology: labeling tissue regions using Weighted Constraint Satisfaction Problems (WCSPs) and related combinatorial optimization techniques.

## Sample Problems

For each problem, we identify the WCSP formulation, describe its combinatorial signature, solve an example with Toulbar2 when possible, and describe how Local Search or A* could approach the same problem outside the WCSP framework.

### Current Implementations

| Problem | Owner | File / Status |
| --- | --- | --- |
| N-Queens | Abhishek | `sample_problems/n_queens.cpp` |
| Sudoku | Rishi | `sample_problems/sudoku.py` |
| Crosswords | Kashvi | TODO |
| Maximum Weighted Independent Set | Ryan | `sample_problems/MWIS/MWIS_solver.py` |
| Minimum Spanning Tree | Aakanksha | `sample_problems/minimum_spanning_tree.py` |
| Traveling Salesman Problem | Urvi | `sample_problems/traveling_salesman.py` |

---

# Sudoku

## Problem

Sudoku consists of a 9 × 9 grid that must be filled with the digits 1 through 9 such that:

- every row contains each digit exactly once,
- every column contains each digit exactly once,
- every 3 × 3 box contains each digit exactly once,
- and the original clues remain unchanged.

## WCSP Formulation

Each grid cell is represented by one WCSP variable.

For a standard Sudoku:

- **Variables:** 81, one variable for each cell
- **Domains:** `{1, ..., 9}` for empty cells
- **Clue cells:** have a domain containing only their fixed value
- **Hard constraints:**
  - all-different for every row
  - all-different for every column
  - all-different for every 3 × 3 box
- **Cost function:**
  - valid assignments have cost 0
  - assignments violating a hard constraint are forbidden

Sudoku is primarily a Constraint Satisfaction Problem rather than an optimization problem. A CSP can be viewed as a special case of a WCSP in which every constraint is hard, so all valid solutions have total cost 0.

## Combinatorial Signature

- **Number of variables:** 81
- **Maximum domain size:** 9
- **All-different constraint arity:** 9
- **Maximum arity:** 9

The primal graph connects two cell variables whenever they occur together in a row, column, or 3 × 3 box constraint.

Each row, column, and box forms a clique of 9 variables. The exact treewidth depends on the chosen encoding, but the graph contains many overlapping cliques and is therefore much more highly connected than a path or tree.

## Toulbar2

The implementation creates 81 variables and uses Toulbar2's `salldiff` global constraint for every row, column, and 3 × 3 box.

For the included sample Sudoku, Toulbar2 returns a valid solution with total WCSP cost:

**0**

Implementation:

`sample_problems/sudoku.py`

## Local Search

A completed grid can be used as the starting state.

Values can be swapped within rows or boxes in an attempt to reduce the number of duplicate values.

A possible objective function is:

**cost = number of row, column, and box conflicts**

The search repeatedly chooses changes that reduce this cost.

A limitation is that Local Search may become stuck in a local optimum.

## A*

A state can represent a partially completed Sudoku grid.

- **g:** cost associated with assignments already made
- **h:** estimate of the remaining conflicts or work needed to complete the grid

A* can be formulated for Sudoku, but constraint propagation and backtracking are generally more natural.

---

# Minimum Spanning Tree

## Problem

Given a connected, edge-weighted, undirected graph G = (V, E), the Minimum Spanning Tree problem finds a subset of n - 1 edges that:

- connects every vertex,
- contains no cycles,
- and minimizes total edge weight.

## WCSP Formulation

A parent-variable encoding is used.

For every graph vertex v, define a variable P_v, where:

**P_v = u**

means that vertex u is the parent of v in a rooted spanning tree.

For a graph containing n vertices:

- **Variables:** n, one parent variable per vertex
- **Domains:** each parent variable can take a vertex index from `0` to `n-1`
- **Hard constraint:** all parent choices together must form one valid spanning tree
- **Cost function:** choosing u as the parent of v costs the weight of edge (u, v)
- **Objective:** minimize the total cost of all selected parent edges

One vertex is chosen as the root and points to itself.

## Combinatorial Signature

- **Number of variables:** n
- **Maximum domain size:** n
- **Unary edge-cost function arity:** 1
- **Global MST constraint arity:** n
- **Maximum arity:** n

The total objective is:

**minimize the sum of w(v, P_v) for all non-root vertices v**

where r is the chosen root.

## Treewidth

Under the standard primal-graph interpretation, the global MST constraint contains all n parent variables in the same scope.

This connects every pair of variables and produces a complete graph:

**K_n**

The treewidth of a complete graph K_n is:

**n - 1**

Therefore, for this parent-variable encoding:

**treewidth = n - 1**

Treewidth is encoding-dependent, so a different representation using auxiliary variables and smaller constraints could produce a different primal graph.

## Toulbar2 Example

The sample graph contains four vertices:

**V = {A, B, C, D}**

with the following weighted edges:

| Edge | Weight |
| --- | ---: |
| A-B | 1 |
| B-C | 2 |
| A-D | 3 |
| A-C | 4 |
| B-D | 5 |
| C-D | 6 |

A is chosen as the root.

Toulbar2 returns:

```text
A -> A
B -> A
C -> B
D -> A
```

This corresponds to the selected edges:

**(A, B), (B, C), (A, D)**

with total cost:

**1 + 2 + 3 = 6**

Implementation:

`sample_problems/minimum_spanning_tree.py`

## Local Search

A Local Search state is any valid spanning tree.

A neighboring solution can be generated using an edge-exchange operation:

1. Add an edge that is not currently in the tree.
2. Adding the edge creates exactly one cycle.
3. Remove another edge from that cycle.
4. The result is another valid spanning tree.
5. Keep the new tree if its total edge weight is lower.

This allows Local Search to explore nearby spanning trees while maintaining feasibility.

## A*

A state can represent a partial acyclic forest.

- **g:** total weight of the edges already selected
- **h:** a lower bound on the additional cost required to connect the remaining components

A* expands the state with the lowest:

**f = g + h**

A* can be formulated for MST, but it is unnecessary in practice because MST can be solved exactly in polynomial time using algorithms such as:

- Kruskal's algorithm
- Prim's algorithm
- Borůvka's algorithm

---

# Traveling Salesman Problem

## Problem

Given n cities and distances between them, the Traveling Salesman Problem asks for a minimum-cost tour that:

- begins at a city,
- visits every city exactly once,
- and returns to the starting city.

## WCSP Formulation

City 0 is fixed as the starting city.

The remaining n - 1 positions in the tour are represented by WCSP variables.

Let X_p represent the city visited at position p.

The formulation has:

- **Variables:** n - 1, one variable per tour position
- **Domain size:** n - 1, representing all cities except city 0
- **Hard constraint:** all tour-position variables must contain different cities
- **Cost functions:**
  - distance from city 0 to the first stop
  - distance between consecutive tour positions
  - distance from the final stop back to city 0

The objective is to minimize total tour distance.

## Combinatorial Signature

- **Number of variables:** n - 1
- **Domain size:** n - 1
- **Unary distance-cost arity:** 1
- **Consecutive-distance function arity:** 2
- **All-different constraint arity:** n - 1
- **Maximum arity:** n - 1

## Treewidth

The consecutive-distance cost functions alone create a path-like structure.

However, the all-different constraint contains all n - 1 tour-position variables in the same scope.

Under the primal-graph interpretation, this produces the complete graph:

**K_(n-1)**

Therefore, for this encoding:

**treewidth = n - 2**

## Toulbar2

The implementation uses:

- unary costs for the first and final tour edges
- binary cost functions for travel between consecutive positions
- Toulbar2's `salldiff` global constraint to ensure that every city is visited only once

Implementation:

`sample_problems/traveling_salesman.py`

## Local Search

A common Local Search technique for TSP is **2-opt**.

Starting from a complete tour:

1. Choose two edges.
2. Remove them.
3. Reverse the section of the tour between those edges.
4. Reconnect the tour.
5. Keep the change if it reduces the total distance.

The process repeats until no 2-opt move improves the tour.

The result may be a local optimum rather than the globally optimal tour.

## A*

A state is a partial tour.

- **g:** total distance traveled so far
- **h:** optimistic lower bound on the remaining distance

A useful heuristic can be constructed using:

- a Minimum Spanning Tree over the unvisited cities
- a minimum-cost connection from the current city
- a minimum-cost connection back to the starting city

A* always expands the state with the smallest:

**f = g + h**

With an admissible heuristic, A* can find the optimal TSP tour, although the search space remains exponential.

---

# N-Queens

**Owner:** Abhishek

**Status:** TODO

This section will include:

- WCSP formulation
- variables and domains
- column and diagonal constraints
- domain size
- constraint arities
- treewidth
- Toulbar2 example
- Local Search / min-conflicts approach
- A* or backtracking formulation

---

# Crosswords

**Owner:** Kashvi

**Status:** TODO

This section will include:

- WCSP formulation
- one variable per word slot
- candidate-word domains
- crossing-letter constraints
- domain sizes
- constraint arities
- treewidth
- Toulbar2 example
- Local Search approach
- A* approach

---

# Maximum Weighted Independent Set

**Owner:** Ryan

**Status:** TODO

This section will include:

- WCSP formulation
- one Boolean variable per graph vertex
- domain size
- hard independence constraints
- vertex-weight objective
- constraint arities
- treewidth
- Toulbar2 example
- Local Search approach
- A* approach

---

# Setup

Create and activate a Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install Toulbar2:

```bash
pip install pytoulbar2
```

Run an implementation, for example:

```bash
python3 sample_problems/minimum_spanning_tree.py
```

```bash
python3 sample_problems/sudoku.py
```

```bash
python3 sample_problems/traveling_salesman.py
```
