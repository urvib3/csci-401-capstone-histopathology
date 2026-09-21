# Histopathology WCSP Capstone

AI methods for histopathology: labeling tissue regions using Weighted Constraint Satisfaction Problems (WCSPs) and related combinatorial optimization techniques.

## Sample Problems

For each problem, we identify the WCSP formulation, describe its combinatorial signature, solve an example with Toulbar2 when possible, and describe how Local Search or A* could approach the same problem outside the WCSP framework.

### Current Implementations

| Problem | Owner | File / Status |
| --- | --- | --- |
| N-Queens | Abhishek | `sample_problems/n_queens.cpp` |
| Sudoku | Rishi | `sample_problems/sudoku.py` |
| Crosswords | Kashvi | `sample_problems/crosswords.py` |
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

**Name:** Abhishek

## WCSP framework

Variables: A_1,A_2,A_3... A_n, where A_i = (x_i, y_i) the position of the ith queen.

Domain: 0 <= x_i < n, 0 <= y_i < n

Constraints: two queens A_i and A_j cannot be on the same row nor column, and they must not be attacking each other diagonally.

Arity = n since there are n queen positions that must be satisfied.


## Approach with backtracking

A standard approach is to use backtracking (essentially DFS with pruning) (check sample_problems/n_queens.cpp) where a queen is
placed on a row and then recursively placed on each following row for all possible positions. 
The algorithmic time complexity is O(N!). It is best to run backtracking for this problem where N is at max 9 queens.

## A*

It could also be oslved with local search or A*. We would just need a heuristic 
for A* when searching (i.e number of conflicting pairs of queens.).

## Local Search
For local search we could place all queens on the board and search there locally on removing a queen and the placement that minimizes the cost (conflicts).

## Approach with toulbar

source: https://toulbar2.github.io/toulbar2/examples/tuto_wnqp.html

followed this source to model the problem as a WCSP in cpp.

It models the n variables as queens on each column, and the domain is the row values the queen can take. the constraints are that queens may not be on the same row, nor on the same upper or lower diagonals. 

### The program can be run like so (replace abi with user)
```
$ g++ n_queens_toulbar.cpp \
  -I/home/abi/class/toulbar2/build/tb2config \
  -I/home/abi/class/toulbar2/src \
  -L/home/abi/class/toulbar2/build/lib/Linux \
  -Wl,-rpath,/home/abi/class/toulbar2/build/lib/Linux \
  -ltb2 -lgmp -o a.out && ./a.out 8

```
Output:

```
Solving for n=8
Reverse original DAC dual bound: 15 (+6.667%)
Cost function decomposition time : 0.000 seconds.
Preprocessing time: 0.011 seconds.
8 unassigned variables, 64 values in all current domains (med. size:8, max size:8) and 29 
non-unary cost functions (med. arity:2, med. degree:7)
Initial lower and upper bounds: [15, 65] 76.923%
Optimality gap: [16, 65] 75.385 % (3 backtracks, 6 nodes, 0.013 seconds)
New solution: 34 (3 backtracks, 10 nodes, depth 5, 0.014 seconds)
Optimality gap: [16, 34] 52.941 % (6 backtracks, 13 nodes, 0.015 seconds)
Optimality gap: [18, 34] 47.059 % (9 backtracks, 21 nodes, 0.018 seconds)
New solution: 30 (24 backtracks, 65 nodes, depth 4, 0.027 seconds)
Optimality gap: [18, 30] 40.000 % (26 backtracks, 67 nodes, 0.027 seconds)
New solution: 27 (35 backtracks, 95 nodes, depth 5, 0.033 seconds)
Optimality gap: [20, 27] 25.926 % (41 backtracks, 104 nodes, 0.036 seconds)
New solution: 23 (43 backtracks, 116 nodes, depth 4, 0.037 seconds)
Optimality gap: [21, 23] 8.696 % (45 backtracks, 119 nodes, 0.038 seconds)
Optimality gap: [22, 23] 4.348 % (45 backtracks, 122 nodes, 0.039 seconds)
Optimality gap: [23, 23] 0.000 % (45 backtracks, 125 nodes, 0.039 seconds)
Node redundancy during HBFS: 27.200
Optimum: 23 in 45 backtracks and 125 nodes ( 0 removals by DEE) and 0.039 seconds.
Optimal soln found with cost: 23

```

---

# Crosswords

## Problem

A crossword consists of horizontal and vertical word slots that intersect at shared cells.

The goal is to assign one word to every slot such that:

- Every assigned word has the correct length.
- Letters agree wherever two slots intersect.

## WCSP Formulation

Each horizontal or vertical word slot is represented by one WCSP variable.

For a crossword containing s slots:

- *Variables:* s, one variable for each word slot
- *Domains:* candidate words having the same length as the slot
- *Hard constraints:* intersecting words must contain the same letter at their shared cell
- *Cost function:*
  - matching word combinations have cost 0
  - conflicting word combinations are forbidden

A crossword is primarily a Constraint Satisfaction Problem. It can be represented as a WCSP in which every constraint is hard and every valid solution has total cost 0.

## Combinatorial Signature

Let d be the largest number of candidate words for any slot.

- *Number of variables:* s
- *Maximum domain size:* d
- *Crossing-letter constraint arity:* 2
- *Maximum arity:* 2

The primal graph contains one vertex for every word slot. Two vertices are connected when their corresponding slots intersect.

## Treewidth

The treewidth depends on the structure of the crossword's slot-intersection graph.

Crosswords with fewer interconnected slots generally have lower treewidth, while highly interconnected crosswords have higher treewidth.

For the included 2 × 2 example, the slot-intersection graph is a cycle containing four variables. Its treewidth is:

*2*

## Toulbar2

The sample crossword contains two across slots and two down slots:

```text
A T
N O
```

One valid solution returned by Toulbar2 is:

```text
A1 -> AT
A2 -> NO
D1 -> AN
D2 -> TO
```

Every crossing constraint is satisfied, so the total WCSP cost is:

*0*

Implementation:

`sample_problems/crosswords.py`

## Local Search

A Local Search state assigns one candidate word to every slot.

A neighboring state is generated by replacing the word assigned to one slot with another candidate word.

The objective function is:

*cost = number of conflicting letters at intersections*

The search keeps replacements that reduce this cost. It may become stuck in a local optimum.

## A*

A state represents a partially filled crossword.

- *g:* number or cost of conflicts created by the words assigned so far
- *h:* optimistic estimate of the conflicts or work remaining in the unfilled slots

A* expands the state with the smallest:

*f = g + h*

---

# Maximum Weighted Independent Set

**Owner:** Ryan

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

## Problem

The maximum weighted independent set is a graph problem in which an independent set of maximal weight must be extracted from a randomized graph, $G(V,E)$. The independent set problem is famously NP complete, and requires finding a maximally sized set of nodes S such that $\forall (u,v) \in S(\nexists e(u,v))$, or rather there is no edge between any two vertices in S. In the case of maximum weighted independent set, we assign weights $w_i$ to each vertex, $v\in V$, then select an independent set S such that we obtain $\max{\sum_{v \in S}w_{v}}$

## WCSP Formulation

Each vertex is represented as a WCSP variable, with value 0 or 1 indicating whether that vertex is chosen. If vertex $i$ is selected, a cost of $-w_i$ is incurred so that by minimising the sum of negative weights, we maximize the actual weight of the set. A constraint is added for all adjacent pairs of vertices such that if both are chosen, the function outputs wcsp.Top to invalidate that solution.

In the solution presented in MWIS_solver.py, the graph is initialized with 10000 vertices of random weight between 1 and 100. Edges between nodes are then created with probability 0.2. As such:
- **Variables:** [3,4,...,10000], per vertex, can be any positive integer $n>0$ in the general case
- **Domains:**  [0,1], yes or no selection
- **Cost:** Sum of negative weights of selected vertices $\sum_{v \in S}-w_v$
- **Hard constraints:** No adjecent vertices in S, so no two selected vertices may have an edge between them.

## Combinatorial Signature
- **Maximum arity:** 2 due to the edge constraint.
- **Number of cost functions:** $|V|+|E|$

The primal graph is exactly the input graph. The tree width grows linearly with n as it is a dense random graph, making it $\Theta(n)$.

## Local Search Formulation

We can assign a random initial selection onto $V$. A state is any subset $S \subseteq V$, stored as the same 0/1 vector used by the WCSP variables. Because a random selection will usually contain adjacent vertices, the hard edge constraint is relaxed into a penalty:

$$cost(S) = -\sum_{v \in S} w_v + \lambda \cdot |\{(u,v) \in E : u,v \in S\}|$$

with $\lambda > \max_v w_v$ (e.g. $\lambda = 101$ for the solver's weights). This is the WCSP objective with `wcsp.Top` replaced by a finite penalty.

- **Neighborhood:** flip one vertex in or out of $S$.
  - Removing an endpoint of a conflicting edge saves at least $\lambda$ and loses less than $\lambda$ in weight, so it always lowers the cost.
  - Adding a vertex with no selected neighbors lowers the cost by $w_v$.
  - Therefore every local optimum of the flip neighborhood is a *maximal independent set*, which is the same sanity check made by `verify_solution`.
- **Swap move:** add an unselected vertex $v$ and remove all of its selected neighbors. This improves the solution whenever $w_v > \sum_{u \in N(v) \cap S} w_u$, and lets the search move between independent sets without passing through an infeasible state. The reverse move (remove one vertex, add two non-adjacent vertices that it was blocking) is also useful.
- **Incremental evaluation:** keeping a counter of selected neighbors for every vertex makes the cost change of a move computable in $O(\deg(v))$ rather than re-evaluating the whole graph.

The search repeatedly applies improving moves until none remain. The result is a maximal independent set but not necessarily a *maximum* one, so random restarts, simulated annealing, or a tabu list (forbidding a recently removed vertex from re-entering) are needed to escape local optima. With $p = 0.2$ each vertex has about $0.2n$ neighbors, so independent sets are very small relative to $n$; a sparse random start (or the empty set) avoids spending most of the run repairing conflicts.

## A* Formulation

A state is a partial solution $(S, C)$, where $S$ is the independent set chosen so far and $C$ is the *candidate set*: undecided vertices with no neighbor in $S$. The start state is $(\emptyset, V)$ and a goal state is any state with $C = \emptyset$.

From a state, pick a branching vertex $v \in C$ (e.g. the one with the most neighbors in $C$) and generate two successors:

- **include $v$:** $S \leftarrow S \cup \{v\}$, $C \leftarrow C \setminus N[v]$
- **exclude $v$:** $C \leftarrow C \setminus \{v\}$

A* requires non-negative step costs, so the $-w_v$ costs of the WCSP cannot be used directly. Instead we minimize the weight that is *given up*: excluding $v$ costs $w_v$, and including $v$ costs the weight of its neighbors removed from $C$. At a goal state the path cost is $\sum_{v \in V} w_v - w(S)$, so the cheapest goal is exactly the maximum weight independent set.

- **g:** total weight of the vertices discarded so far, $w(V) - w(S) - w(C)$
- **h:** a lower bound on the weight that must still be discarded from $C$

An admissible $h$ comes from a greedy clique cover of the subgraph induced by $C$. An independent set can contain at most one vertex from each clique $K$, so at least everything except the heaviest vertex of each clique must be given up:

$$h(S, C) = \sum_{K} \left( w(K) - \max_{v \in K} w_v \right)$$

Using a matching instead of a clique cover gives the weaker but cheaper bound $\sum_{(u,v) \in M} \min(w_u, w_v)$, and $h = 0$ reduces A* to uniform-cost search.

A* expands the state with the smallest **f = g + h**, which is equivalent to expanding the state with the largest optimistic total $w(S) + UB(C)$. Since $h$ never overestimates, the first goal state expanded is optimal. Two states with the same candidate set $C$ have identical futures, so only the one with the larger $w(S)$ needs to be kept.

This is best-first branch and bound, closely related to what Toulbar2 does internally with its own lower bounds. The limitation is memory: the open list grows exponentially, so A* is only practical for small graphs and not for the 10000-vertex instances generated in `MWIS_solver.py`.

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
