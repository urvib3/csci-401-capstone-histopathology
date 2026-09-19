# Traveling Salesman Problem as a WCSP (Urvi)
#
# TSP: visit every city once, minimize total distance.
#
# WCSP signature (city 0 fixed as start; variable p = city visited at position p):
#   variables:  n-1, one per stop in the tour
#   domains:    each stop can be any of the n-1 other cities
#   hard:       all-different across all the stops (no city twice)
#   soft:       d(0, first stop), d(stop p, stop p+1), d(last stop, 0)
#   treewidth:  n-2. The distance costs only link neighbors (a path), but
#               ensuring unique stops links EVERY pair of next possible stops. 
#               There are n-1 possible next stops, so tree width is n-2.
#
# Local search (2-opt): start with any tour, then keep reversing a chunk of it
#   if that makes it shorter. Stop when nothing helps (a local optimum).
#
# A*: build the tour one city at a time. g = distance so far,
#   h = optimistic guess of what's left (MST over unvisited cities).
#   Always expand the lowest g+h. Gives the best tour, but exponential.

import random

import pytoulbar2

# makes a fake TSP problem so we have something to solve.
# returns a table where dist[a][b] = random whole-number distance from city a to b
# (same both ways, seed makes it the same every run)
def make_instance(n, seed=0):
    rng = random.Random(seed)
    dist = [[0] * n for _ in range(n)]
    for a in range(n):
        for b in range(a + 1, n):
            dist[a][b] = dist[b][a] = rng.randint(1, 100)
    return dist


# finds the shortest tour using toulbar2.
# takes the distance table, builds the WCSP (variables + constraints below),
# lets toulbar2 solve it, and returns (tour, total distance).
# tour looks like [0, 6, 4, ..., 0]: start at city 0, hit every city, come back.
def solve_toulbar2(dist):
    n = len(dist)
    m = n - 1  # city 0 is the fixed start, so only m stops to decide
    cities = range(1, n)  # the values a stop can take. used for BOTH domains and cost lists so they line up
    big = 1000 * n  # "infinite" cost, bigger than any real tour
    cfn = pytoulbar2.CFN(big)

    # one variable per stop, value = which city (1..n-1)
    for p in range(m):
        cfn.AddVariable(f"pos{p + 1}", cities)

    # soft: home -> first stop, and last stop -> home
    cfn.AddFunction([0], [dist[0][c] for c in cities])
    cfn.AddFunction([m - 1], [dist[c][0] for c in cities])

    # soft: distance between each pair of consecutive stops
    for p in range(m - 1):
        cfn.AddFunction([p, p + 1], [dist[a][b] for a in cities for b in cities])

    # hard: no city twice
    cfn.AddGlobalFunction(list(range(m)), "salldiff", "var", big)

    sol, cost, _ = cfn.Solve()
    return [0] + [int(v) for v in sol] + [0], cost


if __name__ == "__main__":
    # tries it on sample TSP with 8 cities
    dist = make_instance(8)
    tour, cost = solve_toulbar2(dist)
    print("toulbar2:", tour, cost)
