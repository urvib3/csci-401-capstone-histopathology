# Minimum Spanning Tree as a WCSP [Aakanksha]
#
# MST: a subset of edges in a connected, edge-weighted undirected graph that connects all
#  vertices with the minimum possible total edge weight and no cycles
#
# WCSP signature (parent-variable encoding):
#   variables:  n, one parent variable for each vertex
#   domains:    each parent can be any vertex 0..n-1
#   hard constraint:    all parent choices together have to form one valid spanning tree
#   cost function:    choosing u as the parent of v costs the weight of edge (u, v)
#   arity:      unary costs for parent choices + one global MST constraint over n variables
#   treewidth:  n-1 with this encoding, since the global MST constraint connects all variables
#
# Local search: start with any spanning tree. Add an edge that is not in the tree,
#   which creates a cycle, then remove an edge from that cycle. Keep the change
#   if the new tree has a lower total cost.
#
# A*: build the tree edge by edge without creating cycles.
#   g = cost of edges picked so far
#   h = optimistic lower bound on the cost needed to connect the remaining components
#   A* can be formulated for MST, but Kruskal or Prim is much more natural for MST.
#
# Other standard polynomial-time algorithms for finding an MST include Prim's,
# Kruskal's, and Borůvka's

import pytoulbar2

# Small example graph that Toulbar2 will solve.
# The numbers 0, 1, 2, 3 represent vertices A, B, C, D.
vertices = ["A", "B", "C", "D"]

# Each dictionary key is an undirected edge (u, v),
# and the value is the weight/cost of choosing that edge.
#
# Example:
#   (0, 1): 1 means edge A-B has weight 1.
edges = {
    (0, 1): 1,  # A-B
    (1, 2): 2,  # B-C
    (0, 3): 3,  # A-D
    (0, 2): 4,  # A-C
    (1, 3): 5,  # B-D
    (2, 3): 6,  # C-D
}



def solve_toulbar2(vertices, edges):
    # Number of vertices in the graph.
    n = len(vertices)

    # Toulbar2 needs a sufficiently large cost to represent an invalid assignment.
    # Since this is larger than the sum of every real edge weight,
    # Toulbar2 will avoid these assignments whenever a valid tree exists.
    big = sum(edges.values()) + 1

    # Create the WCSP / Cost Function Network.
    # Toulbar2 will search for the assignment with the minimum total cost.
    cfn = pytoulbar2.CFN(big)

    # Create one variable for each graph vertex.
    #
    # parent_A, parent_B, parent_C, parent_D
    #
    # The value assigned to parent_v tells us which vertex is v's parent.
    #
    # Example:
    #   parent_C = 1
    # means that vertex B (index 1) is the parent of C.
    for v in range(n):
        cfn.AddVariable(f"parent_{vertices[v]}", range(n))

    # Choose A (vertex 0) as the root of the spanning tree.
    #
    # In the parent-variable encoding, the root points to itself.
    root = 0  # A

    # Add the cost of choosing each possible parent.
    #
    # These are unary cost functions because the cost depends only
    # on the value assigned to one parent variable.
    for v in range(n):
        # costs[parent] will store the cost of assigning
        # that vertex as the parent of v.
        costs = []

        for parent in range(n):

            # The root must point to itself.
            #
            # parent_A = A has cost 0.
            # Any other parent for A is invalid.
            if v == root:
                cost = 0 if parent == root else big

            # A non-root vertex cannot point to itself.
            #
            # For example, parent_B = B would not help connect B
            # to the rest of the spanning tree.
            elif parent == v:
                cost = big

            else:
                # The graph is undirected, so always store/check the edge
                # using the smaller vertex index first.
                #
                # For example, B-A and A-B both become (0, 1).
                edge = (min(v, parent), max(v, parent))

                # If the edge exists in the original graph,
                # choosing this parent has the same cost as the edge weight.
                if edge in edges:
                    cost = edges[edge]

                # If there is no edge between these vertices,
                # this parent assignment is invalid.
                else:
                    cost = big

        # Add the unary cost function for vertex v.
        #
        # Example for B:
        #   parent_B = A -> cost 1
        #   parent_B = B -> invalid
        #   parent_B = C -> cost 2
        #   parent_B = D -> cost 5

            costs.append(cost)

        cfn.AddFunction([v], costs)

    # The unary costs alone are not enough to guarantee a spanning tree.
    #
    # For example, vertices could otherwise choose cheap parents
    # in a way that creates a cycle.
    #
    # Toulbar2's global MST constraint makes sure that all parent
    # assignments together represent exactly one valid spanning tree.
    cfn.AddGlobalFunction(list(range(n)), "MST")

    # Ask Toulbar2 to find the minimum-cost valid assignment.
    #
    # solution = chosen parent for each vertex
    # cost     = total weight of the resulting MST

    solution, cost, _ = cfn.Solve()

    return solution, cost


if __name__ == "__main__":

     # Solve the example graph.
    solution, cost = solve_toulbar2(vertices, edges)

    print("toulbar2 MST:")
    print("total cost:", cost)

    # Convert the numeric parent indices back into vertex names
    # so the result is easier to read.
    for v in range(len(vertices)):
        parent = int(solution[v])
        print(f"{vertices[v]} -> {vertices[parent]}")


# EXAMPLE OUTPUT:

# toulbar2 MST:
# total cost: 6.0
# A -> A
# B -> A
# C -> B
# D -> A