import random

import pytoulbar2

MAX_VERTICES = 10_000
MAX_WEIGHT = 100
EDGE_PROB = 0.2

def make_instance():
	n = random.randint(3,MAX_VERTICES)
	
	# initialize weights
	vertices = [(random.randint(1,MAX_WEIGHT)) for _ in range(n)]
	edges = [[0]*n for _ in range(n)]

	# adjacency matrix
	for i in range(n):
		for j in range(i+1, n):
			edges[i][j] = edges[j][i] = (1 if random.random() < EDGE_PROB else 0)

	return (vertices, edges)


def verify_solution(vertices, edges, sol, cost):
	chosen = [i for i, v in enumerate(sol) if v == 1]
	ok = True

	# independence: no edge between any two chosen vertices
	for a in range(len(chosen)):
		for b in range(a+1, len(chosen)):
			i, j = chosen[a], chosen[b]
			if edges[i][j]:
				print(f"VERIFY FAIL: chosen vertices {i} and {j} share an edge")
				ok = False

	# weight: reported cost should be minus the sum of chosen weights
	total = sum(vertices[i] for i in chosen)
	if total != -cost:
		print(f"VERIFY FAIL: chosen weight {total} != reported weight {-cost}")
		ok = False

	# maximality (sanity only, not optimality): no unchosen vertex can be added freely
	chosen_set = set(chosen)
	for v in range(len(vertices)):
		if v not in chosen_set and not any(edges[v][u] for u in chosen):
			print(f"VERIFY FAIL: vertex {v} could be added without conflict, so solution is not optimal")
			ok = False

	print("Verification:", "PASSED" if ok else "FAILED")
	return ok


def make_wcsp(vertices, edges):
	
	wcsp = pytoulbar2.CFN(1000000, resolution = 0) # basically infinity

	x = [wcsp.AddVariable(f"x{i}",[0,1]) for i in range(len(vertices))] # yes or no on each vertex

	for i, w in enumerate(vertices):
		wcsp.AddFunction([x[i]], [0,-w]) # -w cost if chosen, good to add weight
	
	for i in range(len(vertices)):
		for j in range(i+1, len(vertices)):
			if(edges[i][j]):
				wcsp.AddFunction([x[i],x[j]],[0,0,0,wcsp.Top]) # constrain adjacency
	
	return wcsp
	


def run_solver():
	vertices, edges = make_instance()
	solver = make_wcsp(vertices, edges)
	solution = solver.Solve(showSolutions=0, timeLimit=30)

	if solution is None:
		solution = ([],0,0)
	return solution, vertices, edges


if __name__ == "__main__":
	print("running...")
	(sol, cost, nsol), vertices, edges = run_solver()
	print(f"Solution: {[i for i, v in enumerate(sol) if v == 1]} || Cost: {-1*cost} || No. Solutions: {nsol}")
	verify_solution(vertices, edges, sol, cost)

	
	
