// #include "utils/tb2btlist.hpp" 
#include "core/tb2types.hpp"
// #include "utils/tb2system.hpp"
// #include "search/tb2solver.hpp" 
#include <toulbar2lib.hpp>
#include <iostream>
// #include <vector>
#include <string>
#include <cstdlib>


using namespace std;


int main(int argc, char** argv) {
	if (argc == 1) {
		cout << "Enter n with arg" << endl;
		exit(0);

	}
	int n = atoi(argv[1]);
	cout << "Solving for n=" <<  n << endl;
	tb2init();

	int top = n* n+ 1; // this the upper bound

	// WeightedCSP* problem = WeightedCSP::makeWeightedCSP(top);
	WeightedCSPSolver* solver = WeightedCSPSolver::makeWeightedCSPSolver(top);
	WeightedCSP* problem = solver->getWCSP();

	for (int i =0 ; i < n; ++i) {
		// each variable is a queen, Queen #i, 0 <= i < n
		// the domain of each queen is the assigned row
		// vector<string> row_domain;
		// for (int r = 0; r < n; ++r) {
		// 	row_domain.push_back("row" + to_string(r+1));
		// }
		problem->makeEnumeratedVariable("Q" + to_string(i+1), 0, n-1);
	}


	for (int i =0; i < n; ++i) {
		for (int j =i+1; j < n; ++j) {
			// constraints
			// two queens cant be on same row 
			vector<Cost> row_constraints; // Cost is a long long
			for (int a = 0; a < n; ++a) {
				for (int b = 0; b < n; ++b) {
					if (a != b)row_constraints.push_back(0);
					else
						// penalize (cost)
						row_constraints.push_back(top);
				}
			}
			problem->postBinaryConstraint(i, j, row_constraints);

			// two queens cant be on same upper diagonal
			vector<Cost> upper_diag_constraints;
			for (int a = 0; a < n; ++a ) {
				for (int b = 0; b < n; ++b) {
					if (a+i != b + j) upper_diag_constraints.push_back(0);
					else
						upper_diag_constraints.push_back(top);
				}
			}
			problem->postBinaryConstraint(i, j, upper_diag_constraints);


			// two queens cant be on same lower diagonal
			vector<Cost> lower_diag_constraints;
			for (int a = 0; a < n; ++a ) {
				for (int b = 0; b < n; ++b) {
					if (a - i != b-j) lower_diag_constraints.push_back(0);
					else
						lower_diag_constraints.push_back(top);
				}
			}
			problem->postBinaryConstraint(i, j, lower_diag_constraints);

		}


	}

	// random unary costs
	for (int i = 0; i < n; ++i) {
		vector<Cost> unary_constraints;
		for (int j = 0; j < n; ++j) unary_constraints.push_back(rand() % n + 1);
		problem->postUnaryConstraint(i, unary_constraints);
	}

	int arity = n;
	int* scope_index = new int[arity];
	for (int i =0; i < arity; ++i) {
		scope_index[i] = i;
	}
	problem->postWAllDiff(scope_index, arity, "var", "flow", 1000);




	// solve
	problem->sortConstraints();
	bool res = solver->solve();

	if (res) cout << "Optimal soln found with cost: " << problem->getSolutionCost() << endl;
	else cout << "coldnt find an optimal soln" << endl;



	// cleanup
	delete problem;
	delete[] scope_index;

	return 0;
}
