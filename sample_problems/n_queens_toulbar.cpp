


#include <toulbar2lib.hpp>
#include <iostream>




using namespace std;


int main(int argc, char** argv) {
	if (argc == 1) {
		cout << "Enter n with arg" << endl;
		exit(0);

	}
	int n = atoi(argv[1]);
	cout << "Solving for n=" <<  n << endl;
	tb2init();
	



	return 0;
}
