#include "Graph.h"
#include "Utility.h"
#include "Timer.h"
#include "popl.hpp"

using namespace std;
using namespace popl;

void print_usage() {
	printf("Example usage: ./kPlexT -g path_to_graph -k 3 -o -b\n");
	printf("           or: ./kPlexT path_to_graph 3\n");
	printf("Traversal:    --forward (default: backward)\n");
}

int main(int argc, char *argv[]) {
#ifndef NDEBUG
	printf("**** kPlexT (Debug) build at %s %s ***\n", __TIME__, __DATE__);
	printf("!!! You may want to define NDEBUG in Utility.h to get better performance!\n");
#else
	printf("**** kPlexT (Release) build at %s %s ***\n", __TIME__, __DATE__);
#endif

	bool output = false;
	bool binary_input = false;
	bool mode = true;
	string alg;
	string graph_path;
	int k_value = -1;

	bool positional_mode = false;
	if(argc >= 3 && argv[1][0] != '-' && argv[2][0] != '-') {
		positional_mode = true;
		graph_path = argv[1];
		k_value = atoi(argv[2]);
	}

	OptionParser op("Allowed options");
	auto help_option = op.add<Switch>("h", "help", "\'produce help message\'");
	auto graph_option = op.add<Value<string>>("g", "graph", "\'path to input graph file\'");
	auto alg_option = op.add<Value<string>>("a", "alg", "\'algorithm name\' (degen | exact | verify)", "exact", &alg);
	auto k_option = op.add<Value<int>>("k", "k", "\'the value of k for k-plex\'");
	op.add<Switch>("", "forward", "\'traverse peel_sequence forward during exact search\'", &mode);
	op.add<Switch>("o", "output", "\'write the kplex to ./kplex.txt\'", &output);
	op.add<Switch>("b", "binary", "\'read the input graph from binary files b_adj.bin and b_degree.bin\'", &binary_input);

	op.parse(argc, argv);

	if(help_option->is_set()||argc <= 1) {
		cout << op << endl;
		if(argc <= 1) {
			print_usage();
			return 0;
		}
	}
	if(!positional_mode && !graph_option->is_set()) {
		printf("!!! The argument -g is required! Exit !!!\n");
		return 0;
	}
	if(!positional_mode && !k_option->is_set()) {
		printf("!!! The argument -k is required! Exit !!!\n");
		return 0;
	}
	if(positional_mode && k_value <= 0) {
		printf("!!! The positional k argument must be a positive integer! Exit !!!\n");
		return 0;
	}

	if(!positional_mode) {
		graph_path = graph_option->value();
		k_value = k_option->value();
	}
	if(graph_path.size() >= 4 && graph_path.substr(graph_path.size() - 4) == ".bin")
		binary_input = true;

	Graph *graph = new Graph(graph_path.c_str(), k_value);
	if(binary_input) graph->read_graph_binary();
	else graph->read_graph();

#ifndef NDEBUG
	printf("\t*** Finished reading graph\n");
#endif

	if(strcmp(alg.c_str(), "degen") == 0) graph->kPlex_degen(); //degeneracy-based k-plex
	else if(strcmp(alg.c_str(), "exact") == 0) graph->kPlex_exact(!mode);
	else if(strcmp(alg.c_str(), "verify") == 0) graph->verify_kplex();
	else print_usage();

	graph->output_one_kplex();

	delete graph;

	printf("\n");
	return 0;
}
