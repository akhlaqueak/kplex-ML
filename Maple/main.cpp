#include "Graph.h"
#include "Utility.h"
#include "Timer.h"

using namespace std;
#define LEN_LIMIT (1<<10)
char filename[LEN_LIMIT];

int main(int argc, char *argv[]) {
	printf("\n-----------------------------------------------------------------------------------------\n");
	if (argc >= 3) {
		strncpy(filename, argv[1], LEN_LIMIT);
		int k = atoi(argv[2]);
		bool searcher_enableCTCP = true;
		bool searcher_enableUBR1 = true;
		for (int i = 3; i < argc; ++i) {
			string arg = argv[i];
			if (arg == "--disable-ctcp") searcher_enableCTCP = false;
			else if (arg == "--disable-ubr1") searcher_enableUBR1 = false;
			else if (arg == "--enable-ctcp") searcher_enableCTCP = true;
			else if (arg == "--enable-ubr1") searcher_enableUBR1 = true;
			else {
				printf("[usage]: exe file k [--disable-ctcp] [--disable-ubr1] [--enable-ctcp] [--enable-ubr1]\n");
				printf("Unknown option: %s\n", argv[i]);
				printf("-----------------------------------------------------------------------------------------\n\n");
				return 0;
			}
		}
		Graph *graph = new Graph(filename, k);
		graph->searcher_enableCTCP = searcher_enableCTCP;
		graph->searcher_enableUBR1 = searcher_enableUBR1;
		printf("MSearcher enableCTCP: %s\n", graph->searcher_enableCTCP ? "true" : "false");
		printf("MSearcher enableUBR1: %s\n", graph->searcher_enableUBR1 ? "true" : "false");
		graph->read();
		graph->search();
		graph->write();
		// delete graph; // there are some bugs in releasing memory
	}
	else printf("[usage]: exe file k [--disable-ctcp] [--disable-ubr1] [--enable-ctcp] [--enable-ubr1]\n");
	printf("-----------------------------------------------------------------------------------------\n\n");
}
