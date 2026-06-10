ROOT := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

all : 
	g++ -flto -Ofast -DNDEBUG -march=native -o $(ROOT)Maple $(ROOT)main.cpp $(ROOT)Graph.cpp -w -std=c++17
prune-diag:
	g++ -flto -Ofast -DNDEBUG -march=native -DPRUNE_DIAG -o $(ROOT)Maple_diag $(ROOT)main.cpp $(ROOT)Graph.cpp -w -std=c++17
clean:
	rm -f $(ROOT)Maple $(ROOT)Maple_diag
