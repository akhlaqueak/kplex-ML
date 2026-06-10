ROOT := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

all : 
	g++ -std=c++11 -O3 -w $(ROOT)main.cpp -o $(ROOT)UMKPEX  -DNO_PROGRESS_BAR -DNDEBUG
prune-diag:
	g++ -std=c++11 -O3 -w $(ROOT)main.cpp -o $(ROOT)UMKPEX_diag -DNO_PROGRESS_BAR -DNDEBUG -DPRUNE_DIAG
debug: 
	g++ -std=c++11 -g -gdwarf-4 -O0 -w $(ROOT)main.cpp -o $(ROOT)UMKPEX  -DNO_PROGRESS_BAR 
clean:
	rm -f $(ROOT)UMKPEX $(ROOT)UMKPEX_diag
