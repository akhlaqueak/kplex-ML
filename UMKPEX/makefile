all : 
	g++ -std=c++11 -O3 -w main.cpp -o UMKPEX  -DNO_PROGRESS_BAR -DNDEBUG
	g++ -std=c++11 -O3 -w main.cpp -o UMKPEX-CTCP  -DNO_PROGRESS_BAR -DNDEBUG -Denable_CTCP
debug: 
	g++ -std=c++11 -g -w main.cpp -o kPEX  -DNO_PROGRESS_BAR 

clean:
	rm kPEX