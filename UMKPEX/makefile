all :
	g++ -std=c++11 -O3 -w main.cpp -o UMKPEX -pthread -DNO_PROGRESS_BAR -DNDEBUG
debug:
	g++ -std=c++11 -g -gdwarf-4 -O0 -w main.cpp -o UMKPEX -pthread -DNO_PROGRESS_BAR

# Build with ML-driven racing (-ml-race) enabled: requires libxgboost with its C API
# headers/lib installed, e.g. built from https://github.com/dmlc/xgboost with
# `cmake -DBUILD_STATIC_LIB=ON` or installed to a prefix. Point XGBOOST_ROOT at that
# prefix (it must contain include/xgboost/c_api.h and lib/libxgboost.*).
XGBOOST_ROOT ?= /usr/local
ml:
	g++ -std=c++14 -O3 -w main.cpp -o UMKPEX_ml -pthread -DNO_PROGRESS_BAR -DNDEBUG -DUSE_ML_PREDICTOR \
	    -I$(XGBOOST_ROOT)/include -L$(XGBOOST_ROOT)/lib -lxgboost -Wl,-rpath,$(XGBOOST_ROOT)/lib

clean:
	rm -f UMKPEX UMKPEX_ml