FN := database

CUDA_HOME := /usr/local/cuda
CUDA_LIBS := -lcuda
CUDA_LIBS += -lcudart

NVCC := $(CUDA_HOME)/bin/nvcc

CPP_FLAGS := -std=c++11 -O3
NVCC_FLAGS := -arch=sm_35
LINK_FLAGS := -L$(CUDA_HOME)/lib64 $(CUDA_LIBS)
LINK_FLAGS += -lpthread
LINK_FLAGS += -lboost_thread
LINK_FLAGS += -lboost_system

all: main.cpp $(FN).o
	g++ $(CPP_FLAGS) -g3 -o dbtest $^ $(LINK_FLAGS)
	rm database.o

clean : 
	rm dbtest

$(FN).o : $(FN).cu $(FN).h
	$(NVCC) $(CPP_FLAGS) $(NVCC_FLAGS) -I$(CUDA_HOME)/include/ -c -o $@ $<
