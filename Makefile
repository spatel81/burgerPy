NVCC = $(shell which nvcc)
NVCC_FLAGS = -g -G -Xcompiler -Wall

#CONDA_SOFT_DIR = /lus/theta-fs0/software/thetagpu/conda/2021-11-30/mconda3

#* SET THE PATH TO YOUR BASE CONDA DIRECTORY *#
#CONDA_SOFT_DIR = /PATH/TO/YOUR/LOCAL/CONDA/DIR

#Example
#CONDA_SOFT_DIR = /lus/theta-fs0/software/thetagpu/conda/2021-11-30/mconda3

INCLUDES  = -I$(CONDA_SOFT_DIR)/lib/python3.8/site-packages/tensorflow/include/external/local_config_python/numpy_include
INCLUDES += -I$(CONDA_SOFT_DIR)/include
INCLUDES += -I$(CONDA_SOFT_DIR)/include/python3.8

LIBRARIES = -L$(CONDA_SOFT_DIR)/lib -lpython3.8

all: main.exe

main.exe: main.o 
	$(NVCC) $(LIBRARIES) $^ -o $@

main.o: main.cu kernel.h
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f *.o *.exe
