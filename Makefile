
NVCC        = nvcc
ifeq (,$(shell which nvprof))
NVCC_FLAGS  = -O3 -I/usr/local/cuda/include 
else
NVCC_FLAGS  = -O3 -I/usr/local/cuda/include --std=c++03
endif
LD_FLAGS    = -lcublas -lcudart -L/usr/local/cuda/lib64
EXE	        = sgemm-tiled
OBJ	        = main.o support.o

default: $(EXE)

main.o: main.cu kernel.cu support.h
	$(NVCC) -c -o $@ main.cu $(NVCC_FLAGS)

support.o: support.cu support.h
	$(NVCC) -c -o $@ support.cu $(NVCC_FLAGS)

$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE) $(LD_FLAGS)

clean:
	rm -rf *.o $(EXE)
