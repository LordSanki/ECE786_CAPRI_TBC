NVCC=/home/sanki/CUDA_env/cuda/bin/nvcc

SRC=vector_addition.cu

all: OBJ

OBJ: $(SRC:%.cu=%.o)

%.o: benchmark/%.cu
	@$(NVCC) $< -o $@

clean:
	@rm *.o _cu* gpgpusim_power* gpgpu_inst* -f
