CUDA_INSTALL_PATH?=/usr/local/cuda/
NVCC=$(CUDA_INSTALL_PATH)/bin/nvcc

SRC=divergent_kernel.cu

all: OBJ

OBJ: $(SRC:%.cu=%.o)

%.o: benchmark/%.cu
	@$(NVCC) $< -o $@

clean:
	@rm *.o _cu* gpgpusim_power* gpgpu_inst* -f
