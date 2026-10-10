# Build every benchmark into build/.   make            (T4 / Turing default)
#                                      make ARCH=sm_80  (A100), sm_86 (RTX 30xx), sm_89 (L4), sm_90 (H100)
NVCC  ?= nvcc
ARCH  ?= sm_75
FLAGS := -O3 -arch=$(ARCH) -lineinfo -std=c++17
DEPS  := common/common.h common/reduce.cuh

BINS := build/vecadd_bench build/reduction build/transpose build/gemm \
        build/softmax build/layernorm build/attention

all: $(BINS)

build:
	mkdir -p build

build/vecadd_bench: kernels/01_vecadd/vecadd_bench.cu $(DEPS) | build
	$(NVCC) $(FLAGS) $< -o $@
build/reduction: kernels/02_reduction/reduction.cu $(DEPS) | build
	$(NVCC) $(FLAGS) $< -o $@
build/transpose: kernels/03_transpose/transpose.cu $(DEPS) | build
	$(NVCC) $(FLAGS) $< -o $@
build/gemm: kernels/04_gemm/gemm.cu $(DEPS) | build
	$(NVCC) $(FLAGS) $< -o $@ -lcublas
build/softmax: kernels/05_softmax/softmax.cu $(DEPS) | build
	$(NVCC) $(FLAGS) $< -o $@
build/layernorm: kernels/06_layernorm/layernorm.cu $(DEPS) | build
	$(NVCC) $(FLAGS) $< -o $@
build/attention: kernels/07_attention/attention.cu $(DEPS) | build
	$(NVCC) $(FLAGS) $< -o $@ -lcublas

clean:
	rm -rf build

.PHONY: all clean
