LLVM_PROJECT=${PWD}/vendor/llvm-project
LLVM_PROJECT_BUILD=${LLVM_PROJECT}/build

BENCHMARK=${PWD}/vendor/benchmark
BENCHMARK_BUILD_DIR=${BENCHMARK}/build

.PHONY: all
all: venv llvm benchmark

.PHONY: submodules
submodules:
	git submodule update --init --recursive

.PHONY: llvm
llvm: submodules
	mkdir -p ${LLVM_PROJECT_BUILD}

	cd ${LLVM_PROJECT_BUILD} && cmake -G Ninja ../llvm --fresh \
		-DLLVM_ENABLE_PROJECTS="mlir;clang" \
		-DLLVM_BUILD_EXAMPLES=ON \
		-DLLVM_TARGETS_TO_BUILD="Native" \
		-DCMAKE_BUILD_TYPE=Release \
		-DLLVM_ENABLE_ASSERTIONS=ON \
		-DLLVM_FORCE_ENABLE_STATS=ON \
		-DLLVM_CCACHE_BUILD=ON \
		-DLLVM_USE_SANITIZER="Address;Undefined" \
		-DCMAKE_C_COMPILER=clang \
		-DCMAKE_CXX_COMPILER=clang++ \
		-DLLVM_ENABLE_LLD=ON

	cd ${LLVM_PROJECT_BUILD} && cmake --build . --target check-mlir

.PHONY: benchmark
benchmark: submodules
	mkdir -p ${BENCHMARK_BUILD_DIR}
	cd ${BENCHMARK_BUILD_DIR} && cmake -DBENCHMARK_DOWNLOAD_DEPENDENCIES=on -DCMAKE_BUILD_TYPE=Release ../
	cd ${BENCHMARK_BUILD_DIR} && cmake --build . --config Release

.PHONY: venv
venv:
	uv venv
	uv sync --all-extras

.PHONY: clean
clean: clean-mlir clean-benchmark clean-venv

.PHONY: clean-mlir
clean-mlir:
	rm -rf ${LLVM_PROJECT_BUILD}

.PHONY: clean-benchmark
clean-benchmark:
	rm -rf ${BENCHMARK_BUILD_DIR}

.PHONY: clean-venv
clean-venv:
	rm -rf .venv
