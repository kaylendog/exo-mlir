.ONESHELL:

.PHONY: clean
clean:
	rm -rf build

.PHONY: env
env:
	uv venv
	uv sync --all-extras
	@. .venv/bin/activate

BENCHMARKS := $(wildcard benchmarks/*.py)

compile-submodule-benchmark:
	mkdir -p build
	cd submodules/benchmark
	cmake -E make_directory build
	cmake -DCMAKE_BUILD_TYPE=Release -DBENCHMARK_ENABLE_TESTING=false -S . -B "build"
	cmake --build "build" --config Release

compile-benchmarks: env compile-submodule-benchmark $(BENCHMARKS) benchmarks/benchmark.cc
	mkdir -p build
	for bench in $(BENCHMARKS); do \
		exocc -o build/benchmark --stem $$(basename $$bench .py) $$bench; \
	done
	for src in $(wildcard build/benchmark/*.c); do \
		gcc -std=c17 -O3 -Wall -Wextra -mavx2 -march=x86-64-v3 -I build/benchmark -c $$src -o build/benchmark/$$(basename $$src .c).o; \
	done
	g++ -std=c++17 -O3 -Wall -Wextra \
		-I build/benchmark \
		-I submodules/benchmark/include \
		-c benchmarks/benchmark.cc \
		-o build/benchmark/benchmark.o
	g++ -o \
		build/benchmark/benchmark \
		build/benchmark/*.o \
		-L submodules/benchmark/build/src \
		-lbenchmark -lpthread -lbenchmark_main -lstdc++ -lm
	chmod +x build/benchmark/benchmark

benchmark: compile-benchmarks
	build/benchmark/benchmark
