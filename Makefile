.ONESHELL:

COMMAND := $(firstword $(MAKECMDGOALS))
ARGS := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))

BUILD_DIR := build
BIN := $(BUILD_DIR)/bin

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BIN):
	mkdir -p $(BIN)
	
.PHONY: clean
clean:
	rm -rf build submodules/benchmark/build

.PHONY: env
env:
	uv venv
	uv sync --all-extras
	. .venv/bin/activate

# --- Submodules ---

submodules/benchmark:
	git submodule update --init --recursive

submodules/benchmark/build: submodules/benchmark
	cmake -S submodules/benchmark -B submodules/benchmark/build \
		-DCMAKE_BUILD_TYPE=Release -DBENCHMARK_ENABLE_TESTING=false
	cmake --build submodules/benchmark/build --config Release

# --- Benchmarking ---

BENCHMARK_SRC_DIR := benchmarks
BENCHMARK_BUILD_DIR := $(BUILD_DIR)/benchmark
BENCHMARK_BIN := $(BIN)/benchmark
BENCHMARK_PROC_PY := $(wildcard $(BENCHMARK_SRC_DIR)/*.py)
BENCHMARK_PROC_OBJ := $(patsubst $(BENCHMARK_SRC_DIR)/%.py, $(BENCHMARK_BUILD_DIR)/%.o, $(BENCHMARK_PROC_PY))

$(BENCHMARK_BUILD_DIR):
	mkdir -p $(BENCHMARK_BUILD_DIR)

$(BENCHMARK_BUILD_DIR)/%.c: $(BENCHMARK_SRC_DIR)/%.py | $(BENCHMARK_BUILD_DIR)
	exocc -o $(BENCHMARK_BUILD_DIR) --stem $(basename $(notdir $<)) $<

$(BENCHMARK_BUILD_DIR)/%.o: $(BENCHMARK_BUILD_DIR)/%.c
	clang -std=c17 -O3 -Wall -Wextra -I $(BENCHMARK_BUILD_DIR) -c $< -o $@

$(BENCHMARK_BUILD_DIR)/avx2_matmul.o: $(BENCHMARK_BUILD_DIR)/avx2_matmul.c
	clang -std=c17 -O3 -Wall -Wextra -I $(BENCHMARK_BUILD_DIR) -c $< -o $@ -mavx2 -march=x86-64-v3

$(BENCHMARK_BIN): $(BENCHMARK_PROC_OBJ) submodules/benchmark/build $(BENCHMARK_SRC_DIR)/benchmark.cc | $(BIN)
	clang++ -std=c++17 -O3 -Wall -Wextra \
		-I $(BENCHMARK_BUILD_DIR) -I submodules/benchmark/include \
		$(BENCHMARK_PROC_OBJ) $(BENCHMARK_SRC_DIR)/benchmark.cc \
		-o $(BENCHMARK_BIN) \
		-L submodules/benchmark/build/src -lbenchmark -lpthread -lbenchmark_main -lm

.PHONY: benchmark
benchmark: $(BENCHMARK_BIN)
	$(BENCHMARK_BIN) $(ARGS)
