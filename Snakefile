import platform

HOST_ARCH = platform.machine()

configfile: "config.yaml"

rule all:
    input:
        "bin/benchmark"

rule clean:
    shell:
        "rm -rf build submodules/benchmark/build"

rule env:
    shell:
        """
        uv venv
        uv sync --all-extras
        source .venv/bin/activate
        """

rule submodules:
    output:
        "submodules"
    shell:
        "git submodule update --init --recursive"

rule libbenchmark:
    input:
        "submodules/benchmark"
    output:
        "submodules/benchmark/build"
    shell:
        """
        cmake -S submodules/benchmark -B submodules/benchmark/build -DCMAKE_BUILD_TYPE=Release -DBENCHMARK_ENABLE_TESTING=OFF
        cmake --build submodules/benchmark/build --config Release
        """

rule mkdirs:
    shell:
        f"mkdir -p build/common build/{HOST_ARCH} bin"

rule compile:
    input:
        "build/{arch}/{stem}.c"
    output:
        "build/{arch}/{stem}.o"
    params:
        arch = "{arch}",
        stem = "{stem}"
    shell:
        "clang -c {input} -o {output} {config[arch_flags][params.arch]} {config[cflags]}"

rule exo:
    input:
        "benchmarks/{arch}/{stem}.py"
    output:
        "build/{arch}/{stem}.c"
        "build/{arch}/{stem}.h"
    params:
        arch = "{arch}",
        stem = "{stem}"
    shell:
        "exocc -o build/{params.arch} --stem {params.stem} {input}"

# compute benchmarks we can run on this host
ARCH_SRC = glob_wildcards(f"benchmarks/{HOST_ARCH}/{{stem}}.py")
COMMON_SRC = glob_wildcards("benchmarks/common/{stem}.py")

rule benchmark:
    input:
        arch=expand(f"build/{HOST_ARCH}/{{stem}}.o", stem=ARCH_SRC.stem),
        common=expand("build/common/{stem}.o", stem=COMMON_SRC.stem),
        libbenchmark="submodules/benchmark/build/src",
        src="benchmarks/benchmark.cc"
    output:
        "bin/benchmark"
    shell:
        "clang++ -o {output} {input.arch} {input.common} {input.src} {config[cxxflags]}"
        " -I build/{HOST_ARCH} -I build/common"
        " -I submodules/benchmark/include -L {input.libbenchmark} -lbenchmark -lpthread"